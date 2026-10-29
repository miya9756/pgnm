"""SIDD Validation scores of the trained denoisers and Tab. 2.

    python -m experiments.denoise evaluate --source pgnm   # -> runs/denoise/<source>/sidd_val.json
    python -m experiments.denoise table                    # Tab. 2 and paired differences

Each of the 1,280 official 256x256 Validation blocks (40 images) is denoised
whole: output clip(f(noisy/256), 0, 1) * 256, rounded half up and clipped to
uint8. PSNR (peak 255) and SSIM are computed per block on uint8 values and
averaged over blocks. SSIM is scikit-image's structural_similarity with
channel_axis=-1, data_range=255 and its defaults (7x7 uniform window, sample
covariance, 3-pixel border crop), NM-FlowGAN's evaluation function, written out
here in float64. Paired differences between two denoisers resample the eight
physical scenes of SIDD Validation (100,000 draws, seed 11).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from .. import config
from .data import CLASSES
from .noise import SOURCES

SUBSETS = dict(all="All 40 images", S19="19 in training classes", S21="21 in other classes")
DRAWS, BOOTSTRAP_SEED = 100_000, 11


def block_labels():
    """Scene of each of the 1,280 blocks (32 per image) and the subset masks."""
    labels = config.dataset("sidd")["validation"]
    trained = np.repeat(["%s_%d" % (label["camera"], label["iso"]) in CLASSES for label in labels], 32)
    return dict(scenes=np.repeat([label["scene"] for label in labels], 32),
                subsets=dict(all=np.ones(1280, bool), S19=trained, S21=~trained))


def validation_blocks():
    """Noisy and clean uint8 blocks, (1280, 256, 256, 3) each."""
    from .. import sidd
    noisy, clean = sidd.validation_blocks()
    return noisy.reshape(-1, 256, 256, 3), clean.reshape(-1, 256, 256, 3)


def psnr(output, clean):
    error = np.asarray(output, np.float64) - np.asarray(clean, np.float64)
    return float(10.0 * np.log10(255.0 ** 2 / np.mean(error * error)))


def ssim(output, clean, window=7, k1=0.01, k2=0.03, peak=255.0):
    first, second = np.asarray(output, np.float64), np.asarray(clean, np.float64)
    pad, count = (window - 1) // 2, window ** 2
    norm, c1, c2 = count / (count - 1.0), (k1 * peak) ** 2, (k2 * peak) ** 2
    values = []
    for channel in range(first.shape[2]):
        x, y = first[..., channel], second[..., channel]
        mean_x, mean_y = uniform_filter(x, window), uniform_filter(y, window)
        var_x = norm * (uniform_filter(x * x, window) - mean_x * mean_x)
        var_y = norm * (uniform_filter(y * y, window) - mean_y * mean_y)
        cov_xy = norm * (uniform_filter(x * y, window) - mean_x * mean_y)
        index = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
                 / ((mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2)))
        values.append(index[pad:-pad, pad:-pad].mean(dtype=np.float64))
    return float(np.mean(values))


def denoise(model, blocks, device, batch=16):
    """uint8 (N, H, W, 3) -> uint8 denoised blocks."""
    import torch
    out = []
    with torch.no_grad():
        for start in range(0, len(blocks), batch):
            x = torch.from_numpy(blocks[start:start + batch]).to(device).permute(0, 3, 1, 2).contiguous().float()
            y = model(x / 256.0).clamp(0, 1) * 256.0
            out.append(torch.floor(y + 0.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


def evaluate(source, checkpoint=None, device="cuda", out=None):
    import torch
    from .train import DnCNN, full_float32
    full_float32()
    checkpoint = Path(checkpoint or config.runs("denoise", source, "model.pt"))
    model = DnCNN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    noisy, clean = validation_blocks()
    outputs = denoise(model, noisy, device)
    result = dict(source=source, psnr=[psnr(o, c) for o, c in zip(outputs, clean)],
                  ssim=[ssim(o, c) for o, c in zip(outputs, clean)])
    target = Path(out or config.runs("denoise", source, "sidd_val.json"))
    config.write_json(target, result)
    print(SOURCES[source], "PSNR %.2f SSIM %.3f" % (np.mean(result["psnr"]), np.mean(result["ssim"])), flush=True)
    return result


def scene_interval(differences, scenes, draws=DRAWS, seed=BOOTSTRAP_SEED):
    """Block-mean difference and its 95% interval from resampling whole scenes."""
    names, inverse = np.unique(np.asarray(scenes), return_inverse=True)
    sums = np.bincount(inverse, weights=differences, minlength=len(names))
    sizes = np.bincount(inverse, minlength=len(names)).astype(float)
    counts = np.random.default_rng(seed).multinomial(len(names), np.full(len(names), 1 / len(names)), size=draws)
    low, high = np.percentile((counts @ sums) / (counts @ sizes), [2.5, 97.5])
    return dict(estimate=float(np.mean(differences)), low=float(low), high=float(high))


def table(scores=None):
    """Tab. 2 (block means on all 40 images) and PGNM's paired PSNR differences."""
    scores = Path(scores or config.runs("denoise"))
    data = block_labels()
    blocks = {source: config.read_json(scores / source / "sidd_val.json") for source in SOURCES
              if (scores / source / "sidd_val.json").exists()}
    result = dict(scores={}, differences={})
    for source, values in blocks.items():
        result["scores"][source] = {subset: dict(psnr=float(np.mean(np.asarray(values["psnr"])[mask])),
                                                 ssim=float(np.mean(np.asarray(values["ssim"])[mask])))
                                    for subset, mask in data["subsets"].items()}
    if "pgnm" in blocks:
        for source in blocks:
            if source == "pgnm":
                continue
            difference = np.asarray(blocks["pgnm"]["psnr"]) - np.asarray(blocks[source]["psnr"])
            result["differences"][source] = {subset: scene_interval(difference[mask], data["scenes"][mask])
                                             for subset, mask in data["subsets"].items()}
    config.write_json(scores / "table.json", result)
    return result


def print_table(result):
    print("Tab. 2: SIDD Validation, 40 images, 1,280 blocks")
    print("%-26s%8s%8s" % ("Training noise", "PSNR", "SSIM"))
    for source, label in SOURCES.items():
        row = result["scores"].get(source)
        print("%-26s" % label + ("%8s%8s" % ("--", "--") if row is None else
                                 "%8.2f%8.3f" % (row["all"]["psnr"], row["all"]["ssim"])))
    if result["differences"]:
        print("\nPGNM minus row, PSNR (dB), 95% scene-bootstrap interval")
        print("%-26s" % "" + "".join("%25s" % SUBSETS[s] for s in SUBSETS))
        for source, cells in result["differences"].items():
            print("%-26s" % SOURCES[source] + "".join(
                "%25s" % ("%+.2f [%+.2f, %+.2f]" % (c["estimate"], c["low"], c["high"])) for c in cells.values()))
