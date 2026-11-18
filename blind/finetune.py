"""Does a single-image noise estimate improve the denoiser? `python -m blind.finetune`.

The noisy test image's noise is estimated from that image alone (naive or
calibrated, with the native DnCNN as the blind denoiser). The native DnCNN is
then fine-tuned on the *other* clean images plus synthetic noise from the
estimate, and used to denoise the test image again. Fine-tuning on the true
preset is the upper bound. Needs PyTorch (CPU is enough: 300 steps of 16
patches of 64×64). Writes runs/blind/finetune.json and blind/finetune_*.png.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from pgnm import load

from . import estimate as E
from .denoisers import dncnn
from .run import IMAGES, Cached, render_batch, source_image

ROOT = Path(__file__).resolve().parents[1]
STEPS, BATCH, PATCH, LR = 300, 16, 64, 1e-4


def psnr(a, b):
    return float(10 * np.log10(255 ** 2 / np.mean((a.astype(float) - b.astype(float)) ** 2)))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))


def finetune(params, cleans, seed=0, steps=STEPS):
    import torch
    from experiments.denoise.train import DnCNN, LEVELS
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = DnCNN()
    model.load_state_dict(torch.load(ROOT / "checkpoints/denoise/real.pt", map_location="cpu"))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    pool = []
    for step in range(steps):
        if step % 50 == 0:   # fresh noise on every clean image every 50 steps
            pool = [(c, render_batch(params, [c], int(rng.integers(1 << 30)))[0]) for c in cleans]
        noisy, clean = [], []
        for _ in range(BATCH):
            c, n = pool[rng.integers(len(pool))]
            top, left = rng.integers(c.shape[0] - PATCH), rng.integers(c.shape[1] - PATCH)
            flip = rng.integers(2)
            pc, pn = c[top:top + PATCH, left:left + PATCH], n[top:top + PATCH, left:left + PATCH]
            if flip:
                pc, pn = pc[:, ::-1], pn[:, ::-1]
            clean.append(pc); noisy.append(pn)
        to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a))).permute(0, 3, 1, 2).float()
        loss = torch.nn.functional.mse_loss(model(to_tensor(noisy) / LEVELS) * LEVELS, to_tensor(clean))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()

    def denoise(noisy):
        with torch.no_grad():
            x = torch.from_numpy(noisy).permute(2, 0, 1)[None].float()
            y = model(x / LEVELS).clamp(0, 1) * LEVELS
        return torch.floor(y + .5).clamp(0, 255)[0].permute(1, 2, 0).to(torch.uint8).numpy()
    return denoise


def run(presets=("N6_800", "S6_1600"), test="coffee"):
    out = Path(os.environ.get("PGNM_RUNS", "runs")) / "blind"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    cleans = [source_image(n) for n in IMAGES if n != test]
    x = source_image(test)
    native = Cached(dncnn("real"))
    for name in presets:
        preset = load(ROOT / "presets" / "sidd" / f"{name}.json")
        y = render_batch(preset, [x], 1000 + IMAGES.index(test))[0]
        outputs = {"native DnCNN": native(y)}
        sources = {"naive": E.naive(y, native), "calibrated": E.calibrated(y, native), "true preset": preset}
        for label, params in sources.items():
            start = time.time()
            outputs[f"fine-tuned: {label}"] = finetune(params, cleans)(y)
            print(name, label, f"{time.time() - start:.0f} s", flush=True)
        results[name] = {k: dict(psnr=psnr(v, x), ssim=ssim(v, x)) for k, v in outputs.items()}
        results[name]["noisy"] = dict(psnr=psnr(y, x), ssim=ssim(y, x))
        print(name, json.dumps(results[name], indent=1), flush=True)
        figure(name, x, y, outputs, results[name])
        np.savez_compressed(out / f"finetune_{name}.npz", clean=x, noisy=y, **{k.replace(" ", "_").replace(":", ""): v for k, v in outputs.items()})
    (out / "finetune.json").write_text(json.dumps(results, indent=1))


def figure(name, x, y, outputs, scores, boxes=((150, 250, 144), (20, 380, 144))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, surface = "#52514e", "#fcfcfb"
    panels = [("noisy", y), *outputs.items(), ("clean", x)]
    fig, axes = plt.subplots(len(boxes), len(panels), figsize=(3.1 * len(panels), 3.4 * len(boxes)), facecolor=surface)
    for row, (top, left, size) in enumerate(boxes):
        for ax, (label, image) in zip(axes[row], panels):
            ax.imshow(image[top:top + size, left:left + size], interpolation="nearest")
            if row == 0:
                s = scores.get(label)
                ax.set_title(label + (f"\n{s['psnr']:.2f} dB · SSIM {s['ssim']:.3f}" if s else "\n"), color=ink, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(f"{name} noise on 'coffee': native DnCNN vs. DnCNN fine-tuned on synthetic noise "
                 f"({STEPS} steps, other 5 clean images)", color=ink)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / f"finetune_{name}.png", dpi=140, facecolor=surface)


if __name__ == "__main__":
    run(*([tuple(sys.argv[1:])] if sys.argv[1:] else []))
