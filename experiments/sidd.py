"""Camera-noise synthesis on SIDD (Tab. 1, Fig. 5).

    python -m experiments.sidd fit                 # 16 camera-ISO presets (optional)
    python -m experiments.sidd render              # PGNM, 5 runs x 39 sources
    python -m experiments.sidd score               # every method with outputs
    python -m experiments.sidd table               # Tab. 1

Fitting uses SIDD-Medium; scoring uses the official SIDD Validation blocks:
40 images of 32 blocks of 256x256 pixels each. One PGNM preset is fitted per
camera-ISO class on 256x256 crops of the training captures listed in
datasets/sidd.json (captures held out of fitting are listed there too). Image
19 (IP, ISO 1250) has no training class, which leaves 39 scored sources.
Baseline outputs come from baselines/render_sidd.py in each baseline's
environment and are scored by the same code.

Outputs: runs/sidd/generated/<method>/r<run>/img<NN>.npy, uint8 (32, 256, 256, 3).
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config
from .seeds import RUNS, bank_seeds, stream_seed, to_uint8

METHODS = dict(pgnm="PGNM (ours)", naflow="NAFlow", gmdcn="GMDCN", danet="DANet",
               srgb_flow="sRGB-Flow", nmflowgan="NM-FlowGAN")
# Tab. 1: all methods with a model for each scored class on the 39 sources;
# all methods on the 19 sources whose classes sRGB-Flow and NM-FlowGAN cover.
TABLE = (("S39", "All 39 sources (16 camera-ISO classes)", ("pgnm", "naflow", "gmdcn", "danet")),
         ("S19", "19 sources in sRGB-Flow and NM-FlowGAN", tuple(METHODS)))
COLUMNS = (("kl_tiles32", "KL"), ("conditional_w1", "W1"),
           ("spatial_correlation_error", "Spatial"), ("color_correlation_error", "Color"))


@lru_cache(maxsize=1)
def split():
    return config.dataset("sidd")


def class_name(image):
    label = split()["validation"][int(image)]
    return f"{label['camera']}_{label['iso']}"


@lru_cache(maxsize=1)
def validation_blocks():
    """(noisy, clean) uint8 arrays of shape (40, 32, 256, 256, 3)."""
    from scipy.io import loadmat
    root = config.sidd_validation()
    noisy = loadmat(root / "ValidationNoisyBlocksSrgb.mat")["ValidationNoisyBlocksSrgb"]
    clean = loadmat(root / "ValidationGtBlocksSrgb.mat")["ValidationGtBlocksSrgb"]
    return noisy, clean


# Training pairs ----------------------------------------------------------------

def _crop_origins(size, crop=256):
    """A 256-pixel grid plus one last crop flush with the far edge."""
    origins = list(range(0, size - crop + 1, crop))
    if size - (origins[-1] + crop) > 0:
        origins.append(size - crop)
    return origins


def _read_rgb(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def training_pairs(name):
    """Callable yielding the class's (clean, noisy) float32 crops in split order.

    A key '<capture>/<image>/<k>' names crop k, in row-major grid order, of the
    SIDD-Medium pair <capture>/<id>_{GT,NOISY}_SRGB_<image>.PNG. Each full image
    is decoded once; pgnm.fit may pass over the pairs several times.
    """
    keys = split()["classes"][name]["fit_keys"]
    images = {}

    def full(capture, image):
        if (capture, image) not in images:
            folder = config.sidd_medium() / "Data" / capture
            images[capture, image] = tuple(
                _read_rgb(folder / f"{capture[:4]}_{kind}_SRGB_{image}.PNG") for kind in ("GT", "NOISY"))
        return images[capture, image]

    def pairs():
        for key in keys:
            capture, image, index = key.split("/")
            clean, noisy = full(capture, image)
            height, width = clean.shape[:2]
            y, x = [(y, x) for y in _crop_origins(height) for x in _crop_origins(width)][int(index)]
            yield tuple(a[y:y + 256, x:x + 256].astype(np.float32) / 255 for a in (clean, noisy))
    return pairs


def fit(classes=None, out=None):
    from pgnm import fit as fit_pgnm, save
    out = Path(out or config.runs("sidd/presets"))
    for name in classes or sorted(split()["classes"]):
        target = out / (name + ".json")
        if target.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        save(target, fit_pgnm(training_pairs(name), progress=print))
        print("fitted", name, flush=True)


# Rendering and scoring ---------------------------------------------------------

def output_path(method, run, image, root=None):
    return Path(root or config.runs("sidd")) / "generated" / method / f"r{run}" / ("img%02d.npy" % image)


def render(images=None, presets=None, runs=RUNS, root=None, method="pgnm"):
    """PGNM outputs: one bank per run, a fresh window draw for every block.

    ``presets`` (default presets/sidd) and ``method`` let variants such as the
    ablation reuse the same random streams under their own name.
    """
    from pgnm import PGNM, load
    presets = Path(presets or config.REPO / "presets/sidd")
    _, clean = validation_blocks()
    for image in images or split()["sets"]["S39"]:
        name = class_name(image)
        for run in runs:
            target = output_path(method, run, image, root)
            if target.exists():
                continue
            model = PGNM(load(presets / (name + ".json")), workers=2, bank_seeds=bank_seeds(run))
            blocks = []
            for block, x in enumerate(clean[image]):
                model.reset_window_selection()
                rng = np.random.default_rng(stream_seed("cvpr-sidd", name, image, run, block))
                blocks.append(to_uint8(model.render(x.astype(np.float32) / 255, rng)))
            config.save_array(target, np.stack(blocks))
            print("rendered", image, run, flush=True)


def score(images=None, methods=None, root=None):
    from metrics import kl, residual
    root = Path(root or config.runs("sidd"))
    noisy, clean = validation_blocks()
    for image in images or split()["sets"]["S39"]:
        state = None
        for method in methods or METHODS:
            target = root / "scores" / method / ("img%02d.json" % image)
            paths = [output_path(method, run, image, root) for run in RUNS]
            if target.exists() or not all(p.exists() for p in paths):
                continue
            state = state or residual.prepare_rgb(clean[image], noisy[image])
            runs = []
            for path in paths:
                generated = np.load(path)
                scores = residual.compare_rgb(state, generated)
                scores.update(kl.block_and_tile_kl(clean[image], noisy[image], generated))
                runs.append(scores)
            config.write_json(target, dict(image=image, method=method, runs=runs))
            print("scored", image, method, flush=True)


def table(scores=None):
    """Tab. 1 levels from the per-source score files (None if any is missing)."""
    from metrics.statistics import level, source_weights
    scores = Path(scores or config.runs("sidd/scores"))
    data = split()
    result = {}
    for scope, _, methods in TABLE:
        sources = data["sets"][scope]
        weights = source_weights(sources, data["validation"])
        result[scope] = {}
        for method in methods:
            files = [scores / method / ("img%02d.json" % i) for i in sources]
            if not all(f.exists() for f in files):
                result[scope][method] = None
                continue
            runs = [config.read_json(f)["runs"] for f in files]
            result[scope][method] = {column: level([[r[column] for r in item] for item in runs], weights)
                                     for column, _ in COLUMNS}
    config.write_json(config.runs("sidd/table.json"), result)
    return result


def print_table(result):
    print("%-26s" % "Method" + "".join("%10s" % label for _, label in COLUMNS))
    for scope, title, methods in TABLE:
        print(title)
        for method in methods:
            row = result[scope][method]
            cells = ["%10s" % "--"] * len(COLUMNS) if row is None else [
                "%10.3f" % row[column] for column, _ in COLUMNS]
            print("  %-24s" % METHODS[method] + "".join(cells))


def export_clean():
    """Clean Validation blocks as one .npy, read by the baseline environments."""
    target = config.runs("sidd/validation_clean.npy")
    if not target.exists():
        config.save_array(target, validation_blocks()[1])
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("fit", "render", "score", "table", "export-clean"))
    parser.add_argument("--classes", nargs="+", help="camera-ISO classes to fit, e.g. S6_800")
    parser.add_argument("--images", nargs="+", type=int, help="Validation images (default: the 39 sources)")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS))
    parser.add_argument("--presets", type=Path, help="preset folder (default: presets/sidd)")
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.classes)
    elif args.stage == "render":
        render(args.images, args.presets)
    elif args.stage == "score":
        score(args.images, args.methods)
    elif args.stage == "table":
        print_table(table())
    else:
        export_clean()


if __name__ == "__main__":
    main()
