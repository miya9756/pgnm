"""Simulated Boolean film grain (Tab. 3, Fig. 6).

    python -m experiments.film fit-blur      # 7x7 image blur B per radius (optional)
    python -m experiments.film fit           # PGNM preset per radius (optional)
    python -m experiments.film references    # 5 Newson renders of each test image
    python -m experiments.film render        # PGNM, Zhang23, independent Newson runs
    python -m experiments.film score         # needs torch, DISTS_pytorch and lpips
    python -m experiments.film table         # Tab. 3

Newson's Boolean renderer places grains of one radius r and blurs the result
(filter sigma 0.8 px), so the clean image x is not the expected grainy image.
For each radius a 7x7 linear filter B is fitted on the training pairs, and
residuals are measured against mu = B*x. PGNM and Zhang23 add grain to mu;
GrainNet (baselines/render_film.py) and the independent Newson run
use x. Every method's output is compared with five independent Newson
references of each of GrainNet's 60 test images: 25 output/reference pairs.

Outputs: runs/film/generated/<method>/<radius>/g<run>/<image>.npy (uint8 HxW),
references: runs/film/reference/<radius>/r<run>/<image>.npy.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from statistics import fmean
import subprocess
import tempfile

import numpy as np
from PIL import Image
from scipy.ndimage import correlate

from . import config
from .seeds import RUNS, bank_seeds, film_pgnm_seed, film_seed, radius_key, to_uint8

RADII = (.025, .05, .075, .1, .2, .4, .6, .8)
BORDER = 3          # pixels excluded at every edge, by every fit and score
BLUR_SIZE = 7
METHODS = dict(pgnm="PGNM (ours)", zhang23="Zhang23", grainnet="GrainNet",
               newson="Ind. Newson render")
COLUMNS = (("conditional_w1", "W1"), ("spatial_correlation_error", "Spatial"),
           ("dists", "DISTS"), ("lpips", "LPIPS"))
# Newson's renderer: Monte Carlo samples and output filter as in GrainNet's evaluation.
NEWSON = dict(MC=100000, filter_sigma=0.8)


@lru_cache(maxsize=1)
def split():
    return config.dataset("film")


def gray(path):
    """uint8 HxW; color files must have identical channels."""
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        if not np.all(array == array[..., :1]):
            raise ValueError("Film inputs must be grayscale: " + str(path))
        array = array[..., 0]
    return np.ascontiguousarray(array)


def test_image_path(name):
    return config.external("grainnet/src") / split()["test_images"][name]["file"]


def expected_image(clean, blur):
    """mu = clip(B * x, 0, 255) / 255, reflected edges, never rounded."""
    return np.clip(correlate(clean.astype(np.float64), np.asarray(blur, np.float64), mode="reflect"), 0, 255) / 255


def blurs():
    return config.read_json(config.REPO / "presets/film/blur.json")


# Fitting -------------------------------------------------------------------------

def training_rows(radius):
    root = config.film_data()
    return [(root / row["clean"], root / row["noisy"]) for row in split()["training"][radius_key(radius)]]


def fit_blur(radius):
    """Least-squares 7x7 filter, clean codes -> grainy codes, no intercept.

    Uses every pixel of the border-cropped training images whose full 7x7
    window lies inside the crop (float64 normal equations).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    k, h = BLUR_SIZE, BLUR_SIZE // 2
    A, b = np.zeros((k * k, k * k)), np.zeros(k * k)
    for clean_path, noisy_path in training_rows(radius):
        clean = gray(clean_path)[BORDER:-BORDER, BORDER:-BORDER].astype(np.float64)
        noisy = gray(noisy_path)[BORDER:-BORDER, BORDER:-BORDER].astype(np.float64)
        X = sliding_window_view(clean, (k, k)).reshape(-1, k * k)
        y = noisy[h:-h, h:-h].ravel()
        A += X.T @ X
        b += X.T @ y
    return np.linalg.solve(A, b).reshape(k, k)


def training_pairs(radius, blur):
    """(mu, grainy) float32 RGB pairs inside the border, gray replicated."""
    pairs = []
    for clean_path, noisy_path in training_rows(radius):
        mu = expected_image(gray(clean_path), blur).astype(np.float32)
        observed = gray(noisy_path).astype(np.float32) / 255
        crop = (slice(BORDER, -BORDER), slice(BORDER, -BORDER))
        pairs.append(tuple(np.ascontiguousarray(np.repeat(a[..., None], 3, -1)[crop]) for a in (mu, observed)))
    return pairs


def fit(radii=RADII, out=None):
    from pgnm import fit as fit_pgnm, save
    out = Path(out or config.runs("film/presets"))
    for radius in radii:
        target = out / (radius_key(radius) + ".json")
        if target.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        save(target, fit_pgnm(training_pairs(radius, blurs()[radius_key(radius)]), progress=print))
        print("fitted", radius_key(radius), flush=True)


# Rendering -----------------------------------------------------------------------

def output_path(method, radius, run, image, root=None):
    return Path(root or config.runs("film")) / "generated" / method / radius_key(radius) / f"g{run}" / (image + ".npy")


def reference_path(radius, run, image):
    return config.runs("film/reference", radius_key(radius), f"r{run}", image + ".npy")


def newson(image, radius, seed):
    """One render by Newson et al.'s Boolean model (the authors' CUDA renderer)."""
    binary = config.external("newson/build/bin/film_grain_rendering_main")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.png"
        subprocess.run([str(binary), str(test_image_path(image)), str(out), "-r", str(radius),
                        "-grainSigma", "0", "-filterSigma", str(NEWSON["filter_sigma"]), "-zoom", "1",
                        "-algorithmID", "0", "-NmonteCarlo", str(NEWSON["MC"]), "-randomizeSeed", str(seed),
                        "-verbose", "0", "-color", "0"], check=True, capture_output=True, timeout=900)
        return gray(out)


def references(radii=RADII, images=None):
    for radius in radii:
        for image in images or split()["test_images"]:
            for run in RUNS:
                target = reference_path(radius, run, image)
                if not target.exists():
                    config.save_array(target, newson(image, radius, film_seed("film3-reference", radius, image, run)))
            print("references", radius_key(radius), image, flush=True)


def render(radii=RADII, images=None, methods=("pgnm", "zhang23", "newson"), presets=None, root=None,
           pgnm_name="pgnm"):
    """Outputs of PGNM, Zhang23 and the independent Newson run.

    ``presets`` (default presets/film) and ``pgnm_name`` let variants such as the
    ablation reuse PGNM's random streams under their own name.
    """
    from pgnm import PGNM, load
    from baselines import zhang23
    presets = Path(presets or config.REPO / "presets/film")
    zhang = config.read_json(config.REPO / "presets/film/zhang23.json")
    for radius in radii:
        key, blur = radius_key(radius), blurs()[radius_key(radius)]
        for run in RUNS:
            model = PGNM(load(presets / f"{key}.json"), workers=2, bank_seeds=bank_seeds(run)) if "pgnm" in methods else None
            for image in images or split()["test_images"]:
                clean = gray(test_image_path(image))
                for method in methods:
                    target = output_path(pgnm_name if method == "pgnm" else method, radius, run, image, root)
                    if target.exists():
                        continue
                    if method == "pgnm":
                        mu = expected_image(clean, blur).astype(np.float32)
                        model.reset_window_selection()
                        rng = np.random.default_rng(film_pgnm_seed(radius, image, run))
                        output = to_uint8(model.render(np.repeat(mu[..., None], 3, -1), rng)[..., 0])
                    elif method == "zhang23":
                        mu = expected_image(clean, blur)[..., None]
                        rng = np.random.default_rng(film_seed("film3-generator-zhang_blur", radius, image, run))
                        output = to_uint8(mu[..., 0] + zhang23.synthesize(mu, zhang[key], rng)[..., 0])
                    else:  # an independent realization of the simulator itself
                        output = newson(image, radius, film_seed("film3-newson", radius, image, run))
                    config.save_array(target, output)
            print("rendered", key, run, flush=True)


# Scoring -------------------------------------------------------------------------

def score(radii=RADII, images=None, methods=None, root=None):
    from metrics import residual
    from metrics.appearance import Appearance
    root = Path(root or config.runs("film"))
    appearance = Appearance()
    for radius in radii:
        blur = blurs()[radius_key(radius)]
        for image in images or split()["test_images"]:
            pending = [m for m in methods or METHODS
                       if not (root / "scores" / m / radius_key(radius) / (image + ".json")).exists()
                       and all(output_path(m, radius, g, image, root).exists() for g in RUNS)]
            if not pending:
                continue
            mu = expected_image(gray(test_image_path(image)), blur)
            targets = [np.load(reference_path(radius, r, image)) for r in RUNS]
            states = [residual.prepare_gray(mu, target, BORDER) for target in targets]
            for method in pending:
                pairs = []
                for g in RUNS:
                    delivered = np.load(output_path(method, radius, g, image, root))
                    summary = residual.summarize_gray(states[0], delivered)
                    for r, target in enumerate(targets):
                        scores = residual.compare_gray(states[r], summary)
                        scores.update(appearance(target, delivered, BORDER))
                        pairs.append(dict(g=g, r=r, **scores))
                config.write_json(root / "scores" / method / radius_key(radius) / (image + ".json"),
                                  dict(method=method, radius=radius, image=image, pairs=pairs))
            print("scored", radius_key(radius), image, pending, flush=True)


def table(scores=None, methods=METHODS):
    """Mean over 25 pairs per image, then the 60 images, then the radii."""
    scores = Path(scores or config.runs("film/scores"))
    images = list(split()["test_images"])
    result = {}
    for scope, radii in (("all", RADII), ("small", [r for r in RADII if r <= .2])):
        result[scope] = {}
        for method in methods:
            files = {(r, i): scores / method / radius_key(r) / (i + ".json") for r in radii for i in images}
            if not all(f.exists() for f in files.values()):
                result[scope][method] = None
                continue
            cells = {k: config.read_json(f)["pairs"] for k, f in files.items()}
            result[scope][method] = {column: fmean(fmean(fmean(p[column] for p in cells[r, i]) for i in images)
                                                   for r in radii) for column, _ in COLUMNS}
    config.write_json(config.runs("film/table.json"), result)
    return result


def print_table(result):
    print("%-22s" % "Method" + "".join("%10s" % label for _, label in COLUMNS))
    for scope, title in (("all", "All eight radii (0.025-0.8)"), ("small", "Radii up to 0.2")):
        print(title)
        for method, label in METHODS.items():
            row = result[scope][method]
            print("  %-20s" % label + "".join("%10s" % "--" if row is None else "%10.4f" % row[c]
                                              for c, _ in COLUMNS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("fit-blur", "fit", "references", "render", "score", "table"))
    parser.add_argument("--radii", nargs="+", type=float, choices=RADII, default=list(RADII))
    parser.add_argument("--images", nargs="+", help="test image names 01..60")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS))
    args = parser.parse_args()
    if args.stage == "fit-blur":
        result = {radius_key(r): fit_blur(r).tolist() for r in args.radii}
        config.write_json(config.runs("film/blur.json"), result)
    elif args.stage == "fit":
        fit(args.radii)
    elif args.stage == "references":
        references(args.radii, args.images)
    elif args.stage == "render":
        render(args.radii, args.images, args.methods or ("pgnm", "zhang23", "newson"))
    elif args.stage == "score":
        score(args.radii, args.images, args.methods)
    else:
        print_table(table())


if __name__ == "__main__":
    main()
