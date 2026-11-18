"""Single-image PGNM estimation on synthetic ground truth: `python -m blind.run`.

For every clean source image, preset (the true noise) and blind denoiser, one
noisy image is rendered with the preset and handed to each estimator together
with the denoiser. An estimate is scored by rendering it and the true preset
on the *other* source images (center 256×256 crops, two seeds) and comparing
the two with the paper's metrics, plus a split of the marginal error into
strength and offset. `truth` is the true preset against itself with new seeds
(the scores' noise floor); `oracle` fits the true pair of the same single image
(the best a single image allows with the paper's fitter).

Writes runs/blind/results.json (resumable) and runs/blind/results.md.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import skimage.data

from metrics.residual import compare_rgb, prepare_rgb
from pgnm import load

from . import estimate
from .denoisers import dncnn, nlm

IMAGES = ("astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry", "retina")
PRESETS = ("IP_1000", "N6_800", "S6_1600", "GP_10000")
# NLM strength is set from the true noise std of the noisy image (an experimental
# condition, never shown to an estimator): 1x leaves some noise, 2x and 3x remove
# it and increasingly erase fine signal, the regime of an over-smoothing blind CNN.
DENOISERS = dict(nlm1x=lambda sigma: nlm(round(sigma)), nlm2x=lambda sigma: nlm(round(2 * sigma)),
                 nlm3x=lambda sigma: nlm(round(3 * sigma)), dncnn=lambda sigma: _dncnn())
_DNCNN = []


def _dncnn():
    if not _DNCNN:
        _DNCNN.append(dncnn("real"))
    return _DNCNN[0]
ESTIMATORS = ("oracle", "naive", "masked", "envelope", "calibrated")
CROP, EVAL_SEEDS = 256, (11, 12)


def source_image(name):
    image = getattr(skimage.data, name)()
    if max(image.shape[:2]) > 640:
        scale = 512 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image[..., :3])


def center_crop(image, size=CROP):
    top, left = (image.shape[0] - size) // 2, (image.shape[1] - size) // 2
    return image[top:top + size, left:left + size]


def render_batch(params, cleans, seed):
    rng = np.random.default_rng(seed)
    return np.stack([estimate.to_uint8(estimate.render_fresh(params, estimate.to_float(c), rng))
                     for c in cleans])


def split_marginal(cleans, reference, generated, bins=16):
    """Strength error (|log std ratio|, %), signed strength bias (%) and offset error (codes)."""
    clean = cleans.astype(np.float64)
    group = np.minimum((clean / 256 * bins).astype(int), bins - 1)
    absolute, signed, offset, weights = [], [], [], []
    for c in range(3):
        for b in range(bins):
            selected = group[..., c] == b
            if selected.sum() < 512:
                continue
            r = reference[..., c][selected] - clean[..., c][selected]
            g = generated[..., c][selected] - clean[..., c][selected]
            ratio = np.log(max(g.std(), .1) / max(r.std(), .1))
            absolute.append(abs(ratio)); signed.append(ratio)
            offset.append(abs(g.mean() - r.mean())); weights.append(selected.sum())
    w = np.array(weights, float) / np.sum(weights)
    return dict(strength_error=100 * float(w @ absolute), strength_bias=100 * float(w @ signed),
                offset_error=float(w @ offset))


class Scorer:
    """Reference renderings of one preset on the held-out crops, prepared once."""

    def __init__(self, preset, held_out):
        self.cleans = np.stack([center_crop(image) for image in held_out])
        self.reference = [render_batch(preset, self.cleans, seed) for seed in EVAL_SEEDS]
        self.states = [prepare_rgb(self.cleans, reference) for reference in self.reference]

    def __call__(self, params, seed_offset=100):
        scores = []
        for state, reference, seed in zip(self.states, self.reference, EVAL_SEEDS):
            generated = render_batch(params, self.cleans, seed + seed_offset)
            scores.append(dict(**compare_rgb(state, generated),
                               **split_marginal(self.cleans, reference, generated)))
        return {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}


class Cached:
    """Denoiser with a one-entry cache, so every estimator sees the same x̂."""

    def __init__(self, denoise):
        self.denoise, self.key, self.value, self.calls = denoise, None, None, 0

    def __call__(self, noisy):
        key = hash(noisy.tobytes())
        if key != self.key:
            self.key, self.value, self.calls = key, self.denoise(noisy), self.calls + 1
        return self.value


def run(out, images=IMAGES, presets=PRESETS, denoisers=tuple(DENOISERS)):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    sources = {name: source_image(name) for name in IMAGES}
    built = {}
    for preset_name in presets:
        preset = load(Path(__file__).resolve().parents[1] / "presets" / "sidd" / f"{preset_name}.json")
        for image_index, image_name in enumerate(images):
            scorer = None
            noisy = render_batch(preset, [sources[image_name]], 1000 + image_index)[0]
            sigma = float((noisy.astype(float) - sources[image_name]).std())
            for denoiser_name in ("none", *denoisers):
                estimators = ("truth", "oracle") if denoiser_name == "none" else ESTIMATORS[1:]
                for estimator in estimators:
                    key = f"{preset_name}/{image_name}/{denoiser_name}/{estimator}"
                    if key in results:
                        continue
                    if scorer is None:
                        scorer = Scorer(preset, [sources[n] for n in IMAGES if n != image_name])
                    start = time.time()
                    extra = {}
                    if estimator == "truth":
                        params = preset
                    elif estimator == "oracle":
                        params = estimate.oracle(noisy, sources[image_name])
                    else:
                        if (denoiser_name, sigma) not in built:
                            built.clear()
                            built[denoiser_name, sigma] = Cached(DENOISERS[denoiser_name](sigma))
                        denoise = built[denoiser_name, sigma]
                        if estimator == "calibrated":
                            history = []
                            params = estimate.calibrated(noisy, denoise, history=history)
                            extra["history"] = history
                        else:
                            params = estimate.ESTIMATORS[estimator](noisy, denoise)
                    results[key] = dict(**scorer(params), seconds=time.time() - start, **extra)
                    path.write_text(json.dumps(results, indent=1))
                    print(key, {k: round(v, 3) for k, v in results[key].items() if k != "history"},
                          flush=True)
    (out / "results.md").write_text(table(results))
    print(table(results))


METRICS = (("conditional_w1", "W1", 2), ("strength_error", "strength err %", 1),
           ("strength_bias", "strength bias %", 1), ("offset_error", "offset", 2),
           ("spatial_correlation_error", "spatial", 3), ("color_correlation_error", "color", 3))


def table(results):
    """Means over images; one block per denoiser, one row per estimator."""
    keys = [k.split("/") for k in results]
    presets = list(dict.fromkeys(k[0] for k in keys))
    denoisers = list(dict.fromkeys(k[2] for k in keys))
    lines = []
    for preset in presets:
        lines += [f"### {preset}", "", "| denoiser | estimator | " + " | ".join(m[1] for m in METRICS) + " |",
                  "|---|---|" + "---|" * len(METRICS)]
        for denoiser in denoisers:
            estimators = list(dict.fromkeys(k[3] for k in keys if k[0] == preset and k[2] == denoiser))
            for estimator in estimators:
                rows = [v for k, v in results.items()
                        if k.startswith(f"{preset}/") and k.endswith(f"/{denoiser}/{estimator}")]
                values = [np.mean([r[m] for r in rows if r.get(m) is not None]) for m, _, _ in METRICS]
                lines.append(f"| {denoiser} | {estimator} | " +
                             " | ".join(f"{v:.{d}f}" for v, (_, _, d) in zip(values, METRICS)) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", nargs="+", default=list(IMAGES), choices=IMAGES)
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    parser.add_argument("--denoisers", nargs="+", default=list(DENOISERS), choices=list(DENOISERS))
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("PGNM_RUNS", "runs")) / "blind")
    arguments = parser.parse_args()
    run(arguments.out, arguments.images, arguments.presets, arguments.denoisers)
