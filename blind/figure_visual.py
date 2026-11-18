"""Visual comparison: `python -m blind.figure_visual [preset|shadows]` -> blind/visual_*.png.

Top row: the source image (coffee) as the estimators see it: noisy y, the two
denoisers' x̂, and the true clean image. Bottom row: noise rendered on a held-out
image (astronaut) with the true preset and with each single-image estimate from
the DnCNN, all from the same latent field, so only the parameters differ.
Crops are shown at 2× with nearest-neighbour upsampling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pgnm import load

from . import estimate as E
from .denoisers import dncnn, nlm
from .run import IMAGES, Cached, render_batch, source_image

INK2, SURF = "#52514e", "#fcfcfb"


def crop(image, top, left, size):
    return image[top:top + size, left:left + size]


def main(preset_name="N6_800", source="coffee", held_out="astronaut", size=144,
         source_box=(150, 250), held_box=(40, 160)):
    preset = load(Path(__file__).resolve().parents[1] / "presets" / "sidd" / f"{preset_name}.json")
    x = source_image(source)
    y = render_batch(preset, [x], 1000 + IMAGES.index(source))[0]
    sigma = float((y.astype(float) - x).std())
    D = Cached(dncnn("real"))
    estimates = dict(naive=E.naive(y, D), envelope=E.envelope(y, D), calibrated=E.calibrated(y, D))
    smooth = nlm(round(2 * sigma))(y)

    target = source_image(held_out)
    renders = {"true noise": preset, **{f"{k} (DnCNN)": v for k, v in estimates.items()}}
    rendered = {k: render_batch(p, [target], 77)[0] for k, p in renders.items()}   # one shared seed

    top = [("noisy y", y), ("x̂: DnCNN", D(y)), ("x̂: NL-means 2σ", smooth), ("clean (unknown)", x)]
    bottom = [("clean (held out)", target)] + list(rendered.items())
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.6), facecolor=SURF)
    for ax, (label, image) in zip(axes[0], top):
        ax.imshow(crop(image, *source_box, size), interpolation="nearest"); ax.set_title(label, color=INK2)
    axes[0, 4].axis("off")
    axes[0, 4].text(0, .5, f"{preset_name} noise, std {sigma:.0f} codes\n\nTop: what the estimators see\n(source image '{source}').\n\n"
                    "Bottom: noise synthesized on a\nheld-out image from each estimate,\nsame latent field in every panel.",
                    va="center", color=INK2, fontsize=11)
    for ax, (label, image) in zip(axes[1], bottom):
        ax.imshow(crop(image, *held_box, size), interpolation="nearest"); ax.set_title(label, color=INK2)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / f"visual_{preset_name}.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(out)


def shadows(presets=("N6_800", "S6_1600"), source="coffee", held_out="astronaut", size=72):
    """Zoom on the darkest region of the held-out image: true noise vs estimates."""
    from scipy.ndimage import uniform_filter
    target = source_image(held_out)
    lum = uniform_filter(target.mean(-1).astype(float), size)[size // 2:-(size // 2), size // 2:-(size // 2)]
    top, left = np.unravel_index(np.argmin(lum), lum.shape)
    D = Cached(dncnn("real"))
    fig, axes = plt.subplots(len(presets), 5, figsize=(16, 3.6 * len(presets)), facecolor=SURF)
    for row, name in enumerate(presets):
        preset = load(Path(__file__).resolve().parents[1] / "presets" / "sidd" / f"{name}.json")
        x = source_image(source)
        y = render_batch(preset, [x], 1000 + IMAGES.index(source))[0]
        est = {"true noise": preset, "naive": E.naive(y, D), "envelope": E.envelope(y, D),
               "calibrated": E.calibrated(y, D)}
        clean = crop(target, top, left, size).astype(float)
        axes[row, 0].imshow(crop(target, top, left, size), interpolation="nearest")
        axes[row, 0].set_title(f"clean (darkest region)\nmean code {clean.mean():.0f}", color=INK2)
        for ax, (k, p) in zip(axes[row, 1:], est.items()):
            r = crop(render_batch(p, [target], 77)[0], top, left, size)
            d = r.astype(float) - clean
            sd, mu = d.std(axis=(0, 1)), d.mean(axis=(0, 1))
            ax.imshow(r, interpolation="nearest")
            ax.set_title(f"{name} · {k}\nstd R/G/B {sd[0]:.0f}/{sd[1]:.0f}/{sd[2]:.0f}   "
                         f"mean R/G/B {mu[0]:+.0f}/{mu[1]:+.0f}/{mu[2]:+.0f}", color=INK2, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(h_pad=2.5)
    out = Path(__file__).resolve().parent / "visual_shadows.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["shadows"]:
        shadows()
    else:
        main(*sys.argv[1:2])
