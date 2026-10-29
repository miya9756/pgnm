"""Fig. 1: fitted PGNM presets interpolated across a photo.

    python -m figures.teaser --photo PHOTO     # -> runs/figures/pgnm-teaser.pdf and .png

The photo is not part of this repository; any photo works. Only its pixels are
read; none of its metadata (such as a location) is written.

From left to right the noise passes through four SIDD presets (Google Pixel,
ISO 10000 to 1600) and five Newson film-grain presets (radius 0.025 to 0.8,
each with its image blur B), placed at fixed column positions. Every column
between two neighboring presets blends the two: their latent fields filter one
shared white-noise draw and are mixed with variance-preserving weights, and
their residual lookups and blurred clean images are interpolated linearly (for
the lookups a quantile, i.e. Wasserstein, interpolation). Within the photo's
box the noisy image fades, from top to bottom, into its residual (x2.5, gray is
zero).

Fonts: Linux Biolinum for the labels (Debian/Ubuntu package fonts-linuxlibertine).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, correlate
from scipy.signal import fftconvolve

from experiments import config, film
from .common import output, pyplot, save

LABEL_FONT = "/usr/share/fonts/opentype/linux-libertine/LinBiolinum_R.otf"
LABEL_FONT_BOLD = "/usr/share/fonts/opentype/linux-libertine/LinBiolinum_RB.otf"
# (tick label, preset under presets/, column position in [0, 1])
ANCHORS = (("10000", "sidd/GP_10000", 0.05), ("6400", "sidd/GP_6400", 0.17),
           ("3200", "sidd/GP_3200", 0.29), ("1600", "sidd/GP_1600", 0.41),
           ("0.025", "film/r025", 0.59), ("0.1", "film/r100", 0.68), ("0.2", "film/r200", 0.77),
           ("0.4", "film/r400", 0.86), ("0.8", "film/r800", 0.95))
GRADIENT = ((0.0, "#1b2440"), (0.36, "#2c5a7c"), (0.62, "#5d5136"), (1.0, "#a24c27"))
RESIDUAL_GAIN = 2.5
SEED = 2027


# Clean image ---------------------------------------------------------------------

def hex_rgb(value):
    return np.array([int(value[i:i + 2], 16) for i in (1, 3, 5)], np.float64) / 255


def gradient(height, width):
    """Background: a horizontal color gradient, slightly darker at top and bottom."""
    xs = np.linspace(0, 1, width)
    stops = np.array([p for p, _ in GRADIENT])
    colors = np.stack([hex_rgb(c) for _, c in GRADIENT])
    row = np.stack([np.interp(xs, stops, colors[:, c]) for c in range(3)], -1)
    shade = 1 - 0.18 * np.abs(np.linspace(-1, 1, height))[:, None, None] ** 2
    return row[None] * shade


def photo_fill(path, height, width, box, vertical_offset):
    """The photo scaled to the box width (aspect ratio kept), cropped to the box; pixels only."""
    left, top, right, bottom = box
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        scale = (right - left) / rgb.width
        resized = rgb.resize((right - left, round(rgb.height * scale)), Image.LANCZOS)
    array = np.asarray(resized, np.float64) / 255
    start = int(vertical_offset * array.shape[0])
    crop = array[start:start + bottom - top]
    if crop.shape[0] != bottom - top:
        raise ValueError("The vertical offset leaves the photo; lower --photo-offset")
    out = np.zeros((height, width, 3))
    out[top:bottom, left:right] = crop
    return out


# Noise ---------------------------------------------------------------------------

def load_presets(clean):
    presets = []
    for label, name, position in ANCHORS:
        params = config.read_json(config.REPO / "presets" / (name + ".json"))
        if not np.array_equal(params["marginal"]["t"], np.arange(-3, 3.001, 0.25)):
            raise ValueError("lookup() expects the latent grid -3, -2.75, ..., 3")
        blurred = clean
        if name.startswith("film/"):
            blur = np.asarray(film.blurs()[name[5:]], np.float64)
            blurred = np.stack([correlate(clean[..., c], blur, mode="reflect") for c in range(3)], -1)
        presets.append(dict(label=label, kernels=np.asarray(params["kernels"], np.float64),
                            table=np.asarray(params["marginal"]["g"]), blurred=np.clip(blurred, 0, 1),
                            position=position))
    return presets


def latent(white, kernels):
    """Valid-support convolution of the shared white noise, kernels[y, x, input, output]."""
    return np.stack([sum(fftconvolve(white[..., i], kernels[..., i, c], mode="valid") for i in range(3))
                     for c in range(3)], -1)


def lookup(table, intensity, value):
    """The renderer's bilinear lookup, extrapolated linearly along the latent axis."""
    position = np.clip(intensity, 0, 1) * 255
    index = np.minimum(position.astype(int), 254)
    weight = position - index
    knot = (value + 3.0) / 0.25
    lower_knot = np.clip(np.floor(knot).astype(int), 0, 23)
    knot_weight = knot - lower_knot
    out = np.empty_like(intensity)
    for c in range(3):
        g = table[c]
        i, k, fk = index[..., c], lower_knot[..., c], knot_weight[..., c]
        low = g[i, k] + fk * (g[i, k + 1] - g[i, k])
        high = g[i + 1, k] + fk * (g[i + 1, k + 1] - g[i + 1, k])
        out[..., c] = low + weight[..., c] * (high - low)
    return out


def _pad(kernels, size):
    pad = (size - kernels.shape[0]) // 2
    return np.pad(kernels, ((pad, pad), (pad, pad), (0, 0), (0, 0)))


def synthesize(clean, seed=SEED):
    """-> (expected image, noisy image, presets), column by column between neighboring presets."""
    height, width, _ = clean.shape
    presets = load_presets(clean)
    size = max(p["kernels"].shape[0] for p in presets)
    white = np.random.default_rng(seed).standard_normal((height + size - 1, width + size - 1, 3))
    fields = []
    for p in presets:
        k, pad = p["kernels"].shape[0], (size - p["kernels"].shape[0]) // 2
        fields.append(latent(white[pad:pad + height + k - 1, pad:pad + width + k - 1], p["kernels"]))
    columns = (np.arange(width) + 0.5) / width
    positions = np.array([p["position"] for p in presets])
    segment = np.clip(np.searchsorted(positions, columns) - 1, 0, len(presets) - 2)
    weight = np.clip((columns - positions[segment]) / (positions[segment + 1] - positions[segment]), 0, 1)
    expected = np.empty_like(clean)
    noisy = np.empty_like(clean)
    for j in range(len(presets) - 1):
        cols = np.flatnonzero(segment == j)
        if not len(cols):
            continue
        a, b = presets[j], presets[j + 1]
        w = weight[cols][None, :, None]
        # One white-noise draw: the latent correlation per channel is the kernels' overlap.
        rho = np.einsum("hwic,hwic->c", *(_pad(k, size) for k in (a["kernels"], b["kernels"])))
        scale = np.sqrt(np.maximum((1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * rho, 0.09))
        t = ((1 - w) * fields[j][:, cols] + w * fields[j + 1][:, cols]) / scale
        mu = (1 - w) * a["blurred"][:, cols] + w * b["blurred"][:, cols]
        residual = (1 - w) * lookup(a["table"], mu, t) + w * lookup(b["table"], mu, t)
        expected[:, cols] = mu
        noisy[:, cols] = np.clip(mu + residual, 0, 1)
    return expected, noisy, presets


# Figure --------------------------------------------------------------------------

def compose(photo, width=1800, height=420, photo_offset=0.22):
    """-> (RGB image in [0, 1], presets, first and last photo row as fractions of the height)."""
    left, top, right, bottom = (round(0.075 * width), round(0.205 * height), round(0.925 * width),
                                round(0.835 * height))
    mask = np.zeros((height, width))
    mask[top:bottom, left:right] = 1
    fill = photo_fill(photo, height, width, (left, top, right, bottom), photo_offset)
    clean = mask[..., None] * fill + (1 - mask[..., None]) * gradient(height, width)
    expected, noisy, presets = synthesize(clean)
    # Inside the photo's box: the noisy image at the top, its residual at the bottom.
    rows = (np.arange(height) - top) / max(bottom - 1 - top, 1)
    fade = np.clip((rows - 0.42) / 0.40, 0, 1)
    fade = (fade * fade * (3 - 2 * fade))[:, None, None]
    residual_view = np.clip(0.5 + RESIDUAL_GAIN * (noisy - expected), 0, 1)
    inside = (1 - fade) * noisy + fade * residual_view
    image = mask[..., None] * inside + (1 - mask[..., None]) * noisy
    solid = mask > 0.5
    edge = (binary_dilation(solid, iterations=1) & ~binary_erosion(solid, iterations=1)).astype(np.float64)
    image = image * (1 - 0.45 * edge[..., None]) + 0.45 * edge[..., None] * 0.92
    return image, presets, (top / height, (bottom - 1) / height)


def figure(image, presets, photo_span, path):
    plt = pyplot()
    from matplotlib import font_manager
    for font in (LABEL_FONT, LABEL_FONT_BOLD):
        font_manager.fontManager.addfont(font)
    # Biolinum is an OpenType (CFF) font, which PDF Type 42 cannot hold; Type 3 can.
    plt.rcParams.update({"font.family": "Linux Biolinum O", "pdf.fonttype": 3, "font.size": 6.5})
    height, width, _ = image.shape
    fig = plt.figure(figsize=(6.875, 6.875 * height / width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, interpolation="none", extent=(0, 1, 0, 1), aspect="auto")
    ax.set_axis_off()
    white = dict(color="white", transform=ax.transAxes)
    sidd = [p for p in presets if "." not in p["label"]]
    grain = [p for p in presets if "." in p["label"]]
    boundary = (sidd[-1]["position"] + grain[0]["position"]) / 2
    ax.plot([boundary, boundary], [0, 1], transform=ax.transAxes, color="white", lw=0.6,
            ls=(0, (2, 2)), alpha=0.7)
    ax.text(0.012, 0.955, "SIDD camera noise", weight="bold", fontsize=7.5, va="top", **white)
    ax.text(0.012, 0.86, "Pixel, ISO 10000 → 1600", va="top", alpha=0.9, **white)
    ax.text(boundary + 0.012, 0.955, "Newson film grain", weight="bold", fontsize=7.5, va="top", **white)
    ax.text(boundary + 0.012, 0.86, "grain radius 0.025 → 0.8, with image blur", va="top", alpha=0.9, **white)
    for p in presets:
        ax.plot([p["position"]] * 2, [0, 0.035], transform=ax.transAxes, color="white", lw=0.6, alpha=0.8)
        ax.text(p["position"], 0.05, p["label"], ha="center", va="bottom", fontsize=5.8, alpha=0.9, **white)
    top, bottom = 1 - photo_span[0], 1 - photo_span[1]
    ax.annotate("", xy=(0.05, bottom + 0.01), xytext=(0.05, top - 0.01), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=0.6, mutation_scale=5, alpha=0.85))
    ax.text(0.035, (top + bottom) / 2, f"noisy → residual ×{RESIDUAL_GAIN:g}", rotation=-90,
            ha="center", va="center", fontsize=5.8, alpha=0.9, **white)
    save(fig, path, dpi=width / 6.875)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photo", type=Path, required=True, help="any photo; only its pixels are read")
    parser.add_argument("--out", type=Path, help="output folder (default: runs/figures)")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--photo-offset", type=float, default=0.22,
                        help="first photo row inside the box, as a fraction of the scaled photo height")
    args = parser.parse_args()
    image, presets, photo_span = compose(args.photo, args.width, args.height, args.photo_offset)
    figure(image, presets, photo_span, output("pgnm-teaser.pdf", args.out))


if __name__ == "__main__":
    main()
