"""Figs. 5 and 6: visual comparisons on SIDD and on Newson film grain.

    python -m figures.visual_comparison sidd   # Fig. 5 (no argument: both)
    python -m figures.visual_comparison film   # Fig. 6

write the paper's panels, runs/figures/main_vis_<row>/<method>.png and
<method>-res.png, a one-page preview of the figure (fig5-*.pdf, fig6-*.pdf) and
the chosen windows (fig5-*.json, fig6-*.json). Fig. 5 reads the SIDD
Validation blocks and the outputs of experiments/sidd.py and
baselines/render_sidd.py; Fig. 6 those of experiments/film.py and
baselines/render_film.py.

Every 128x128 window is chosen from the clean images alone; a drawn window is
drawn again when more than 25% of its clean pixels are near saturation (any
channel at most 4 or at least 251).

  SIDD, random window (S6 ISO 400): a generator seeded with 0 visits the 16
  camera-ISO classes in (camera, ISO) order and draws for each a source
  uniformly from the class's scored Validation images, then a block and a
  window position uniformly.
  SIDD, most textured window (GP ISO 6400): over every block of the class's
  sources, at window positions on an 8-pixel grid, the window with the highest
  median contrast of the clean image (7x7 local standard deviation, maximum
  over RGB: the contrast of PGNM's flatness mask); ties go to the first source,
  block, row and column.
  Film grain (radius 0.4 and 0.8): a generator seeded with 1 permutes the 60
  test images, assigns the first eight to the eight radii in ascending order and
  then draws a window inside the 3-pixel border.

Panels show each method's first generator run and the first Newson reference.
Residual panels show 128 + gain * (output - expected), where expected is the
clean image (SIDD) or mu = B*x (film), with one gain per row from the real or
reference residual alone: 100 / (99th percentile of |residual|), clipped to
[1, 16] and rounded down to a multiple of 1/2. Panels are enlarged 2x by pixel
repetition. sRGB-Flow and NM-FlowGAN have checkpoints only for the classes of
19 sources, which exclude GP ISO 6400 (n/a in the figure).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments import config, film, sidd
from experiments.seeds import radius_key
from .common import output, pyplot, save

CROP, SCALE, TEXTURE_STRIDE = 128, 2, 8
NEAR_LOW, NEAR_HIGH, MAX_NEAR = 4, 251, 0.25
ATTEMPTS = 1000
SIDD_COLUMNS = (("clean", "Clean"), ("real", "Real"), ("pgnm", "PGNM"), ("gmdcn", "GMDCN"),
                ("nmflowgan", "NM-FlowGAN"), ("naflow", "NAFlow"), ("srgb_flow", "sRGB-Flow"), ("danet", "DANet"))
NARROW = ("srgb_flow", "nmflowgan")  # S19 sources only
FILM_COLUMNS = (("clean", "Clean"), ("reference", "Newson"), ("pgnm", "PGNM + B"),
                ("zhang23", "Zhang23 + B"), ("grainnet", "GrainNet"))
# Fig. 5 rows: (folder, label, camera, ISO, rule); Fig. 6 rows: (folder, label, radius).
SIDD_ROWS = (("main_vis_s6_400", "S6 400", "S6", 400, "random"),
             ("main_vis_gp_6400", "GP 6400", "GP", 6400, "textured"))
FILM_ROWS = (("main_vis_r_400", "r = 0.4", 0.4), ("main_vis_r_800", "r = 0.8", 0.8))


# Window selection, from clean images only ------------------------------------------

def eligible(clean):
    pixels = clean.reshape(clean.shape[0], clean.shape[1], -1)
    return float(((pixels <= NEAR_LOW) | (pixels >= NEAR_HIGH)).any(-1).mean()) <= MAX_NEAR


def sidd_sources(camera, iso):
    labels = sidd.split()["validation"]
    return [i for i in sidd.split()["sets"]["S39"] if (labels[i]["camera"], labels[i]["iso"]) == (camera, iso)]


def sidd_random_windows(clean):
    """{(camera, ISO): (source, block, y, x)} for all 16 classes, drawn in order."""
    labels = sidd.split()["validation"]
    classes = sorted({(labels[i]["camera"], labels[i]["iso"]) for i in sidd.split()["sets"]["S39"]})
    rng = np.random.default_rng(0)
    windows = {}
    for camera, iso in classes:
        source = int(rng.choice(sidd_sources(camera, iso)))
        for _ in range(ATTEMPTS):
            block = int(rng.integers(32))
            y, x = (int(v) for v in rng.integers(0, 256 - CROP + 1, size=2))
            if eligible(clean[source, block, y:y + CROP, x:x + CROP]):
                break
        else:
            raise RuntimeError(f"No eligible window for {camera} ISO {iso}")
        windows[camera, iso] = (source, block, y, x)
    return windows


def texture_windows(clean, border=0):
    """-> (median contrast, near-saturated fraction, y, x) of every window on the 8-pixel grid."""
    from numpy.lib.stride_tricks import sliding_window_view
    from pgnm.fitting import flat_pixel_contrast
    image = clean.reshape(clean.shape[0], clean.shape[1], -1)
    contrast = flat_pixel_contrast(image)
    near = ((image <= NEAR_LOW) | (image >= NEAR_HIGH)).any(-1)
    h, w = contrast.shape
    ys = np.arange(border, h - border - CROP + 1, TEXTURE_STRIDE)
    xs = np.arange(border, w - border - CROP + 1, TEXTURE_STRIDE)
    windows = sliding_window_view(contrast, (CROP, CROP))
    score = np.concatenate([np.median(windows[y, xs].reshape(len(xs), -1), 1) for y in ys])
    fraction = sliding_window_view(near, (CROP, CROP))[np.ix_(ys, xs)].mean((2, 3)).ravel()
    y, x = (v.ravel() for v in np.meshgrid(ys, xs, indexing="ij"))
    return score, fraction, y, x


def sidd_textured_window(clean, camera, iso):
    """(source, block, y, x) of the class's most textured eligible window."""
    best, best_score = None, None
    for source in sidd_sources(camera, iso):
        for block in range(32):
            score, fraction, y, x = texture_windows(clean[source, block])
            ok = np.flatnonzero(fraction <= MAX_NEAR)
            if len(ok) and (best is None or score[ok].max() > best_score):
                k = ok[int(np.argmax(score[ok]))]
                best, best_score = (source, block, int(y[k]), int(x[k])), float(score[k])
    if best is None:
        raise RuntimeError(f"No eligible window for {camera} ISO {iso}")
    return best


def film_windows():
    """{radius: (image, y, x)} for all eight radii, drawn in ascending order."""
    images = sorted(film.split()["test_images"])
    rng = np.random.default_rng(1)
    windows = {}
    for radius, index in zip(film.RADII, rng.permutation(len(images))[:len(film.RADII)]):
        image = images[int(index)]
        clean = film.gray(film.test_image_path(image))
        h, w = clean.shape
        for _ in range(ATTEMPTS):
            y = int(rng.integers(film.BORDER, h - film.BORDER - CROP + 1))
            x = int(rng.integers(film.BORDER, w - film.BORDER - CROP + 1))
            if eligible(clean[y:y + CROP, x:x + CROP]):
                break
        else:
            raise RuntimeError("No eligible window in film test image " + image)
        windows[radius] = (image, y, x)
    return windows


# Panels ------------------------------------------------------------------------------

def gain_for(real_residual):
    p99 = float(np.percentile(np.abs(real_residual), 99))
    return float(np.floor(np.clip(100 / max(p99, 1e-6), 1, 16) * 2) / 2)


def residual_image(residual, gain):
    return np.clip(np.floor(128 + gain * residual + .5), 0, 255).astype(np.uint8)


def panels(crops, expected, gain_from):
    """{name: crop} -> {name: crop, name-res: residual image} and the row's gain."""
    gain = gain_for(crops[gain_from].astype(expected.dtype) - expected)
    out = dict(crops)
    for name, crop in crops.items():
        if name != "clean":
            out[name + "-res"] = residual_image(crop.astype(expected.dtype) - expected, gain)
    return out, gain


def sidd_row(noisy, clean, source, block, y, x):
    window = np.s_[y:y + CROP, x:x + CROP]
    crops = dict(clean=np.asarray(clean[source, block][window]), real=np.asarray(noisy[source, block][window]))
    for method, _ in SIDD_COLUMNS[2:]:
        if method in NARROW and source not in sidd.split()["sets"]["S19"]:
            continue
        crops[method] = np.asarray(np.load(sidd.output_path(method, 0, source), mmap_mode="r")[block][window])
    return panels(crops, crops["clean"].astype(np.int16), "real")


def film_row(radius, image, y, x):
    window = np.s_[y:y + CROP, x:x + CROP]
    clean = film.gray(film.test_image_path(image))
    mu = 255 * film.expected_image(clean, film.blurs()[radius_key(radius)])
    crops = dict(clean=clean[window], reference=np.load(film.reference_path(radius, 0, image))[window])
    for method, _ in FILM_COLUMNS[2:]:
        crops[method] = np.load(film.output_path(method, radius, 0, image))[window]
    return panels(crops, mu[window], "reference")


def save_png(path, array, scale=SCALE):
    from PIL import Image
    array = np.repeat(np.repeat(array, scale, 0), scale, 1)
    Image.fromarray(np.ascontiguousarray(array)).save(path, optimize=True)


def preview(rows, columns, path):
    """The figure as one PDF: per row an image line and a residual line; empty cells are n/a."""
    plt = pyplot()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42})
    cell = 6.875 / len(columns)
    fig, axes = plt.subplots(2 * len(rows), len(columns), figsize=(6.875, cell * 2 * len(rows) + 0.15),
                             gridspec_kw=dict(wspace=0.02, hspace=0.02, left=0, right=1, bottom=0,
                                              top=1 - 0.15 / (cell * 2 * len(rows) + 0.15)))
    for r, (label, images) in enumerate(rows):
        for c, (name, title) in enumerate(columns):
            for line, suffix in enumerate(("", "-res")):
                ax = axes[2 * r + line, c]
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if r == 0 and line == 0:
                    ax.set_title(title, fontsize=6, pad=2)
                image = images.get(name + suffix)
                if image is not None:
                    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                elif name == "clean":
                    ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.set_facecolor("#f0f0f0")
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="#737373", transform=ax.transAxes)
    save(fig, path, dpi=300)


def write_row(folder, images, out):
    for name, image in images.items():
        save_png(output(f"{folder}/{name}.png", out), image)


def sidd_figure(out=None):
    """Fig. 5: one random and one most textured window."""
    noisy, clean = sidd.validation_blocks()
    random_windows = sidd_random_windows(clean)
    windows, rows = {}, []
    for folder, label, camera, iso, rule in SIDD_ROWS:
        window = random_windows[camera, iso] if rule == "random" else sidd_textured_window(clean, camera, iso)
        images, gain = sidd_row(noisy, clean, *window)
        write_row(folder, images, out)
        windows[folder] = dict(zip(("source", "block", "y", "x"), window), gain=gain)
        rows.append((label, images))
    preview(rows, SIDD_COLUMNS, output("fig5-sidd-visual-comparison.pdf", out))
    config.write_json(output("fig5-sidd-windows.json", out), windows)
    for folder, window in windows.items():
        print(folder, window)


def film_figure(out=None):
    """Fig. 6: the random windows of radius 0.4 and 0.8."""
    grain_windows = film_windows()
    windows, rows = {}, []
    for folder, label, radius in FILM_ROWS:
        images, gain = film_row(radius, *grain_windows[radius])
        write_row(folder, images, out)
        windows[folder] = dict(zip(("image", "y", "x"), grain_windows[radius]), gain=gain)
        rows.append((label, images))
    preview(rows, FILM_COLUMNS, output("fig6-film-visual-comparison.pdf", out))
    config.write_json(output("fig6-film-windows.json", out), windows)
    for folder, window in windows.items():
        print(folder, window)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", nargs="?", choices=("sidd", "film"), help="Fig. 5 or Fig. 6 (default: both)")
    parser.add_argument("--out", type=Path, help="output folder (default: runs/figures)")
    args = parser.parse_args()
    if args.figure in (None, "sidd"):
        sidd_figure(args.out)
    if args.figure in (None, "film"):
        film_figure(args.out)


if __name__ == "__main__":
    main()
