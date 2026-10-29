"""Fig. 7: joint PCA of the 24 fitted presets, as one strip of six panels.

    python -m figures.joint_pca   # -> runs/figures/joint-pca-strip/*.pdf

One PCA of the 24 convolution tensors (16 SIDD camera-ISO classes in
presets/sidd/, 8 Newson radii in presets/film/) and one of their 24 residual
lookups, each over the flattened arrays after subtracting their mean. A
component's sign makes its largest-magnitude entry positive. Per PCA the strip
has three panels: the scores in PC1 (horizontal) and PC2, SIDD as blue circles
and Newson as orange triangles that grow with ISO or radius; PC1; and PC2. A
kernel component shows its central 11x11 taps, one block per output (rows) and
input (columns) channel; a lookup component shows its green channel over
clean intensity 0..1 (horizontal) and latent value -3..3. Red is positive, blue
negative. The panels carry no axes or labels; the paper's caption explains them
and prints the explained variance, which this script also prints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments import config
from .common import output, pyplot, save

HEIGHT, SCATTER_WIDTH = 0.92, 1.3  # inches, as printed in the paper
COLORS = {"SIDD": "#2a78d6", "Newson": "#eb6834"}
MARKERS = {"SIDD": "o", "Newson": "^"}


def load():
    """The 24 presets: dataset, ISO or radius, kernels (25, 25, 3, 3), lookup (3, 256, 25)."""
    paths = [("SIDD", p) for p in sorted((config.REPO / "presets/sidd").glob("*.json"))]
    paths += [("Newson", p) for p in sorted((config.REPO / "presets/film").glob("r*.json"))]
    if len(paths) != 24:
        raise ValueError("Expected 16 SIDD and 8 film presets, found %d" % len(paths))
    presets = []
    for dataset, path in paths:
        params = config.read_json(path)
        order = int(path.stem.split("_")[1]) if dataset == "SIDD" else int(path.stem[1:]) / 1000
        presets.append(dict(dataset=dataset, order=order, kernels=np.asarray(params["kernels"], np.float64),
                            lookup=np.asarray(params["marginal"]["g"], np.float64)))
    return presets


def pca(stack):
    rows = stack.reshape(len(stack), -1)
    u, s, vt = np.linalg.svd(rows - rows.mean(0), full_matrices=False)
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(1)])  # largest entry positive
    signs[signs == 0] = 1
    return dict(components=(vt * signs[:, None]).reshape((-1,) + stack.shape[1:]),
                scores=u * s * signs[None, :], variance=s ** 2 / (s ** 2).sum())


def _cells(ax, values, cmap, limit, x=None, y=None):
    """One filled vector rectangle per tap or table entry, so viewers never smooth them."""
    rows, cols = values.shape
    x = np.arange(cols + 1) if x is None else x
    y = np.arange(rows + 1) if y is None else y
    ax.pcolormesh(x, y, values, cmap=cmap, vmin=-limit, vmax=limit, shading="flat",
                  edgecolors="face", linewidth=0.1, antialiased=False, rasterized=False)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])


def scatter(plt, presets, result, path):
    fig = plt.figure(figsize=(SCATTER_WIDTH, HEIGHT))
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.92])
    ax.axhline(0, color="#dce0e4", lw=0.4, zorder=0)
    ax.axvline(0, color="#dce0e4", lw=0.4, zorder=0)
    for dataset in ("SIDD", "Newson"):
        idx = [i for i, p in enumerate(presets) if p["dataset"] == dataset]
        order = np.array([presets[i]["order"] for i in idx])
        rank = (np.argsort(np.argsort(order)) + 0.5) / len(order)  # ISO or radius rank within the dataset
        ax.scatter(result["scores"][idx, 0], result["scores"][idx, 1], s=4 + 18 * rank, color=COLORS[dataset],
                   marker=MARKERS[dataset], edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_axis_off()
    ax.margins(0.08)
    save(fig, path, dpi=400)


def kernel_component(plt, component, path, crop=11, gap=1):
    start = (component.shape[0] - crop) // 2
    block = component[start:start + crop, start:start + crop]
    size = 3 * crop + 2 * gap
    mosaic = np.full((size, size), np.nan)
    for out in range(3):
        for inp in range(3):
            y, x = out * (crop + gap), inp * (crop + gap)
            mosaic[y:y + crop, x:x + crop] = block[:, :, inp, out]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    fig = plt.figure(figsize=(HEIGHT, HEIGHT))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    _cells(ax, np.ma.masked_invalid(mosaic[::-1]), cmap, np.nanmax(np.abs(mosaic)))  # row 0 at the top
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(fig, path, dpi=400)


def lookup_component(plt, component, path):
    table = component[1].T  # green channel: latent value (rows) x clean intensity (columns)
    fig = plt.figure(figsize=(HEIGHT, HEIGHT))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    _cells(ax, table, plt.get_cmap("RdBu_r"), np.abs(table).max(), x=np.linspace(0, 1, table.shape[1] + 1),
           y=np.linspace(-3, 3, table.shape[0] + 1))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_linewidth(0.35)
        side.set_color("#9aa1a9")
    save(fig, path, dpi=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="output folder (default: runs/figures)")
    args = parser.parse_args()
    plt = pyplot()
    plt.rcParams["pdf.fonttype"] = 42
    presets = load()
    for part, prefix, draw in (("kernels", "kernel", kernel_component), ("lookup", "lookup", lookup_component)):
        result = pca(np.stack([p[part] for p in presets]))
        scatter(plt, presets, result, output(f"joint-pca-strip/{prefix}-scores.pdf", args.out))
        for index in (0, 1):
            draw(plt, result["components"][index], output(f"joint-pca-strip/{prefix}-pc{index + 1}.pdf", args.out))
        print(prefix, "explained variance of PC1-PC4:", np.round(result["variance"][:4], 4).tolist())


if __name__ == "__main__":
    main()
