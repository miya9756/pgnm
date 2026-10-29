"""Fig. 4: conditional quantile mapping, a schematic.

    python -m figures.quantile_mapping   # -> runs/figures/pgnm-conditional-quantile-mapping.pdf

A latent Gaussian value t (default -1) has cumulative probability Phi(t). At
each clean intensity x the lookup returns the residual with the same
probability under that intensity's residual distribution, n = F_x^-1(Phi(t)).
The two residual distributions are illustrative, not fitted to data: a
two-component Gaussian mixture at x = 0.25 and a Gaussian at x = 0.75. The
inverse is solved numerically to 1e-14.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import ConnectionPatch
from scipy.optimize import brentq
from scipy.special import ndtr

from .common import output, pyplot, save

COLORS = ("#1f77b4", "#ff7f0e")
INK, GUIDE = "#23303d", "#7d8e9a"
INTENSITIES = (.25, .75)


def gaussian_pdf(value, mean=0, sd=1):
    return np.exp(-.5 * ((np.asarray(value) - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def residual_cdf(value, intensity):
    if intensity == .25:
        return .30 * ndtr((np.asarray(value) + .033) / .009) + .70 * ndtr((np.asarray(value) - .021) / .021)
    return ndtr((np.asarray(value) - .004) / .065)


def residual_pdf(value, intensity):
    if intensity == .25:
        return .30 * gaussian_pdf(value, -.033, .009) + .70 * gaussian_pdf(value, .021, .021)
    return gaussian_pdf(value, .004, .065)


def quantile(probability, intensity):
    return brentq(lambda n: float(residual_cdf(n, intensity)) - probability, -1, 1, xtol=1e-14)


def plot(t_selected, path):
    plt = pyplot()
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.edgecolor": "black", "axes.linewidth": .6,
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
    fig = plt.figure(figsize=(7.2, 3.55), facecolor="white")
    left_top = fig.add_axes([.085, .56, .37, .395])
    right_top = fig.add_axes([.575, .56, .385, .395])
    left_pdf = fig.add_axes([.085, .17, .37, .37], sharex=left_top)
    right_pdf = fig.add_axes([.575, .17, .385, .37], sharex=right_top)
    t = np.linspace(-3.6, 3.6, 1601)
    n = np.linspace(-.27, .27, 5401)
    probability = float(ndtr(t_selected))
    mapped = [quantile(probability, x) for x in INTENSITIES]

    # Left: the latent Gaussian, its CDF and the probability mass below t.
    left_top.plot(t, ndtr(t), color=INK, lw=1.3, label="Gaussian")
    left_top.set(xlim=(-3.2, 3.2), ylim=(-.035, 1.055), yticks=[0, .5, 1], xticks=[-3, 0, 3])
    left_top.set_ylabel("CDF", fontsize=9)
    left_pdf.plot(t, gaussian_pdf(t), color=INK, lw=1.3, label="Gaussian")
    cut = np.append(t[t <= t_selected], t_selected)
    left_pdf.fill_between(cut, gaussian_pdf(cut), color="#b8c2ca", alpha=.6)
    left_pdf.set(xlim=(-3.2, 3.2), ylim=(0, .435), xticks=[-3, 0, 3], yticks=[0, .2, .4])
    left_pdf.set_ylabel("Density", fontsize=9)
    left_pdf.set_xlabel(r"$t_c$", fontsize=11, labelpad=4, loc="right")
    left_pdf.axvline(t_selected, color=GUIDE, lw=.9, ls="--")
    left_pdf.plot(t_selected, 0, "o", color=INK, ms=4, clip_on=False)
    left_pdf.annotate(rf"$t_c={t_selected:g}$", xy=(t_selected, 0), xytext=(0, -24), textcoords="offset points",
                      ha="center", va="top", fontsize=9, color=INK)

    # Right: the residual distribution at each clean intensity and the residual of equal probability.
    for x, color, value in zip(INTENSITIES, COLORS, mapped):
        right_top.plot(n, residual_cdf(n, x), color=color, lw=1.3, label=rf"$x_c={x:g}$")
        right_pdf.plot(n, residual_pdf(n, x), color=color, lw=1.3, label=rf"$x_c={x:g}$")
        cut = np.append(n[n <= value], value)
        right_pdf.fill_between(cut, residual_pdf(cut, x), color=color, alpha=.16)
        right_top.plot(value, probability, "o", ms=5, mfc="white", mec=color, mew=1.5, zorder=5)
        right_top.vlines(value, 0, probability, color=color, lw=.9, ls="--")
        right_pdf.axvline(value, color=color, lw=.8, ls="--", alpha=.7)
        right_pdf.plot(value, 0, "o", color=color, ms=4, clip_on=False)
    right_top.set(xlim=(-.24, .24), ylim=(-.035, 1.055), yticks=[0, .5, 1], xticks=[-.2, 0, .2])
    right_top.set_ylabel("CDF", fontsize=9)
    right_pdf.set(xlim=(-.24, .24), ylim=(0, 16), xticks=[-.2, 0, .2], yticks=[0, 5, 10, 15])
    right_pdf.set_ylabel("Density", fontsize=9)
    right_pdf.set_xlabel(r"$n_c$", fontsize=11, labelpad=4, loc="right")
    for j, (value, color) in enumerate(zip(mapped, COLORS)):
        right_pdf.annotate(rf"$n_c^{{({j + 1})}}$", xy=(value, 0), xytext=(12 if j == 0 else -12, -24),
                           textcoords="offset points", ha="center", va="top", fontsize=9, color=color,
                           arrowprops={"arrowstyle": "-", "color": color, "lw": .65, "shrinkA": 2, "shrinkB": 4})
    for axis in (left_top, right_top, left_pdf, right_pdf):
        axis.tick_params(labelsize=8, length=3, width=.5, pad=2)
        axis.set_axisbelow(True)
        axis.grid(color="#b0b0b0", lw=.45, alpha=.9)
    for axis in (left_top, right_top):  # one x-axis and one legend per column
        axis.tick_params(axis="x", labelbottom=False, bottom=False)
        axis.legend(loc="upper left", fontsize=8, frameon=True, fancybox=False, edgecolor="#b0b0b0",
                    framealpha=.9, borderpad=.3, labelspacing=.25, handlelength=1.7, handletextpad=.5)

    # The shared probability level, horizontal from the Gaussian CDF to both residual CDFs.
    left_top.plot(t_selected, probability, "o", ms=5, color=INK, zorder=6)
    left_top.hlines(probability, -3.2, t_selected, color=GUIDE, ls=":", lw=.9)
    left_top.vlines(t_selected, 0, probability, color=GUIDE, ls="--", lw=.9)
    fig.add_artist(ConnectionPatch(xyA=(t_selected, probability), coordsA="data", axesA=left_top,
                                   xyB=(max(mapped), probability), coordsB="data", axesB=right_top,
                                   arrowstyle="-|>", mutation_scale=10, lw=1, color=GUIDE, zorder=3))
    fig.canvas.draw()
    level = fig.transFigure.inverted().transform(left_top.transData.transform((t_selected, probability)))[1]
    label_x = (left_top.get_position().x1 + right_top.get_position().x0) / 2
    fig.text(label_x, level + .024, r"$\Phi(t_c)$", ha="center", va="bottom", fontsize=11, color=INK)
    save(fig, path, dpi=300)
    print("Phi(t) = %.6f, residuals %s" % (probability, ", ".join("%.6f" % v for v in mapped)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaussian-value", type=float, default=-1, help="latent value t in [-2, 2]")
    parser.add_argument("--out", type=Path, help="output folder (default: runs/figures)")
    args = parser.parse_args()
    if not -2 <= args.gaussian_value <= 2:
        parser.error("--gaussian-value must lie in [-2, 2]")
    plot(args.gaussian_value, output("pgnm-conditional-quantile-mapping.pdf", args.out))


if __name__ == "__main__":
    main()
