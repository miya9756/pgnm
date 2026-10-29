"""Fig. 3: intensity-conditioned residual densities on SIDD (S6, ISO 3200).

    python -m figures.residual_density   # -> runs/figures/sidd-conditional-residual-density.pdf

Reads only the 256x256 fitting crops of class S6_3200 listed in
datasets/sidd.json, the crops its PGNM preset is fitted on. A residual is the
noisy minus the clean 8-bit code. A pixel counts when the clean crop is flat
there (7x7 local standard deviation at most 2.5 codes in every channel, the
contrast of PGNM's flatness mask), all clean channels lie in 8..247, and it is
at least 3 pixels from the crop border. Per RGB channel, residual histograms
pool all such pixels within three clean-code intervals: 56-71, 120-135, 184-199.

Top row: the middle interval against a Gaussian and a Laplace distribution,
each fitted by maximum likelihood to the same pixels as 8-bit observations
(rounded, and clipped to [0, 255] at each pixel's clean code). Bottom row: the
three intervals. Curves are per-code histogram probabilities smoothed by a
Gaussian of 1 code; the fitted code probabilities get the same smoothing.

The histograms are cached in runs/figures/sidd-conditional-residual-density.npz
and reused on the next run; delete that file to count again.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import laplace

from .common import output, pyplot, save

CLASS = "S6_3200"
INTERVALS = ((56, 72), (120, 136), (184, 200))  # clean codes [lo, hi)
CODES = np.arange(-255, 256, dtype=np.float64)   # residual codes, histogram columns
STEM = "sidd-conditional-residual-density"


def histograms():
    """int64 (3, 256, 511): counts per channel, clean code and residual code + 255."""
    from experiments import sidd
    from pgnm.fitting import flat_pixel_contrast
    counts = np.zeros((3, 256, 511), np.int64)
    for clean, noisy in sidd.training_pairs(CLASS)():
        clean, noisy = (np.rint(a * 255).astype(np.uint8) for a in (clean, noisy))  # back to 8-bit codes
        mask = (flat_pixel_contrast(clean) <= 2.5) & (clean.min(-1) >= 8) & (clean.max(-1) <= 247)
        mask[:3] = mask[-3:] = False
        mask[:, :3] = mask[:, -3:] = False
        residual = noisy.astype(np.int64) - clean
        for c in range(3):
            index = clean[..., c][mask].astype(np.int64) * 511 + residual[..., c][mask] + 255
            counts[c] += np.bincount(index, minlength=256 * 511).reshape(256, 511)
    return counts


def code_log_probabilities(location, log_scale, family, clean_codes):
    """log P(residual code | clean code) of a rounded, clipped Gaussian or Laplace.

    Inner codes get the mass of [r - 1/2, r + 1/2]; the codes where the output
    clips (residual -clean and 255 - clean) get the whole tail; others -inf.
    """
    cdf, sf = (log_ndtr, lambda z: log_ndtr(-z)) if family == "gaussian" else (laplace.logcdf, laplace.logsf)
    lower, upper = -clean_codes, 255 - clean_codes
    width = np.exp(log_scale)
    low = (CODES - .5 - location) / width
    high = (CODES + .5 - location) / width
    # log(F(high) - F(low)), from the side of the median for accuracy in the tails
    a = np.where(low > 0, sf(low), cdf(high))
    b = np.where(low > 0, sf(high), cdf(low))
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = a + np.log(-np.expm1(np.minimum(b - a, 0)))
    shape = (len(clean_codes), len(CODES))
    result = np.broadcast_to(inner, shape).copy()
    at_lower, at_upper = CODES[None] == lower, CODES[None] == upper
    result[at_lower] = np.broadcast_to(cdf((lower + .5 - location) / width), shape)[at_lower]
    result[at_upper] = np.broadcast_to(sf((upper - .5 - location) / width), shape)[at_upper]
    result[(CODES[None] < lower) | (CODES[None] > upper)] = -np.inf
    return result


def fitted_probabilities(joint, lo, hi, family, initial):
    """Maximum-likelihood fit to the interval's pixels -> its residual code probabilities."""
    count = joint.sum()
    clean_codes = np.arange(lo, hi)[:, None]
    occupied = joint > 0

    def objective(theta):
        return -float(np.sum((joint[occupied] / count) * code_log_probabilities(*theta, family, clean_codes)[occupied]))

    fit = minimize(objective, initial, method="Nelder-Mead", options={"maxiter": 1000, "xatol": 1e-7, "fatol": 1e-9})
    if not fit.success:
        raise RuntimeError(f"{family} fit failed: {fit.message}")
    clean_weights = joint.sum(1) / count
    return clean_weights @ np.exp(code_log_probabilities(*fit.x, family, clean_codes))


def densities(counts):
    """[channel][interval] -> dict of smoothed real, Gaussian and Laplace code densities."""
    result = []
    for c in range(3):
        row = []
        for lo, hi in INTERVALS:
            joint = counts[c, lo:hi].astype(np.float64)  # clean code x residual code
            probability = joint.sum(0) / joint.sum()
            mean = float(probability @ CODES)
            sd = float(np.sqrt(probability @ ((CODES - mean) ** 2)))
            median = float(CODES[np.searchsorted(np.cumsum(probability), .5)])
            spread = float(probability @ np.abs(CODES - median))
            gaussian = fitted_probabilities(joint, lo, hi, "gaussian", [mean, np.log(sd)])
            double_exponential = fitted_probabilities(joint, lo, hi, "laplace", [median, np.log(max(spread, .1))])
            row.append({name: gaussian_filter1d(p, 1, mode="constant") for name, p in
                        (("real", probability), ("gaussian", gaussian), ("laplace", double_exponential))})
        result.append(row)
    return result


def plot(curves, path):
    plt = pyplot()
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                         "font.size": 8, "axes.edgecolor": "black", "axes.linewidth": .6,
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    fig.subplots_adjust(left=.083, right=.984, bottom=.107, top=.976, wspace=.28, hspace=.36)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    legend = dict(fontsize=7, frameon=True, fancybox=False, edgecolor="#b0b0b0", framealpha=.9,
                  borderpad=.3, labelspacing=.25, handlelength=1.7, handletextpad=.5)
    for c, channel in enumerate("RGB"):
        top, bottom = axes[0, c], axes[1, c]
        middle = curves[c][1]
        top.plot(CODES, middle["real"], color=colors[0], lw=1.2, label="Real")
        top.plot(CODES, middle["gaussian"], color=colors[1], lw=1.0, ls="--", label="Gaussian")
        top.plot(CODES, middle["laplace"], color=colors[2], lw=1.0, ls=":", label="Laplace")
        top.set_xlabel(f"{channel} residual, clean 120-135", fontsize=8)
        top.legend(loc="upper left", **legend)
        for (lo, hi), curve, color in zip(INTERVALS, curves[c], colors):
            bottom.plot(CODES, curve["real"], color=color, lw=1.2, label=f"{lo}-{hi - 1}")
        bottom.set_xlabel(f"{channel} residual (8-bit codes)", fontsize=8)
        bottom.legend(loc="upper right", title="Clean code", title_fontsize=7, **legend)
        for panel in (top, bottom):
            panel.set_xlim(-205, 205)
            panel.set_xticks([-200, -100, 0, 100, 200])
            panel.set_ylim(bottom=0)
            panel.set_axisbelow(True)
            panel.grid(color="#b0b0b0", lw=.45, alpha=.9)
            panel.tick_params(labelsize=7, width=.5, length=3, pad=2)
    for row in axes:
        row[0].set_ylabel("Density (per code)", fontsize=8)
    save(fig, path, dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="output folder (default: runs/figures)")
    args = parser.parse_args()
    cache = output(STEM + ".npz", args.out)
    if cache.exists():
        counts = np.load(cache)["counts"]
    else:
        counts = histograms()
        np.savez_compressed(cache, counts=counts)
    plot(densities(counts), output(STEM + ".pdf", args.out))


if __name__ == "__main__":
    main()
