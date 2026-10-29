"""Conditional W1 and spatial/color correlation errors (Sec. 4 of the paper).

The residual r = y/255 - mu is grouped by clean intensity: each channel's
intensity range is cut into 16 equal bins, and bins with fewer than 512 pixels
are merged into a neighbor. The groups depend only on mu, so every method and
reference is compared on identical pixel sets.

* Conditional W1: per group, the 1-Wasserstein distance between the reference
  and generated residual distributions, averaged with the group's pixel share,
  then over channels; in 8-bit code units.
* Spatial correlation error: residuals are standardized per group, and the
  normalized autocorrelation over lags of up to 8 pixels in each direction is
  compared (RMS over all nonzero lags and channels).
* Color correlation error (RGB only): RMS difference of the six normalized
  products of different channels at the four nearest-neighbor lags.

Images of one batch (the 32 blocks of a SIDD source) are pooled, but pixel
pairs never cross from one image into another.
"""
from __future__ import annotations

import numpy as np
from scipy import fft

LAGS = tuple((y, x) for y in range(-8, 9) for x in range(-8, 9))
ZERO = LAGS.index((0, 0))
OFF_ZERO = np.arange(len(LAGS)) != ZERO


# Shared pieces -----------------------------------------------------------------

def intensity_groups(expected, mask, minimum=512):
    """16 clean-intensity bins per channel; merge bins with < minimum pixels.

    The smallest group joins whichever neighbor has the closer mean bin index
    (ties go to the lower neighbor), until every group has enough pixels.
    Returns per-pixel group indices (B, H, W, C) and per-channel group lists.
    """
    index = np.minimum((np.clip(expected, 0, 1) * 16).astype(np.int16), 15)
    maps, groups = [], []
    for channel in range(expected.shape[-1]):
        count = np.bincount(index[..., channel][mask], minlength=16)
        members = [[j] for j in range(16)]
        while len(members) > 1:
            totals = [int(count[m].sum()) for m in members]
            small = [j for j, n in enumerate(totals) if n < minimum]
            if not small:
                break
            j = min(small, key=lambda j: (totals[j], members[j][0]))
            neighbors = [k for k in (j - 1, j + 1) if 0 <= k < len(members)]
            k = min(neighbors, key=lambda k: (abs(np.mean(members[k]) - np.mean(members[j])),
                                              members[k][0]))
            low, high = sorted((j, k))
            members[low] += members[high]
            del members[high]
        lookup = np.zeros(16, dtype=np.int16)
        for j, bins in enumerate(members):
            lookup[bins] = j
        maps.append(lookup[index[..., channel]])
        groups.append([dict(bins=bins, count=int(count[bins].sum())) for bins in members])
    return np.stack(maps, -1), groups


def lag_sums(a):
    """Sum over images of a(p) * a(p + lag) for all LAGS, by zero-padded FFT."""
    a = np.asarray(a, dtype=np.float64)
    height, width = a.shape[-2:]
    shape = (fft.next_fast_len(height + 8), fft.next_fast_len(width + 8))
    spectrum = fft.rfftn(a, s=shape, axes=(-2, -1), workers=2)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).sum(0)
    correlation = fft.irfftn(power, s=shape, workers=2)
    return np.array([correlation[y % shape[0], x % shape[1]] for y, x in LAGS])


def _rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# SIDD: RGB blocks, every pixel -----------------------------------------------

def _uint8_rgb(value):
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim not in (3, 4) or array.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB HxWx3 or BxHxWx3")
    return array[None] if array.ndim == 3 else array


def prepare_rgb(clean, real):
    """Fix the intensity groups and summarize the real noisy observation once."""
    clean, real = _uint8_rgb(clean), _uint8_rgb(real)
    if clean.shape != real.shape:
        raise ValueError("Clean and real observations must have matching shapes")
    expected = clean.astype(np.float64) / 255
    mask = np.ones(expected.shape[:-1], dtype=bool)
    bins, groups = intensity_groups(expected, mask)
    pair_counts = np.rint(lag_sums(mask)).astype(np.int64)
    state = dict(clean=clean, expected=expected, bins=bins, groups=groups, pair_counts=pair_counts,
                 eligible=bool(mask.sum() >= 4096 and pair_counts.min() >= 512))
    state["real"] = summarize_rgb(state, real)
    return state


def summarize_rgb(state, observed):
    """Group histograms, standardized residual, ACF and color products."""
    residual = _uint8_rgb(observed).astype(np.float64) / 255 - state["expected"]
    white = np.empty_like(residual)
    histograms = []
    for c, groups in enumerate(state["groups"]):
        bins = state["bins"][..., c]
        count = np.array([g["count"] for g in groups], np.float64)
        values = residual[..., c]
        mean = np.bincount(bins.ravel(), weights=values.ravel(),
                           minlength=len(groups)) / np.maximum(count, 1)
        centered = values - mean[bins]
        std = np.sqrt(np.bincount(bins.ravel(), weights=(centered ** 2).ravel(),
                                  minlength=len(groups)) / np.maximum(count, 1))
        # Residual codes -256..256 in 513 bins, one histogram per group.
        codes = np.clip(np.rint(values * 255).astype(np.int16) + 256, 0, 512)
        histograms.append(np.bincount((bins * 513 + codes).ravel(),
                                      minlength=len(groups) * 513).reshape(len(groups), 513))
        white[..., c] = centered / np.maximum(std[bins], 1 / 255)
    covariance = np.array([lag_sums(white[..., c]) / np.maximum(state["pair_counts"], 1)
                           for c in range(3)])
    variance = covariance[:, ZERO]
    degenerate = variance <= 1e-12
    acf = covariance / np.maximum(variance[:, None], 1e-12)
    acf[degenerate] = 0
    from .color import color_products  # compiled with numba; only RGB scoring needs it
    return dict(histograms=histograms, acf=acf, degenerate=degenerate.tolist(),
                color=color_products(white))


def _histogram_w1(real, generated):
    """W1 between two code histograms, in normalized units (codes / 255)."""
    p = np.cumsum(real / max(real.sum(), 1))
    q = np.cumsum(generated / max(generated.sum(), 1))
    return float(np.abs(p - q).sum() / 255)


def compare_rgb(state, generated):
    """Conditional W1 and both correlation errors of one generated batch."""
    real, fake = state["real"], summarize_rgb(state, generated)
    w1 = []
    for c, groups in enumerate(state["groups"]):
        counts = np.array([g["count"] for g in groups], dtype=np.float64)
        distances = [_histogram_w1(real["histograms"][c][j], fake["histograms"][c][j])
                     for j in range(len(groups))]
        w1.append((counts / counts.sum()) @ distances)
    scores = dict(conditional_w1=255 * float(np.mean(w1)),
                  spatial_correlation_error=None, color_correlation_error=None)
    if state["eligible"] and not any(real["degenerate"]):
        scores["spatial_correlation_error"] = _rms(real["acf"][:, OFF_ZERO], fake["acf"][:, OFF_ZERO])
        scores["color_correlation_error"] = _rms(real["color"], fake["color"])
    return scores


# Film grain: gray images, continuous mu, common border -----------------------

def _gray_batch(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ..., None]
    if array.ndim != 4 or array.shape[-1] != 1:
        raise ValueError("Expected a gray HxW image")
    return array


def prepare_gray(mu, reference, border=3):
    """Groups from the continuous expected image mu; pixels inside the border."""
    expected = _gray_batch(mu)
    if expected.min() < 0 or expected.max() > 1:
        raise ValueError("Expected image mu must lie in [0, 1]")
    count, height, width = expected.shape[:3]
    mask = np.ones(expected.shape[:-1], dtype=bool)
    if border:
        mask[:, :border] = mask[:, -border:] = False
        mask[:, :, :border] = mask[:, :, -border:] = False
    bins, groups = intensity_groups(expected, mask)
    # The scored region is a rectangle, so lag pair counts are exact products.
    pair_counts = np.array([count * max(height - 2 * border - abs(dy), 0) * max(width - 2 * border - abs(dx), 0)
                            for dy, dx in LAGS], dtype=np.int64)
    state = dict(expected=expected, mask=mask, bins=bins, groups=groups, pair_counts=pair_counts,
                 eligible=bool(np.all(pair_counts[OFF_ZERO] > 0)))
    state["real"] = summarize_gray(state, reference)
    return state


def summarize_gray(state, delivered):
    """Sorted residuals per group and the standardized-residual ACF.

    A summary depends only on (mu, output), so one generated output can be
    compared with several references of the same image.
    """
    if np.asarray(delivered).dtype != np.uint8:
        raise ValueError("Expected a delivered uint8 gray observation")
    observed = _gray_batch(delivered)
    residual = observed[..., 0] / 255 - state["expected"][..., 0]
    white = np.zeros_like(residual)
    sorted_residuals = []
    for index, group in enumerate(state["groups"][0]):
        selected = state["mask"] & (state["bins"][..., 0] == index)
        values = residual[selected]
        mean, std = float(values.mean()), float(values.std())
        sorted_residuals.append(np.sort(values))
        white[selected] = (values - mean) / max(std, 1 / 255)
    covariance = lag_sums(white) / np.maximum(state["pair_counts"], 1)
    variance = covariance[ZERO]
    degenerate = bool(variance <= 1e-12)
    acf = np.zeros_like(covariance) if degenerate else covariance / variance
    return dict(sorted_residuals=sorted_residuals, acf=acf, degenerate=degenerate)


def compare_gray(state, summary):
    """Conditional W1 (exact, sorted lists of equal length) and ACF error."""
    real = state["real"]
    counts = np.array([group["count"] for group in state["groups"][0]], dtype=np.float64)
    distances = np.array([np.mean(np.abs(a - b))
                          for a, b in zip(real["sorted_residuals"], summary["sorted_residuals"])])
    scores = dict(conditional_w1=float(255 * ((counts / counts.sum()) @ distances)),
                  spatial_correlation_error=None)
    if state["eligible"] and not real["degenerate"]:
        scores["spatial_correlation_error"] = _rms(real["acf"][OFF_ZERO], summary["acf"][OFF_ZERO])
    return scores
