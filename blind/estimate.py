"""PGNM parameters from one noisy image and a blind denoiser D.

Every estimator receives only `noisy` (uint8 RGB) and `denoise`. The denoised
image x̂ = D(y) is used in two roles only: as the local intensity that
conditions the lookup, and as a map of where the image has structure.

* `naive`: `pgnm.fit([(x̂, y)])`, i.e. x̂ treated as the clean image.
* `masked`: residual statistics only where x̂ is locally flat at two scales
  (flattest 40% within each intensity band). (`alignment_mask`, which also
  rejected residuals correlated with x̂'s fine detail, is not used: noise the
  denoiser leaves in x̂ is anti-correlated with the residual everywhere, so it
  rejected 94% of pixels under the DnCNN.)
* `envelope`: additionally, only 32×32 tiles whose residual variance is close
  to the lower envelope of all tiles of the same intensity. Noise is the same
  everywhere at a given intensity; erased texture can only add energy.
* `calibrated`: indirect inference on top of `envelope`. Simulate the whole
  chain with the current estimate (x̂ + PGNM noise -> D -> the same
  statistics), and move the estimate by the gap between the observed and
  simulated statistics. What D does to the noise (residual low-frequency noise,
  partial removal, clipped tails) and the envelope's own selection bias appear
  in both and cancel. Calibrated statistics are the correlations and the
  residual strength per channel and intensity band; the lookup is rescaled
  around its median, so its shape and offset stay as observed (the offset is
  invisible to residuals against x̂ and cannot be identified from one image).

Statistics, and the parameters built from them, are PGNM's own: the 3×3×33×33
cross-correlation of pilot-normalized residuals (-> kernel) and the residual
histogram per channel and intensity code (-> lookup), from `pgnm.fitting`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from pgnm import PGNM
from pgnm import fitting
from pgnm.parameters import SCHEMA, validate

LUMA = np.array([.299, .587, .114])


def to_float(image):
    return np.asarray(image, np.float32) / 255


def to_uint8(image):
    return np.floor(np.clip(image, 0, 1) * 255 + .5).astype(np.uint8)


# Masks ----------------------------------------------------------------------------

def structure_mask(clean, keep=0.4, bands=16):
    """Flattest `keep` fraction of pixels within each intensity band, two scales."""
    contrast = np.maximum(fitting.flat_pixel_contrast(clean, 7), fitting.flat_pixel_contrast(clean, 15))
    luma = gaussian_filter(clean @ LUMA, 2.0)
    band = np.minimum((np.clip(luma, 0, 1) * bands).astype(int), bands - 1)
    mask = np.zeros(band.shape, bool)
    for b in range(bands):
        selected = band == b
        if selected.sum():
            mask |= selected & (contrast <= np.quantile(contrast[selected], keep))
    return mask


def alignment_mask(clean, noisy, limit=0.3, window=9):
    """False where the residual correlates locally with x̂'s fine detail."""
    detail = clean @ LUMA - gaussian_filter(clean @ LUMA, 2.0)
    residual = (noisy - clean) @ LUMA
    mean = lambda a: uniform_filter(a, window)
    covariance = mean(detail * residual) - mean(detail) * mean(residual)
    spread = np.sqrt(np.maximum(mean(detail ** 2) - mean(detail) ** 2, 1e-12)
                     * np.maximum(mean(residual ** 2) - mean(residual) ** 2, 1e-12))
    return np.abs(covariance / spread) <= limit


def envelope_mask(clean, noisy, mask, tile=32, quantile=0.25, factor=1.5, bins=8):
    """Keep tiles whose residual variance is within `factor` of the lower envelope."""
    height, width, _ = clean.shape
    residual = noisy - clean
    tiles = []
    for top in range(0, height - tile + 1, tile):
        for left in range(0, width - tile + 1, tile):
            m = mask[top:top + tile, left:left + tile]
            if m.sum() < 256:
                continue
            r = residual[top:top + tile, left:left + tile][m]
            tiles.append((top, left, r.var(axis=0), clean[top:top + tile, left:left + tile][m].mean(axis=0)))
    if len(tiles) < 8:
        return mask
    variance = np.array([t[2] for t in tiles])
    intensity = np.array([t[3] for t in tiles])
    keep = np.ones(len(tiles), bool)
    centers = (np.arange(bins) + .5) / bins
    for c in range(3):
        b = np.minimum((intensity[:, c] * bins).astype(int), bins - 1)
        supported = [k for k in range(bins) if (b == k).sum() >= 4]
        if not supported:
            continue
        envelope = np.exp(np.interp(intensity[:, c], centers[supported],
                                    [np.log(np.quantile(variance[b == k, c], quantile)) for k in supported]))
        keep &= variance[:, c] <= factor * envelope
    selected = np.zeros_like(mask)
    for (top, left, _, _), k in zip(tiles, keep):
        if k:
            selected[top:top + tile, left:left + tile] = True
    return mask & selected


# Statistics and parameters ----------------------------------------------------------

def statistics(clean, noisy, mask, histogram_mask=None):
    """(cross-correlation 3×3×33×33, residual histogram 3×256×511) on masked pixels.

    `histogram_mask` (default: `mask`) selects the pixels of the histogram.
    """
    histogram_mask = mask if histogram_mask is None else histogram_mask
    accumulator = fitting.PilotStrengthAccumulator()
    accumulator.add(clean.astype(np.float64), (noisy - clean).astype(np.float64), mask)
    try:
        pilot = accumulator.fit(min_pixels=500, max_censored=0.35)
    except ValueError:
        rms = np.sqrt(np.mean((noisy - clean)[mask] ** 2, axis=0))
        pilot = np.repeat(rms[:, None], fitting.PILOT_KNOTS, axis=1)
    normalized, guarded = fitting.normalize_residuals(clean, noisy, mask, pilot)
    if guarded.sum() < 2000:
        guarded = mask
    correlation, _ = fitting.measure_cross_correlation([(normalized, guarded)])
    histogram = np.zeros((3, 256, 511))
    intensity = np.clip(np.rint(clean * 255), 0, 255).astype(np.int64)
    code = np.clip(np.rint((noisy - clean) * 255), -255, 255).astype(np.int64) + 255
    for c in range(3):
        index = (intensity[..., c] * 511 + code[..., c])[histogram_mask]
        histogram[c] = np.bincount(index, minlength=256 * 511).reshape(256, 511)
    return correlation, histogram


def parameters(correlation, lookup, recipe):
    kernels, details = fitting.factor_cross_channel_kernels(correlation, 25)
    return validate(dict(
        schema=SCHEMA, method="ours", name="PGNM", kernels=kernels.tolist(),
        marginal=dict(t=fitting.GAUSSIAN_KNOTS.tolist(), g=np.clip(lookup, -1, 1).tolist(), step=1.0),
        provenance=dict(status="fitted", recipe=recipe, factorization=details)))


def render_fresh(params, clean, rng):
    """One PGNM sample on `clean` with a fresh latent field (no template bank)."""
    model = PGNM(params, workers=4)
    field = model._build_noise_field(clean.shape, rng).astype(np.float32)
    noisy = np.empty_like(clean)
    model._render_noise_window(field, clean, 0, 0, noisy)
    return noisy


# Estimators -------------------------------------------------------------------------

def paper_recipe(clean, noisy, recipe):
    """`pgnm.fit` for one pair: flat-pixel correlations, all-pixel lookup.

    `pgnm.fit` itself needs flat pixels at two or more intensity knots for its
    pilot strength curve, which a single image often lacks; `statistics` falls
    back to one RMS per channel there, as everywhere in this module.
    """
    threshold = fitting.flat_pixel_threshold([(clean, noisy)])
    flat = fitting.flat_pixel_contrast(clean) <= threshold
    correlation, histogram = statistics(clean, noisy, flat, np.ones_like(flat))
    return parameters(correlation, fitting.build_residual_lookup(histogram), recipe)


def oracle(noisy, clean):
    """Upper bound: the paper's recipe on the true (clean, noisy) pair of this image."""
    return paper_recipe(to_float(clean), to_float(noisy), "single image: true clean")


def naive(noisy, denoise):
    return paper_recipe(to_float(denoise(noisy)), to_float(noisy), "single image: denoised as clean")


def _masked_statistics(clean, noisy, envelope):
    mask = structure_mask(clean)
    if envelope:
        mask = envelope_mask(clean, noisy, mask)
    return statistics(clean, noisy, mask)


def masked(noisy, denoise, envelope=False):
    clean, y = to_float(denoise(noisy)), to_float(noisy)
    correlation, histogram = _masked_statistics(clean, y, envelope)
    return parameters(correlation, fitting.build_residual_lookup(histogram),
                      "single image: envelope mask" if envelope else "single image: structure mask")


def envelope(noisy, denoise):
    return masked(noisy, denoise, envelope=True)


def strength_curves(clean, noisy, mask, bins=16, minimum=200):
    """Residual std per channel and intensity band of x̂ (NaN where unsupported)."""
    band = np.minimum((clean * bins).astype(int), bins - 1)
    residual = noisy - clean
    curves = np.full((3, bins), np.nan)
    for c in range(3):
        for b in range(bins):
            selected = mask & (band[..., c] == b)
            if selected.sum() >= minimum:
                curves[c, b] = residual[..., c][selected].std()
    return curves


def _masks(clean, noisy):
    return envelope_mask(clean, noisy, structure_mask(clean))


def rescale_lookup(lookup, ratio, bins=16):
    """Scale each (channel, intensity code) quantile function around its median."""
    centers = (np.arange(bins) + .5) / bins
    codes = np.arange(256) / 255
    scaled = lookup.copy()
    for c in range(3):
        supported = np.isfinite(ratio[c])
        factor = np.interp(codes, centers[supported], ratio[c, supported]) if supported.any() else np.ones(256)
        median = lookup[c, :, len(fitting.GAUSSIAN_KNOTS) // 2][:, None]
        scaled[c] = median + (lookup[c] - median) * factor[:, None]
    return np.clip(scaled, -1, 1)


def calibrated(noisy, denoise, iterations=4, simulations=2, seed=0, history=None):
    rng = np.random.default_rng(seed)
    clean, y = to_float(denoise(noisy)), to_float(noisy)
    mask = _masks(clean, y)
    observed_correlation, histogram = statistics(clean, y, mask)
    observed_strength = strength_curves(clean, y, mask)
    correlation, lookup = observed_correlation.copy(), fitting.build_residual_lookup(histogram)
    for _ in range(iterations):
        params = parameters(correlation, lookup, "calibration step")
        simulated_correlation, simulated_strength = 0, []
        for _ in range(simulations):
            simulated = to_float(to_uint8(render_fresh(params, clean, rng)))
            simulated_clean = to_float(denoise(to_uint8(simulated)))
            simulated_mask = _masks(simulated_clean, simulated)
            c, _ = statistics(simulated_clean, simulated, simulated_mask)
            simulated_correlation = simulated_correlation + c / simulations
            simulated_strength.append(strength_curves(simulated_clean, simulated, simulated_mask))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = observed_strength / np.nanmean(simulated_strength, axis=0)
        if history is not None:
            history.append(dict(strength_gap=float(np.nanmean(np.abs(np.log(ratio)))),
                                correlation_gap=float(np.abs(observed_correlation - simulated_correlation).max())))
        correlation = correlation + (observed_correlation - simulated_correlation)
        lookup = rescale_lookup(lookup, np.clip(ratio, .5, 2))
    return parameters(correlation, lookup, f"single image: calibrated x{iterations}")


ESTIMATORS = dict(naive=naive, masked=masked, envelope=envelope, calibrated=calibrated)
