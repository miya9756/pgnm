"""Texture statistics: the paper's residual metrics plus three texture-specific ones.

All functions take uint8 batches (B, H, W, 3) of one texture.

* `paper_scores`: `metrics.residual.compare_rgb` with a constant clean image
  (code 128), so "conditional" W1 is the marginal W1 per channel, the spatial
  correlation error covers lags up to 8 and the color error the RGB products.
* `acf_errors`: RMS difference of normalized autocorrelations, split into
  short lags (Chebyshev radius 1..8, inside the 25×25 kernel's trusted range)
  and long lags (9..48, out of its reach).
* `sharpness_index`: Leclaire & Moisan's sharpness index (the closed-form
  successor of Blanchet & Moisan's global phase coherence), with the
  random-phase TV moments estimated by Monte Carlo. It is the -log10
  probability that a random-phase image with the same spectrum has a total
  variation as small as the texture's. About 0.3 means "as random as Gaussian";
  large values mean phase structure (edges, blobs) the spectrum cannot explain.
  Because it grows with the pixel count, the size-free relative TV deficit
  against random phase is reported next to it.
* `cross_scale`: Pearson correlation of the log local energy of the finest band
  with (a) the coarse low-pass value and (b) the coarse band-pass magnitude.
  Both vanish for any Gaussian field, whose bands are independent.
"""
from __future__ import annotations

import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.special import log_ndtr

from metrics.residual import compare_rgb, prepare_rgb

LUMA = np.array([.299, .587, .114])


def paper_scores(reference, generated):
    clean = np.full_like(reference, 128)
    return compare_rgb(prepare_rgb(clean, reference), generated)


def autocorrelation(batch, radius=48):
    """Normalized autocorrelation per channel, (3, 2r+1, 2r+1), no wrap-around."""
    x = batch.astype(np.float64)
    x -= x.mean(axis=(0, 1, 2))
    count, height, width, _ = x.shape
    shape = (fft.next_fast_len(height + radius), fft.next_fast_len(width + radius))
    spectrum = fft.rfftn(x, s=shape, axes=(1, 2), workers=4)
    power = (np.abs(spectrum) ** 2).sum(0)
    full = fft.irfftn(power, s=shape, axes=(0, 1), workers=4)
    lags = np.arange(-radius, radius + 1)
    sums = full[np.ix_(lags % shape[0], lags % shape[1])]
    pairs = count * np.outer(height - np.abs(lags), width - np.abs(lags))
    covariance = np.moveaxis(sums / pairs[..., None], -1, 0)
    return covariance / covariance[:, radius, radius][:, None, None]


def acf_errors(reference, generated, radius=48, short=8):
    a, b = autocorrelation(reference, radius), autocorrelation(generated, radius)
    lags = np.arange(-radius, radius + 1)
    chebyshev = np.maximum(np.abs(lags)[:, None], np.abs(lags)[None, :])
    rms = lambda mask: float(np.sqrt(np.mean((a[:, mask] - b[:, mask]) ** 2)))
    return dict(acf_short=rms((chebyshev > 0) & (chebyshev <= short)),
                acf_long=rms((chebyshev > short) & (chebyshev <= radius)))


def periodic_component(u):
    """Moisan's periodic + smooth decomposition: remove the border jump."""
    height, width = u.shape
    v = np.zeros_like(u)
    v[0, :] += u[-1, :] - u[0, :]
    v[-1, :] += u[0, :] - u[-1, :]
    v[:, 0] += u[:, -1] - u[:, 0]
    v[:, -1] += u[:, 0] - u[:, -1]
    q = np.arange(height)[:, None]
    r = np.arange(width)[None, :]
    denominator = 2 * np.cos(2 * np.pi * q / height) + 2 * np.cos(2 * np.pi * r / width) - 4
    denominator[0, 0] = 1
    smooth = np.fft.fft2(v) / denominator
    smooth[0, 0] = 0
    return u - np.fft.ifft2(smooth).real


def _tv(u):
    return np.abs(u - np.roll(u, 1, 0)).sum() + np.abs(u - np.roll(u, 1, 1)).sum()


def sharpness_index(image, rng, surrogates=32):
    """(sharpness index, relative TV deficit) of one image's luminance.

    The index grows with image size (it is a -log10 probability over all
    pixels), so it is reported with the size-free relative deficit
    1 - TV(u) / E[TV(random phase u)]: 0 for a Gaussian texture, larger for
    images whose phases conspire to make edges and flat regions.
    """
    u = periodic_component(image.astype(np.float64) @ LUMA)
    modulus = np.abs(np.fft.fft2(u))
    tvs = []
    for _ in range(surrogates):
        # The phase of real white noise is a valid (Hermitian) random phase.
        phase = np.exp(1j * np.angle(np.fft.fft2(rng.standard_normal(u.shape))))
        tvs.append(_tv(np.fft.ifft2(modulus * phase).real))
    z = (_tv(u) - np.mean(tvs)) / np.std(tvs, ddof=1)
    return float(-log_ndtr(z) / np.log(10)), float(1 - _tv(u) / np.mean(tvs))


def cross_scale(batch, border=16):
    energy, value, magnitude = [], [], []
    for image in batch:
        luma = image.astype(np.float64) @ LUMA / 255
        fine = luma - gaussian_filter(luma, 1.0)
        crop = (slice(border, -border),) * 2
        energy.append(np.log(gaussian_filter(fine ** 2, 4.0) + 1e-8)[crop].ravel())
        value.append(gaussian_filter(luma, 8.0)[crop].ravel())
        band = gaussian_filter(luma, 3.0) - gaussian_filter(luma, 8.0)
        magnitude.append(gaussian_filter(np.abs(band), 4.0)[crop].ravel())
    energy, value, magnitude = map(np.concatenate, (energy, value, magnitude))
    return dict(xscale_value=float(np.corrcoef(energy, value)[0, 1]),
                xscale_magnitude=float(np.corrcoef(energy, magnitude)[0, 1]))


def describe(batch, seed=0):
    """Statistics of one batch alone (no reference needed)."""
    rng = np.random.default_rng(seed)
    index, deficit = np.mean([sharpness_index(image, rng) for image in batch], axis=0)
    return dict(sharpness_index=float(index), tv_deficit=float(deficit), **cross_scale(batch))


def compare(reference, generated):
    """Distances between a reference batch and a generated batch."""
    scores = paper_scores(reference, generated)
    return dict(w1=scores["conditional_w1"],
                acf_paper=scores["spatial_correlation_error"],
                color=scores["color_correlation_error"],
                **acf_errors(reference, generated))
