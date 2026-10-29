"""Zhang23: Zhang, Wang, Tian & Pappas, "Film Grain Rendering and Parameter
Estimation", ACM TOG 42(4), 2023. Our implementation (fixed radius, sigma_r=0).

  V(u)   = r^2/(2 s^2) * int_0^2 exp(-(r x)^2/(4 s^2)) C_B(u, x, 1) x dx   Eq. (13)
  rho(d) ~ exp(-d^2 / (4 s^2))                                             Eq. (16)
  n = h * (sqrt(V(u)) z),  sum h^2 = 1,  h from sqrt(PSD) of Eq. (16)      Alg. 1

Two parameters per layer: grain radius r and filter width s. The estimator is
the paper's (Eqs. 34-40): s from the decay of correlation between lag 1 and
lag sqrt(2), which cancels white noise, then r from the variance through the
small-grain curve. The paper states it for one flat patch; here moments are
pooled over every flat pixel of the training pairs, r is a least-squares fit
over intensity buckets, refined against the exact Eq. (13). Like PGNM, the
grain is added to the training-fitted blur mu = B*x.

    python -m baselines.zhang23 fit      # presets/film/zhang23.json, all radii
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import minimize_scalar

KNOTS = 17      # strength curve knots over clean intensity
ACF_RADIUS = 16
_X = (np.arange(160) + 0.5) * (2.0 / 160)
_A = 2 * np.arccos(_X / 2) - 0.5 * _X * np.sqrt(np.maximum(4 - _X ** 2, 0))


def _covariance_unit(u):
    u = np.clip(np.asarray(u, float), 0, 0.9995)[:, None]
    density = -np.log1p(-u) / np.pi
    return (1 - u) ** 2 * np.expm1(density * _A[None, :])


def variance(u, r, s, approx=False):
    """Eq. (13); approx=True drops the filter attenuation (small grain)."""
    attenuation = 1.0 if approx else np.exp(-((r * _X) ** 2) / (4 * s * s))
    v = r * r / (2 * s * s) * (attenuation * _covariance_unit(u) * _X).sum(1) * (2.0 / 160)
    return np.where(np.asarray(u) >= 1.0, 0.0, v)


def kernel1d(s, size=64, radius=16):
    """Separable unit-energy filter whose PSD is that of Eq. (16)."""
    d = np.minimum(np.arange(size), size - np.arange(size))
    spectrum = np.sqrt(np.maximum(np.real(np.fft.fft(np.exp(-d * d / (4 * s * s)))), 0))
    k = np.real(np.fft.ifft(spectrum))
    k = np.concatenate([k[-radius:], k[:radius + 1]])
    return k / np.sqrt((k ** 2).sum())


# Fitting -------------------------------------------------------------------------

def _strength_curve(pairs, masks):
    """Residual RMS at 17 clean-intensity knots over flat pixels."""
    channels = pairs[0][0].shape[2]
    s2, n = np.zeros((channels, KNOTS)), np.zeros((channels, KNOTS))
    for (clean, noisy), m in zip(pairs, masks):
        r = (noisy - clean).astype(np.float64)
        for c in range(channels):
            b = np.rint(clean[..., c][m] * (KNOTS - 1)).astype(int)
            s2[c] += np.bincount(b, weights=r[..., c][m] ** 2, minlength=KNOTS)[:KNOTS]
            n[c] += np.bincount(b, minlength=KNOTS)[:KNOTS]
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt(s2 / n)
    sigma[n < 500] = np.nan
    return sigma, n


def _filled(sigma):
    out = np.array(sigma, float)
    x = np.arange(out.shape[1])
    for c in range(out.shape[0]):
        ok = np.isfinite(out[c])
        out[c] = np.interp(x, x[ok], out[c][ok]) if ok.any() else 0.0
    return out


def _whitened(pairs, masks, sigma, guard=2.5):
    """Residual / sigma(I); also drop pixels within 2.5 sigma of 0 or 1."""
    out = []
    for (clean, noisy), m in zip(pairs, masks):
        w = np.zeros(clean.shape, np.float64)
        keep = np.zeros(clean.shape, bool)
        for c in range(clean.shape[2]):
            s = np.maximum(np.interp(np.clip(clean[..., c], 0, 1), np.linspace(0, 1, KNOTS), sigma[c]), 1e-6)
            w[..., c] = (noisy[..., c] - clean[..., c]) / s
            keep[..., c] = m & (clean[..., c] > guard * s) & (clean[..., c] < 1 - guard * s)
        out.append((w, keep))
    return out


def _autocorrelation(whitened):
    """Pooled masked autocorrelation per channel with exact per-lag pair counts."""
    channels, R = whitened[0][0].shape[2], ACF_RADIUS
    numerator = np.zeros((channels, 2 * R + 1, 2 * R + 1))
    denominator = np.zeros_like(numerator)
    lags = np.arange(-R, R + 1)
    for w, keep in whitened:
        height, width, _ = w.shape
        for c in range(channels):
            mask = keep[..., c].astype(np.float64)
            if mask.sum() < 64:
                continue
            a = w[..., c] * mask
            a = a - (a.sum() / mask.sum()) * mask
            fa = np.fft.rfft2(a, s=(2 * height, 2 * width))
            fm = np.fft.rfft2(mask, s=(2 * height, 2 * width))
            numerator[c] += np.fft.irfft2(np.abs(fa) ** 2, s=(2 * height, 2 * width))[np.ix_(lags, lags)]
            denominator[c] += np.fft.irfft2(np.abs(fm) ** 2, s=(2 * height, 2 * width))[np.ix_(lags, lags)]
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = numerator / np.maximum(denominator, 1.0)
    return covariance / np.maximum(covariance[:, R:R + 1, R:R + 1], 1e-12)


def _estimate(sigma, count, acf, bits):
    R = ACF_RADIUS
    rho1 = 0.5 * (acf[R, R + 1] + acf[R + 1, R])
    rho2 = 0.5 * (acf[R + 1, R + 1] + acf[R + 1, R - 1])
    s, source = None, "ratio"
    # The two-lag ratio (Eq. 39) cancels any white component, including the
    # renderer's Monte Carlo noise; the lag-1 form (Eq. 34) is the fallback.
    if rho1 > 0 and rho2 > 0 and rho2 < rho1:
        s = np.sqrt(1.0 / (4 * np.log(rho1 / rho2)))
    if s is None or not np.isfinite(s):
        source = "lag1"
        s = np.sqrt(-1.0 / (4 * np.log(rho1))) if 0 < rho1 < 1 else 0.3
    s = float(np.clip(s, 0.3, 4.0))
    u = np.linspace(0, 1, len(sigma))
    ok = np.isfinite(sigma) & (u > 0.03) & (u < 0.97)
    quantization = (2.0 ** (-2 * bits) / 12) if bits else 0.0
    V = np.maximum(sigma[ok] ** 2 - quantization, 1e-10)
    w = count[ok]
    small_grain = 2 * variance(u[ok], 1, 1, approx=True)
    a2 = 2 * (w * V * small_grain).sum() / max((w * small_grain * small_grain).sum(), 1e-12)
    r_paper = float(np.sqrt(max(a2, 1e-12)) * s)
    loss = lambda log_r: (w * (variance(u[ok], np.exp(log_r), s) - V) ** 2).sum()
    r = float(np.exp(minimize_scalar(loss, bounds=(np.log(1e-3), np.log(3.0)), method="bounded").x))
    return dict(r=r, r_paper=r_paper, s=s, rho1=float(rho1), rho_sqrt2=float(rho2), s_from=source)


def fit(pairs, bits=8):
    """pairs: (mu, noisy) float32 HxWx1 crops; flat pixels as in pgnm.fitting."""
    from pgnm.fitting import flat_pixel_contrast, flat_pixel_threshold
    rgb = [(np.repeat(c, 3, -1), np.repeat(y, 3, -1)) for c, y in pairs]
    threshold = flat_pixel_threshold(rgb)
    masks = [flat_pixel_contrast(clean) <= threshold for clean, _ in rgb]
    sigma, count = _strength_curve(pairs, masks)
    acf = _autocorrelation(_whitened(pairs, masks, _filled(sigma)))
    return dict(layers=[_estimate(sigma[c], count[c], acf[c], bits) for c in range(sigma.shape[0])])


# Synthesis -----------------------------------------------------------------------

def synthesize(mu, params, rng):
    """Additive grain for mu (H, W, C) in [0, 1]: Alg. 1 with edge padding."""
    R = 16
    padded = np.pad(mu, ((R, R), (R, R), (0, 0)), mode="edge")
    u = np.linspace(0, 1, 513)
    out = []
    for c in range(mu.shape[2]):
        layer = params["layers"][c]
        noise = np.sqrt(np.interp(padded[..., c], u, variance(u, layer["r"], layer["s"]))
                        ) * rng.standard_normal(padded.shape[:2])
        k = kernel1d(layer["s"])
        noise = convolve1d(convolve1d(noise, k, axis=0, mode="constant"), k, axis=1, mode="constant")
        out.append(noise[R:-R, R:-R])
    return np.stack(out, -1)


def main():
    from experiments import config, film
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("fit",))
    parser.parse_args()
    blur = config.read_json(config.REPO / "presets/film/blur.json")
    result = {}
    for radius in film.RADII:
        key = film.radius_key(radius)
        pairs = [(mu[..., :1], noisy[..., :1]) for mu, noisy in film.training_pairs(radius, blur[key])]
        result[key] = fit(pairs)
        print("fitted Zhang23", key, result[key]["layers"][0], flush=True)
    config.write_json(config.runs("film/zhang23.json"), result)


if __name__ == "__main__":
    main()
