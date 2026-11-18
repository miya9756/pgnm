"""Single-scale PGNM and a coarse-to-fine pyramid of PGNMs, both fitted on textures alone.

Single scale: a texture is the "noisy" image of a constant clean image
(code 128), so `pgnm.fit` estimates the 25×25 RGB kernel from the texture's
correlations and the residual lookup collapses to one intensity row: its
marginal. That is a stationary Gaussian field followed by a pointwise map.

Pyramid: Gaussian pyramid G_0 (full size) ... G_L (coarsest). The coarsest
level is a single-scale PGNM. Every finer level k is a PGNM whose "clean
image" is the expanded coarser level, fitted on the pairs (up(G_{k+1}), G_k):

    G_k = up(G_{k+1}) + g_k(up(G_{k+1}), A_k * z_k)

The kernel A_k captures the band's correlations, which now reach 2^k times
further, and the lookup g_k makes the band's distribution (strength, skew)
depend on the coarse value at that pixel: a cross-scale dependence that no
sum of Gaussian fields can express.

Part of every real band is fixed by the coarse image (it re-sharpens what
up() blurred). An independent latent cannot produce it, so with `predict`
each level first adds a ridge-regularized linear prediction P_k:
G_k = P_k(up(G_{k+1})) + g_k(P_k(up(G_{k+1})), A_k * z_k).

Rendering draws one fresh latent field of exactly the frame size per sample
(instead of the paper's four-template bank), so samples are independent.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from pgnm import PGNM, fitting
from pgnm.parameters import SCHEMA, validate

GRAY = 128 / 255


def normal_scores(pairs, bins=32):
    """Per channel and clean-intensity bin, residual -> Phi^-1(mid-rank): PGNM's latent t.

    PGNM's generative model is y = x + g(x, t) with g monotone in t, so the
    latent value behind an observed residual is its normal score. Correlating
    normal scores estimates the latent correlation directly; correlating the
    residuals themselves estimates it only when the residuals are Gaussian.
    """
    clean = np.stack([x for x, _ in pairs]).astype(np.float64)
    residual = np.stack([y for _, y in pairs]).astype(np.float64) - clean
    scores = np.empty_like(residual)
    group = np.minimum((np.clip(clean, 0, 1) * bins).astype(int), bins - 1)
    for c in range(3):
        values, groups, out = residual[..., c].ravel(), group[..., c].ravel(), np.empty(residual[..., c].size)
        for b in np.unique(groups):
            index = np.flatnonzero(groups == b)
            out[index] = ndtri((rankdata(values[index]) - .5) / len(index))
        scores[..., c] = out.reshape(residual.shape[:-1])
    return list(scores)


def fit(pairs, gaussianize=False):
    """`pgnm.fit` with every pixel treated as flat and a constant-pilot fallback.

    The pilot strength curve only rescales residuals before correlations are
    measured. `pgnm.fit` needs two supported intensity knots to interpolate it,
    which a constant clean image cannot give; there one RMS per channel is the
    exact pilot. With `gaussianize`, correlations are measured on normal scores
    instead (see `normal_scores`). Everything else (correlation, kernel
    factorization, residual lookup) is `pgnm.fitting` unchanged.
    """
    pairs = list(pairs)
    flat = [np.ones(clean.shape[:2], bool) for clean, _ in pairs]
    if gaussianize:
        samples = list(zip(normal_scores(pairs), flat))
    else:
        try:
            pilot = fitting.fit_pilot_strength(pairs, threshold=np.inf)
        except ValueError:
            rms = np.sqrt(np.mean([np.mean((y - x) ** 2, axis=(0, 1)) for x, y in pairs], axis=0))
            pilot = np.repeat(rms[:, None], fitting.PILOT_KNOTS, axis=1)
        samples = [fitting.normalize_residuals(x, y, m, pilot) for (x, y), m in zip(pairs, flat)]
        if sum(int(mask.sum()) for _, mask in samples) == 0:
            samples = [(normalized, m) for (normalized, _), m in zip(samples, flat)]
    correlation, _ = fitting.measure_cross_correlation(samples)
    kernels, details = fitting.factor_cross_channel_kernels(correlation, 25)
    lookup = fitting.build_residual_lookup(fitting.measure_residual_histogram(pairs))
    recipe = "textures: all pixels flat" + (", normal-score correlation" if gaussianize else "")
    return validate(dict(
        schema=SCHEMA, method="ours", name="PGNM", kernels=kernels.tolist(),
        marginal=dict(t=fitting.GAUSSIAN_KNOTS.tolist(), g=lookup.tolist(), step=1.0),
        provenance=dict(status="fitted", recipe=recipe, factorization=details),
    ))


def _float(batch):
    return [image.astype(np.float32) / 255 for image in batch]


def _render_fresh(model, clean, rng):
    field = model._build_noise_field(clean.shape, rng).astype(np.float32)
    noisy = np.empty_like(clean)
    model._render_noise_window(field, clean, 0, 0, noisy)
    return noisy


def _up(image, like):
    return cv2.pyrUp(image, dstsize=(like.shape[1], like.shape[0]))


class SingleScale:
    def __init__(self, params, label="PGNM"):
        self.model = PGNM(params, workers=4)
        self.label = label

    @classmethod
    def fit(cls, batch, gaussianize=False):
        images = _float(batch)
        return cls(fit([(np.full_like(y, GRAY), y) for y in images], gaussianize),
                   "PGNM, normal scores" if gaussianize else "PGNM")

    def sample(self, shape, rng):
        return _render_fresh(self.model, np.full(shape, GRAY, np.float32), rng)


class Predictor:
    """Per-channel linear filter (2r+1)² + offset: the band part the coarse image fixes."""

    def __init__(self, kernels, offsets):
        self.kernels, self.offsets = kernels, offsets

    @classmethod
    def fit(cls, coarse, fine, radius=4, stride=2, ridge_weight=1e-2):
        kernels, offsets = [], []
        side = 2 * radius + 1
        for c in range(3):
            rows, target = [], []
            for x, y in zip(coarse, fine):
                padded = np.pad(x[..., c], radius, mode="reflect")
                h, w = x.shape[:2]
                shifts = [padded[dy:dy + h:stride, dx:dx + w:stride].ravel()
                          for dy in range(side) for dx in range(side)]
                rows.append(np.stack(shifts + [np.ones_like(shifts[0])], 1))
                target.append((y[..., c] - x[..., c])[::stride, ::stride].ravel())
            design = np.concatenate(rows).astype(np.float64)
            design[:, :-1] -= design[:, :-1].mean(0)   # center so the ridge spares the offset
            gram = design.T @ design
            # Up-sampled images are band-limited, so plain least squares deconvolves
            # wildly; a ridge keeps the filter to what the coarse image really fixes.
            ridge = ridge_weight * np.trace(gram[:-1, :-1]) / (side * side)
            gram[np.arange(side * side), np.arange(side * side)] += ridge
            solution = np.linalg.solve(gram, design.T @ np.concatenate(target).astype(np.float64))
            solution[-1] -= solution[:-1] @ np.concatenate(rows)[:, :-1].mean(0)
            kernels.append(solution[:-1].reshape(side, side).astype(np.float32))
            offsets.append(float(solution[-1]))
        return cls(kernels, offsets)

    def __call__(self, x):
        detail = np.stack([cv2.filter2D(x[..., c], -1, self.kernels[c], borderType=cv2.BORDER_REFLECT_101)
                           + self.offsets[c] for c in range(3)], -1)
        return np.clip(x + detail, 0, 1)


QUANTILES = 4096


def quantile_table(images):
    """Per-channel quantiles of the pooled pixels: (QUANTILES + 1, 3)."""
    pixels = np.concatenate([image.reshape(-1, 3) for image in images])
    return np.quantile(pixels, np.linspace(0, 1, QUANTILES + 1), axis=0).astype(np.float32)


def match_histogram(image, table):
    """Exact histogram specification by rank, channel by channel (Heeger & Bergen)."""
    matched = np.empty_like(image)
    grid = np.linspace(0, 1, len(table))
    for c in range(image.shape[-1]):
        values = image[..., c].ravel()
        rank = np.empty(values.size)
        rank[np.argsort(values, kind="stable")] = (np.arange(values.size) + .5) / values.size
        matched[..., c] = np.interp(rank, grid, table[:, c]).reshape(image.shape[:-1])
    return matched


class Pyramid:
    """Coarse-to-fine PGNMs; with `predict`, each level first adds a linear prediction.

    `match` adds histogram specification to the training marginals: "final"
    only on the output, "levels" on every pyramid level as it is synthesized.
    """

    def __init__(self, params, predictors=None, label="Pyramid PGNM", tables=None, match=None):
        self.models = [PGNM(p, workers=4) for p in params]   # finest first, coarsest last
        self.predictors = predictors or [None] * (len(params) - 1)
        self.label = label
        if match not in (None, "final", "levels"):
            raise ValueError("match must be None, 'final' or 'levels'")
        self.tables, self.match = tables, match

    @staticmethod
    def gaussian_pyramid(image, levels):
        pyramid = [image]
        for _ in range(levels):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid

    @classmethod
    def fit(cls, batch, levels=3, predict=False, gaussianize=False, match=None):
        pyramids = [cls.gaussian_pyramid(y, levels) for y in _float(batch)]
        params, predictors = [], []
        for k in range(levels):
            coarse = [np.clip(_up(p[k + 1], p[k]), 0, 1) for p in pyramids]
            fine = [p[k] for p in pyramids]
            predictor = Predictor.fit(coarse, fine) if predict else None
            if predictor is not None:
                coarse = [predictor(x) for x in coarse]
            params.append(fit(list(zip(coarse, fine)), gaussianize))
            predictors.append(predictor)
        params.append(fit([(np.full_like(p[levels], GRAY), p[levels]) for p in pyramids], gaussianize))
        label = ("Predictive pyramid" if predict else "Pyramid") + (", normal scores" if gaussianize else "")
        label += {None: "", "final": ", final match", "levels": ", matched levels"}[match]
        tables = [quantile_table([p[k] for p in pyramids]) for k in range(levels + 1)]
        return cls(params, predictors, label, tables, match)

    def sample(self, shape, rng):
        shapes = [shape]
        for _ in range(len(self.models) - 1):
            h, w, c = shapes[-1]
            shapes.append(((h + 1) // 2, (w + 1) // 2, c))
        image = _render_fresh(self.models[-1], np.full(shapes[-1], GRAY, np.float32), rng)
        if self.match == "levels":
            image = match_histogram(image, self.tables[-1])
        for k in range(len(self.models) - 2, -1, -1):
            clean = np.clip(_up(image, np.empty(shapes[k])), 0, 1)
            if self.predictors[k] is not None:
                clean = self.predictors[k](clean)
            image = _render_fresh(self.models[k], clean, rng)
            if self.match == "levels" or (self.match == "final" and k == 0):
                image = match_histogram(image, self.tables[k])
        return image


def to_uint8(image):
    return np.floor(np.clip(image, 0, 1) * 255 + .5).astype(np.uint8)
