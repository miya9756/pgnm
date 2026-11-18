"""Six well-defined random textures, ordered from "PGNM's home ground" to "far from it".

Each generator takes a NumPy Generator and a side length and returns one
uint8 H×W×3 image. Every texture is stationary and defined by a short
procedure, so independent samples are free and the ground truth is known.

| name         | what it probes                                                         |
| ------------ | ---------------------------------------------------------------------- |
| gauss_short  | control: Gaussian field, corr. length ~2 px, colour mix, skewed marginal |
| gauss_long   | Gaussian field, oriented, corr. length ~25 px (beyond the 25×25 kernel) |
| lattice      | repetitive: jittered grid of soft blobs, period 32 px                  |
| cascade      | cross-scale: fine grain whose amplitude follows a coarse field         |
| dead_leaves  | phase structure: occluding disks, sharp edges, power-law radii         |
| strokes      | hand-drawn-like hatching: dark pen strokes on paper                    |
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def _quantize(image):
    return np.floor(np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)


def _unit(field):
    field = field - field.mean(axis=(0, 1))
    return field / field.std(axis=(0, 1))


def _spectral_field(rng, size, amplitude):
    """Periodic Gaussian field with the given (size, size) spectral amplitude."""
    z = rng.standard_normal((size, size))
    return _unit(np.fft.ifft2(np.fft.fft2(z) * amplitude).real)


def gauss_short(rng, size=512):
    fields = _unit(gaussian_filter(rng.standard_normal((size, size, 3)), (2, 2, 0), mode="wrap"))
    mix = np.linalg.cholesky(np.array([[1, .7, .4], [.7, 1, .7], [.4, .7, 1]]))
    latent = fields @ mix.T
    return _quantize(np.array([.42, .38, .33]) + .075 * (np.exp(.8 * latent) - np.exp(.32)))


def gauss_long(rng, size=512):
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    angle = np.deg2rad(30)
    along = fx * np.cos(angle) + fy * np.sin(angle)
    across = -fx * np.sin(angle) + fy * np.cos(angle)
    # Correlation length ~25 px along the grain, ~5 px across it.
    amplitude = np.exp(-0.5 * ((along * 2 * np.pi * 25) ** 2 + (across * 2 * np.pi * 5) ** 2))
    latent = _spectral_field(rng, size, amplitude)
    fine = .015 * rng.standard_normal((size, size, 3))
    return _quantize(np.array([.55, .45, .35]) + latent[..., None] * np.array([.12, .10, .085]) + fine)


def lattice(rng, size=512, period=32):
    image = np.full((size, size, 3), (.20, .22, .26))
    offset = rng.uniform(0, period, 2)
    centers = np.stack(np.meshgrid(np.arange(-1, size // period + 1), np.arange(-1, size // period + 1),
                                   indexing="ij"), -1).reshape(-1, 2) * period + offset
    centers += rng.uniform(-3, 3, centers.shape)
    heights = rng.uniform(.5, 1, len(centers))
    yy, xx = np.mgrid[:size, :size]
    blobs = np.zeros((size, size))
    for (cy, cx), h in zip(centers, heights):
        y0, y1 = int(max(cy - 20, 0)), int(min(cy + 21, size))
        x0, x1 = int(max(cx - 20, 0)), int(min(cx + 21, size))
        if y0 >= y1 or x0 >= x1:
            continue
        r2 = (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2
        blobs[y0:y1, x0:x1] += h * np.exp(-r2 / (2 * 6.0 ** 2))
    image += blobs[..., None] * np.array([.60, .45, .25])
    image += .02 * rng.standard_normal((size, size, 3))
    return _quantize(image)


def cascade(rng, size=512):
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    coarse = _spectral_field(rng, size, np.exp(-0.5 * (np.hypot(fx, fy) * 2 * np.pi * 16) ** 2))
    fine = _unit(gaussian_filter(rng.standard_normal((size, size)), 0.8, mode="wrap"))
    # Grain amplitude grows ~7x from the darkest to the brightest coarse regions.
    value = .50 + .10 * coarse + .045 * np.exp(.8 * coarse) * fine
    return _quantize(value[..., None] * np.array([1.0, .95, .88]))


def dead_leaves(rng, size=512, rmin=3.0, rmax=40.0, count=30000):
    canvas = np.zeros((size, size, 3), np.uint8)
    palette = np.array([[.80, .62, .40], [.45, .55, .30], [.30, .35, .45], [.70, .30, .25]])
    # Inverse-CDF draw from p(r) ∝ r^-3 on [rmin, rmax], the scale-invariant law.
    u = rng.uniform(size=count)
    radii = (rmin ** -2 - u * (rmin ** -2 - rmax ** -2)) ** -.5
    for radius in radii:
        center = rng.uniform(-rmax, size + rmax, 2)
        color = palette[rng.integers(len(palette))] * rng.uniform(.6, 1.15)
        cv2.circle(canvas, (int(center[1] * 16), int(center[0] * 16)), int(radius * 16),
                   tuple(int(c) for c in _quantize(color)), -1, cv2.LINE_AA, shift=4)
    return canvas


def strokes(rng, size=512, count=700):
    paper = .93 + .012 * gaussian_filter(rng.standard_normal((size, size)), 1.2, mode="wrap")
    ink = np.full((size, size), 255, np.uint8)
    for _ in range(count):
        length = rng.uniform(15, 55)
        heading = np.deg2rad(rng.normal(45, 12))
        turn = rng.normal(0, .03)
        point = rng.uniform(-30, size + 30, 2)
        points = []
        for _ in range(int(length / 2)):
            points.append(point.copy())
            heading += turn + rng.normal(0, .02)
            point += 2 * np.array([np.sin(heading), np.cos(heading)])
        points = np.array(points)[:, ::-1] * 16
        cv2.polylines(ink, [points.astype(np.int32)], False, int(rng.uniform(20, 90)),
                      int(rng.choice([1, 1, 2])), cv2.LINE_AA, shift=4)
    darkness = 1 - ink.astype(np.float64) / 255
    value = paper * (1 - darkness)
    return _quantize(value[..., None] * np.array([1.0, .97, .90]) + np.array([0, 0, .02]))


TEXTURES = dict(gauss_short=gauss_short, gauss_long=gauss_long, lattice=lattice,
                cascade=cascade, dead_leaves=dead_leaves, strokes=strokes)


def samples(name, count, seed, size=512):
    """`count` independent samples of one texture; `seed` names the set."""
    rng = np.random.default_rng([seed, list(TEXTURES).index(name)])
    return np.stack([TEXTURES[name](rng, size) for _ in range(count)])
