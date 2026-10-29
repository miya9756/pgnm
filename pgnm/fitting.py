"""Fit PGNM's full RGB convolution and all-pixel conditional residual lookup.

Read ``fit()`` first, then follow flatness, pilot strength, correlation and
quantile estimation below. The pilot only normalizes correlation samples;
rendering obtains residual amplitude directly from the quantile lookup.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.special import ndtr

from .parameters import SCHEMA, validate

PILOT_KNOTS = 17
PILOT_SUBBINS = 32
FLAT_WINDOW = 7
AUTOCORRELATION_RADIUS = 16
CLIPPING_GUARD = 2.5
CORRELATION_MASK_POLICY = "flat-with-empty-clipping-fallback/1"
GAUSSIAN_KNOTS = np.round(np.arange(-3.0, 3.0001, 0.25), 2)
MIN_MARGINAL_SAMPLES = 2000
MIN_POOLING_RADIUS = 2
_NORMAL_PDF_SCALE = np.sqrt(2.0 * np.pi)


def fit(pairs, threshold=None, *, kernel_size=25, progress=None):
    """Fit one ungated convolution tensor and one conditional quantile lookup.

    ``pairs`` is a repeatable sequence of aligned float RGB (clean, noisy)
    images in [0, 1], or a callable returning a fresh iterator for each pass.
    Whole images and FFT fields are processed one pair at a time. Automatic
    threshold selection retains only every fourth row/column of clean contrast
    to compute the original exact percentile; pass ``threshold`` to omit it.
    If the combined flat/clipping mask is empty across the entire fitting set,
    correlation is measured in one additional pass using only the flat mask.
    ``progress``, when supplied, receives human-readable sample-count messages.
    """
    if progress is not None and not callable(progress):
        raise TypeError("progress must be a callable receiving a message")
    source = _pair_source(pairs)
    kernel_size = _validate_kernel_size(kernel_size)
    threshold = flat_pixel_threshold(source) if threshold is None else float(threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("flatness threshold must be finite and nonnegative")
    pilot_strength = fit_pilot_strength(source, threshold)
    sample_counts = dict(total_pixels=0, flat_pixels=0, guarded_pixels=0)

    def correlation_samples(flat_only=False):
        for clean, noisy in source():
            flat_mask = flat_pixel_contrast(clean) <= threshold
            normalized, guarded_mask = normalize_residuals(
                clean, noisy, flat_mask, pilot_strength
            )
            if not flat_only:
                sample_counts["total_pixels"] += int(flat_mask.size)
                sample_counts["flat_pixels"] += int(flat_mask.sum())
                sample_counts["guarded_pixels"] += int(guarded_mask.sum())
            # Flat-only means exactly this mask, including clean endpoints.
            # A zero-sigma guard would still exclude clean values at 0 or 1.
            yield normalized, flat_mask if flat_only else guarded_mask
        if not flat_only and progress is not None:
            progress(
                f"Fitting samples: {sample_counts['total_pixels']:,} total pixels -> "
                f"{sample_counts['flat_pixels']:,} flat pixels -> "
                f"{sample_counts['guarded_pixels']:,} flat + clipping-guard pixels."
            )

    clipping_guard_fallback = False
    try:
        correlation, pair_counts = measure_cross_correlation(correlation_samples())
    except _EmptyCorrelationSamplesError:
        if sample_counts["guarded_pixels"] != 0:
            raise
        clipping_guard_fallback = True
        if progress is not None:
            progress(
                "Clipping guard selected zero pixels across the fitting set; "
                "using flat pixels only."
            )
        correlation, pair_counts = measure_cross_correlation(correlation_samples(flat_only=True))
    correlation_pixels = int(round(pair_counts[AUTOCORRELATION_RADIUS,
                                              AUTOCORRELATION_RADIUS]))
    correlation_mask = "flat-only" if clipping_guard_fallback else "flat-and-clipping"
    if progress is not None:
        mask_name = "flat-only" if clipping_guard_fallback else "flat + clipping guard"
        progress(f"Correlation fitting: {correlation_pixels:,} pixels with the {mask_name} mask.")
    kernels, factor_details = factor_cross_channel_kernels(correlation, kernel_size)
    histogram = measure_residual_histogram(source)
    return validate(
        dict(
            schema=SCHEMA,
            method="ours",
            name="PGNM",
            kernels=kernels.tolist(),
            marginal=dict(
                t=GAUSSIAN_KNOTS.tolist(),
                g=build_residual_lookup(histogram).tolist(),
                step=1.0,
            ),
            provenance=dict(
                status="fitted",
                recipe="full-rgb-convolution-all-pixel-quantiles",
                flat_threshold=threshold,
                pair_count=source.pair_count,
                pixel_count=int(histogram[0].sum()),
                flat_pixels=sample_counts["flat_pixels"],
                guarded_pixels=sample_counts["guarded_pixels"],
                correlation_pixels=correlation_pixels,
                correlation_mask=correlation_mask,
                clipping_guard_fallback=clipping_guard_fallback,
                correlation_mask_policy=CORRELATION_MASK_POLICY,
                correlation_radius=AUTOCORRELATION_RADIUS,
                clipping_guard_sigma=CLIPPING_GUARD,
                kernel_size=kernel_size,
                factorization=factor_details,
            ),
        )
    )


def _pair_source(pairs):
    return pairs if isinstance(pairs, _PairSource) else _PairSource(pairs)


class _PairSource:
    """Reopen validated pairs without retaining the image collection."""

    def __init__(self, pairs):
        if callable(pairs):
            self._open = pairs
        else:
            if iter(pairs) is pairs:
                raise TypeError(
                    "pairs must be repeatable; use a callable returning a fresh iterator"
                )
            self._open = lambda: iter(pairs)
        self.pair_count = None

    def __call__(self):
        count = 0
        for pair in self._open():
            if len(pair) != 2:
                raise ValueError("each fitting pair must contain clean and noisy RGB images")
            clean, noisy = (np.asarray(image, np.float64) for image in pair)
            if (clean.ndim != 3 or clean.shape[-1] != 3 or clean.shape != noisy.shape
                    or min(clean.shape[:2]) == 0):
                raise ValueError("fitting requires paired RGB images with matching nonempty shapes")
            if (not np.isfinite(clean).all() or not np.isfinite(noisy).all()
                    or clean.min() < 0 or clean.max() > 1
                    or noisy.min() < 0 or noisy.max() > 1):
                raise ValueError("fitting images must contain finite RGB values in [0, 1]")
            count += 1
            yield clean, noisy
        if count == 0:
            raise ValueError("fitting requires at least one clean/noisy pair")
        if self.pair_count is not None and count != self.pair_count:
            raise ValueError("the pair source must return the same pairs on every pass")
        self.pair_count = count


# One clean-image flatness calculation for the pilot and all correlations.


def flat_pixel_contrast(clean, win=FLAT_WINDOW):
    """Float64 reflected local standard deviation, maximum over RGB."""
    image = np.asarray(clean, np.float64)
    mean = uniform_filter(image, size=(win, win, 1), mode="reflect")
    second_moment = uniform_filter(image * image, size=(win, win, 1), mode="reflect")
    return np.sqrt(np.maximum(second_moment - mean * mean, 0.0)).max(axis=2)


def flat_pixel_threshold(pairs, scale=2.0, percentile=2.0, minimum=0.003, maximum=0.030):
    """Twice the sampled clean contrast floor, with the recorded bounds."""
    contrast = np.concatenate([
        flat_pixel_contrast(clean)[::4, ::4].ravel()
        for clean, _ in _pair_source(pairs)()
    ])
    return float(np.clip(scale * np.percentile(contrast, percentile), minimum, maximum))


# Pilot second moments account for clipping before correlation normalization.


def fit_pilot_strength(pairs, threshold):
    """Fit 17 intensity knots on the same inclusive float64 flatness mask."""
    accumulator = PilotStrengthAccumulator()
    for clean, noisy in _pair_source(pairs)():
        flat_mask = flat_pixel_contrast(clean) <= threshold
        accumulator.add(clean, noisy - clean, flat_mask)
    if accumulator.sample_count.sum() == 0:
        raise ValueError("no flat pixels are available for pilot strength fitting")
    return accumulator.fit(min_pixels=500, max_censored=0.35)


def interpolate_strength(curve, intensity):
    """Interpolate a pilot curve at clean intensity."""
    return np.interp(np.clip(intensity, 0, 1), np.linspace(0, 1, len(curve)), curve)


def normalize_residuals(clean, noisy, flat_mask, strength_curves, guard=CLIPPING_GUARD):
    """Return residual/sigma and one shared flat-and-unclipped RGB mask.

    The guard depends only on clean intensity and the fitted pilot. Observed
    residuals, including whether the noisy image clipped, never select samples.
    A pixel is usable only when every RGB channel passes the guard.
    """
    clean = np.asarray(clean, np.float64)
    strength = np.stack([
        np.maximum(interpolate_strength(strength_curves[channel], clean[..., channel]), 1e-6)
        for channel in range(3)
    ], axis=2)
    unclipped = np.all((clean > guard * strength) & (clean < 1 - guard * strength), axis=2)
    normalized = (np.asarray(noisy, np.float64) - clean) / strength
    return normalized, np.asarray(flat_mask, bool) & unclipped


# Estimate all channel-pair lags together, including zero-lag colour correlation.


class _EmptyCorrelationSamplesError(ValueError):
    """No image supplies enough selected pixels to measure correlation."""


def measure_cross_correlation(samples, radius=AUTOCORRELATION_RADIUS):
    """Return C[a,b,dy,dx] = corr(f_a(x), f_b(x + lag)) and exact pair counts.

    Each input is (normalized RGB residual, shared 2D mask). Center each image
    on its usable pixels. Zero padding prevents pairs from wrapping across
    image edges; separate images never form lag pairs.
    """
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("correlation radius must be a nonnegative integer")
    side = 2 * radius + 1
    products = np.zeros((3, 3, side, side))
    pair_counts = np.zeros((side, side))
    lags = np.arange(-radius, radius + 1)
    for field, mask in samples:
        field, mask = np.asarray(field, np.float64), np.asarray(mask, bool)
        if field.ndim != 3 or field.shape[-1] != 3 or mask.shape != field.shape[:2]:
            raise ValueError("correlation samples require RGB fields and a shared 2D mask")
        usable = int(mask.sum())
        if usable < 2:
            continue
        height, width = mask.shape
        fft_shape = (max(2 * height, side), max(2 * width, side))
        centered = (field - field[mask].mean(axis=0)) * mask[..., None]
        field_fft = np.fft.rfft2(centered, s=fft_shape, axes=(0, 1))
        mask_fft = np.fft.rfft2(mask.astype(np.float64), s=fft_shape)
        counts = np.fft.irfft2(np.abs(mask_fft) ** 2, s=fft_shape)
        indices = np.ix_(lags % fft_shape[0], lags % fft_shape[1])
        pair_counts += np.maximum(np.rint(counts[indices]), 0)
        for first in range(3):
            for second in range(3):
                lag_products = np.fft.irfft2(
                    np.conj(field_fft[..., first]) * field_fft[..., second], s=fft_shape
                )
                products[first, second] += lag_products[indices]
    if pair_counts[radius, radius] < 2:
        raise _EmptyCorrelationSamplesError(
            "no usable selected pixels are available for correlation fitting"
        )
    covariance = products / np.maximum(pair_counts, 1)
    variance = np.diag(covariance[:, :, radius, radius])
    if np.any(variance <= 1e-12) or not np.isfinite(variance).all():
        raise ValueError(
            "correlation fitting requires nonzero residual variance in every RGB channel"
        )
    correlation = covariance / np.sqrt(np.outer(variance, variance))[:, :, None, None]
    return correlation, pair_counts


def _validate_kernel_size(size):
    if (not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_))
            or size < 1 or size % 2 == 0):
        raise ValueError("kernel_size must be a positive odd integer")
    return int(size)


def factor_cross_channel_kernels(correlation, kernel_size=25):
    """Direct spectral square root -> (height, width, input, output) filters.

    Trust lags through radius 8, then taper to zero at radius 16. Project each
    frequency's Hermitian spectrum onto the PSD cone, take its square root,
    crop to the requested support, and normalize each output's filter energy.
    No parent filters, rotation toward a previous model, or acceptance gate
    participates in this estimate.
    """
    kernel_size = _validate_kernel_size(kernel_size)
    correlation = np.asarray(correlation, np.float64)
    if (correlation.ndim != 4 or correlation.shape[:2] != (3, 3)
            or correlation.shape[2] != correlation.shape[3] or correlation.shape[2] % 2 == 0
            or not np.isfinite(correlation).all()):
        raise ValueError("correlation must be finite with shape (3, 3, odd, odd)")
    radius = correlation.shape[-1] // 2
    symmetric = 0.5 * (correlation + correlation.transpose(1, 0, 2, 3)[..., ::-1, ::-1])
    rows, columns = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    distance = np.hypot(rows, columns)
    taper = np.cos(0.5 * np.pi * np.clip((distance - 8) / 8, 0, 1)) ** 2
    fft_size = max(64, 1 << max(kernel_size, 2 * radius + 1).bit_length())
    lag_indices = np.arange(-radius, radius + 1) % fft_size
    covariance_grid = np.zeros((fft_size, fft_size, 3, 3))
    covariance_grid[lag_indices[:, None], lag_indices[None, :]] = np.moveaxis(
        symmetric * taper, (0, 1), (2, 3)
    )
    spectrum = np.fft.fft2(covariance_grid, axes=(0, 1))
    spectrum = 0.5 * (spectrum + np.conj(spectrum.swapaxes(-1, -2)))
    eigenvalues, eigenvectors = np.linalg.eigh(spectrum)
    root = (
        eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[..., None, :]
    ) @ np.conj(eigenvectors.swapaxes(-1, -2))
    # C[a,b,lag] uses conj(F_a)*F_b. Its conjugate root is the output/input
    # transfer matrix for ordinary convolution; reversing this flips cross lags.
    full_filters = np.fft.ifft2(np.conj(root), axes=(0, 1)).real
    half_support = kernel_size // 2
    indices = np.arange(-half_support, half_support + 1) % fft_size
    kernels = full_filters[indices[:, None], indices[None, :]].swapaxes(-1, -2)
    retained_energy = np.sum(kernels * kernels, axis=(0, 1, 2))
    if np.any(retained_energy <= 1e-12):
        raise ValueError("the correlation estimate has an empty output filter")
    kernels /= np.sqrt(retained_energy)[None, None, None, :]
    total_energy = np.sum(full_filters * full_filters, axis=(0, 1, 3))
    return kernels, dict(
        spectral_grid=fft_size,
        trust_radius=8,
        taper_radius=16,
        retained_energy_fraction=(retained_energy / np.maximum(total_energy, 1e-12)).tolist(),
        negative_spectral_fraction=float(
            np.maximum(-eigenvalues, 0).sum() / max(np.abs(eigenvalues).sum(), 1e-12)
        ),
    )


# Clipping-aware pilot moment accumulator.


class PilotStrengthAccumulator:
    """Second moments and clean-intensity subbins at 17 centered knots.

    Subbins preserve the observed clean-intensity distribution inside each
    bucket for the clipping correction; using bucket centers would bias the
    inferred strength near black and white.
    """

    def __init__(self):
        self.residual_second_moment = np.zeros((3, PILOT_KNOTS))
        self.sample_count = np.zeros((3, PILOT_KNOTS))
        self.intensity_subbin_count = np.zeros((3, PILOT_KNOTS, PILOT_SUBBINS))
        self.intensity_subbin_sum = np.zeros((3, PILOT_KNOTS, PILOT_SUBBINS))

    def add(self, clean, residual, mask):
        edges = (np.arange(PILOT_KNOTS + 1) - 0.5) / (PILOT_KNOTS - 1)
        for channel in range(3):
            intensity = clean[..., channel][mask]
            values = residual[..., channel][mask]
            bucket = np.clip(np.digitize(intensity, edges[1:-1]), 0, PILOT_KNOTS - 1)
            self.residual_second_moment[channel] += np.bincount(
                bucket, weights=values * values, minlength=PILOT_KNOTS
            )
            self.sample_count[channel] += np.bincount(bucket, minlength=PILOT_KNOTS)
            fraction = np.clip(
                (intensity - edges[bucket]) / (edges[bucket + 1] - edges[bucket]),
                0, 1 - 1e-9,
            )
            subbin = bucket * PILOT_SUBBINS + (fraction * PILOT_SUBBINS).astype(int)
            shape = (PILOT_KNOTS, PILOT_SUBBINS)
            self.intensity_subbin_count[channel] += np.bincount(
                subbin, minlength=PILOT_KNOTS * PILOT_SUBBINS
            ).reshape(shape)
            self.intensity_subbin_sum[channel] += np.bincount(
                subbin, weights=intensity, minlength=PILOT_KNOTS * PILOT_SUBBINS
            ).reshape(shape)

    def fit(self, min_pixels, max_censored):
        """Invert clipped moments, fill unsupported knots, then smooth once.

        Reject solutions at the solver ceiling and buckets whose predicted
        censoring exceeds 35%; these require extrapolation from adjacent knots.
        At least two supported intensity knots are needed per channel.
        """
        curve = np.full((3, PILOT_KNOTS), np.nan)
        knots = np.linspace(0, 1, PILOT_KNOTS)
        for channel in range(3):
            for bucket in range(PILOT_KNOTS):
                if self.sample_count[channel, bucket] < min_pixels:
                    continue
                second_moment = (
                    self.residual_second_moment[channel, bucket]
                    / self.sample_count[channel, bucket]
                )
                counts = self.intensity_subbin_count[channel, bucket]
                occupied = counts > 0
                centers = self.intensity_subbin_sum[channel, bucket][occupied] / counts[occupied]
                repetitions = np.maximum(
                    np.rint(counts[occupied] / counts.sum() * 512), 1
                ).astype(int)
                intensity_sample = np.repeat(centers, repetitions)
                strength = strength_from_clipped_moment(second_moment, intensity_sample)
                censored = np.mean(
                    normal_cdf(-intensity_sample / max(strength, 1e-9))
                    + normal_cdf(-(1 - intensity_sample) / max(strength, 1e-9))
                )
                if np.isfinite(strength) and strength < 1.999 and censored <= max_censored:
                    curve[channel, bucket] = strength
            supported = np.isfinite(curve[channel])
            if supported.sum() < 2:
                raise ValueError(
                    "pilot intensity coverage requires two supported knots per RGB channel"
                )
            curve[channel] = smooth_strength_curve(
                np.interp(knots, knots[supported], curve[channel, supported])
            )
        return curve


# All samples contribute to the conditional empirical marginal.


def measure_residual_histogram(pairs):
    """All-pixel counts (RGB, clean code, residual code), with no masks."""
    histogram = np.zeros((3, 256, 511))
    for clean, noisy in _pair_source(pairs)():
        intensity_bins = np.clip(np.rint(clean * 255), 0, 255).astype(np.int64)
        residual_bins = np.clip(np.rint((noisy - clean) * 255), -255, 255).astype(np.int64) + 255
        for channel in range(3):
            indices = intensity_bins[..., channel] * 511 + residual_bins[..., channel]
            histogram[channel] += np.bincount(
                indices.ravel(), minlength=256 * 511
            ).reshape(256, 511)
    return histogram


def pool_residual_histograms(
    histogram, min_samples=MIN_MARGINAL_SAMPLES, min_radius=MIN_POOLING_RADIUS
):
    """Pool neighboring intensity bins into residual probabilities.

    Expand a window from +/- min_radius until it contains min_samples, or
    all intensity bins have been included. Pool separately for each channel.
    """
    if histogram.sum(axis=(1, 2)).min() <= 0:
        raise ValueError("every channel needs residual samples")
    num_residual_bins = histogram.shape[2]
    cumulative_histogram = np.concatenate(
        [np.zeros((3, 1, num_residual_bins)), np.cumsum(histogram, 1)], 1
    )
    cumulative_count = cumulative_histogram.sum(-1)
    probabilities = np.empty((3, 256, num_residual_bins))
    for channel in range(3):
        for intensity_bin in range(256):
            radius = min_radius
            while True:
                lower, upper = max(0, intensity_bin - radius), min(
                    256, intensity_bin + radius + 1
                )
                if cumulative_count[channel, upper] - cumulative_count[
                    channel, lower
                ] >= min_samples or (lower == 0 and upper == 256):
                    break
                radius += 1
            counts = (
                cumulative_histogram[channel, upper]
                - cumulative_histogram[channel, lower]
            )
            probabilities[channel, intensity_bin] = counts / counts.sum()
    return probabilities


def build_residual_lookup(histogram):
    """Build g_c(X,t) = F_inverse(residual | X,c)(Phi(t)) on the fixed grid.

    The 25 latent Gaussian knots span [-3, 3]. Residual histogram bins have
    width one code; interpolation inside a bin preserves the established
    empirical quantile estimator. Values are signed normalized residuals.
    """
    if np.shape(histogram) != (3, 256, 511):
        raise ValueError("residual histogram must have shape (3, 256, 511)")
    probabilities = pool_residual_histograms(histogram)
    quantiles = ndtr(GAUSSIAN_KNOTS)
    lookup = np.empty((3, 256, len(GAUSSIAN_KNOTS)))
    for channel in range(3):
        for intensity_bin in range(256):
            probability = probabilities[channel, intensity_bin]
            cdf = np.cumsum(probability)
            residual_index = np.minimum(np.searchsorted(cdf, quantiles, side="left"), 510)
            cdf_below = np.where(residual_index > 0, cdf[np.maximum(residual_index - 1, 0)], 0.0)
            residual_code = (residual_index - 255 - 0.5) + (
                quantiles - cdf_below
            ) / np.maximum(probability[residual_index], 1e-300)
            lookup[channel, intensity_bin] = np.maximum.accumulate(residual_code) / 255
    return np.clip(lookup, -1, 1)


def normal_pdf(x):
    """Standard normal pdf."""
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / _NORMAL_PDF_SCALE


def normal_cdf(x):
    """Standard normal CDF."""
    return ndtr(np.asarray(x, float))


def clipped_second_moment(sigma, intensity):
    """E[(clip(I+n,0,1) - I)^2] for n ~ N(0, sigma^2), elementwise over I.

    The observed residual is a two-sided censored normal: it lands on -I with
    probability Phi(-I/sigma), on 1-I with probability Phi(-(1-I)/sigma), and is
    the truncated normal in between.  Using
    ``int_a^b t^2 phi(t) dt = [Phi(t) - t phi(t)]_a^b`` the whole thing is
    closed form, and it is strictly increasing in sigma -- which is what lets
    ``strength_from_clipped_moment`` bisect on it.
    """
    sigma = max(float(sigma), 1e-12)
    intensity_values = np.asarray(intensity, float)
    a = -intensity_values / sigma  # lower censoring point, standardised
    b = (1.0 - intensity_values) / sigma  # upper censoring point
    Pa, Pb, pa, pb = normal_cdf(a), normal_cdf(b), normal_pdf(a), normal_pdf(b)
    trunc = (Pb - Pa) + a * pa - b * pb  # int_a^b t^2 phi dt
    return (
        intensity_values**2 * Pa
        + (1.0 - intensity_values) ** 2 * (1.0 - Pb)
        + sigma**2 * trunc
    )


def strength_from_clipped_moment(mean_r2, intensity, lo=1e-6, hi=2.0, iters=60):
    """Invert ``clipped_second_moment`` for sigma, given mean(residual^2).

    ``intensity`` is the sample of clean values the bucket actually contained
    (subsample it; only its distribution matters), so bins straddling a
    censoring boundary are handled without an arbitrary exclusion rule.  This
    correction uses the intensity distribution of the selected flat pixels.

    Returns ``hi`` if even that cannot reach the measured moment, which means
    the residual is not this model's noise (structure leakage, misregistration)
    -- the caller should treat a pegged value as a failed bucket.
    """
    intensity_values = np.asarray(intensity, float)
    if intensity_values.size == 0 or not np.isfinite(mean_r2) or mean_r2 <= 0:
        return float("nan")
    if clipped_second_moment(hi, intensity_values).mean() < mean_r2:
        return float(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if clipped_second_moment(mid, intensity_values).mean() < mean_r2:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def smooth_strength_curve(curve, passes=1):
    """Light [1,2,1] smoothing across knots, with endpoints held."""
    y = np.array(curve, float)
    for _ in range(passes):
        z = y.copy()
        z[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
        y = z
    return np.maximum(y, 0.0)
