"""CPU inference: direct RGB convolution, four-field bank and residual lookup.

Setup filters independent Gaussian noise with kernels[y, x, input, output].
Rendering selects a cyclic bank window and evaluates g(clean intensity, field),
then adds the signed residual and clips once. No sensor phase or strength control
is needed: the single conditional lookup supplies the residual distribution.
"""
import numpy as np
from numba import njit, prange
from scipy.signal import fftconvolve

from .parameters import validate


class PGNM:
    name = "ours"
    label = "PGNM"

    def __init__(self, params, workers=2, bank_seeds=(6100, 6101, 6102, 6103)):
        self.params = validate(params)
        if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer")
        bank_seeds = tuple(bank_seeds)
        if len(bank_seeds) != 4 or any(
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0
            for seed in bank_seeds
        ) or len(set(bank_seeds)) != 4:
            raise ValueError("bank_seeds must contain four distinct nonnegative integers")
        self.workers = int(workers)
        self.bank_seeds = tuple(int(seed) for seed in bank_seeds)
        self._convolution_kernels = np.asarray(self.params["kernels"], np.float64)
        self._residual_lookup = np.ascontiguousarray(self.params["marginal"]["g"], dtype=np.float64)
        knots = self.params["marginal"]["t"]
        self._gaussian_knot_start = float(knots[0])
        self._gaussian_knot_step = float(knots[1] - knots[0])
        self._frame_shape = None
        self._noise_bank = []
        self._previous_bank_index = -1
        self._window_rng = np.random.default_rng(2026)

    @staticmethod
    def fit(pairs, **options):
        """Fit aligned clean/noisy training pairs; see fitting.fit for options."""
        from .fitting import fit

        return fit(pairs, **options)

    def _build_noise_field(self, shape, rng):
        """Convolve independent RGB noise into one unit-variance latent field."""
        height, width, channels = shape
        kernel_height, kernel_width = self._convolution_kernels.shape[:2]
        independent_noise = rng.standard_normal(
            (height + kernel_height - 1, width + kernel_width - 1, channels),
            dtype=np.float32,
        )
        # Standard convolution reverses the filter, as scipy.signal defines it.
        # Valid support gives exactly H×W without introducing a padded boundary.
        return np.stack(
            [
                sum(
                    fftconvolve(
                        independent_noise[..., source_channel],
                        self._convolution_kernels[..., source_channel, output_channel],
                        mode="valid",
                    )
                    for source_channel in range(channels)
                )
                for output_channel in range(channels)
            ],
            axis=-1,
        )

    def warmup(self, shape):
        """Build four 2H×2W float32 fields and compile the fused sampler once."""
        shape = tuple(shape)
        if (
            len(shape) != 3 or shape[2] != 3
            or any(isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1 for size in shape)
        ):
            raise ValueError("Expected a positive integer H×W×3 frame shape")
        if self._frame_shape == shape:
            return
        height, width, channels = shape
        self._noise_bank = []
        self._noise_bank = [
            self._build_noise_field(
                (2 * height, 2 * width, channels), np.random.default_rng(seed)
            ).astype(np.float32)
            for seed in self.bank_seeds
        ]
        self._frame_shape = shape
        self.reset_window_selection()
        # Compile without consuming any window-selection draws.
        tiny_clean = np.zeros((1, 1, 3), np.float32)
        self._render_noise_window(self._noise_bank[0], tiny_clean, 0, 0, np.empty_like(tiny_clean))

    def reset_window_selection(self):
        """Allow every bank field next time without changing the bank or RNG."""
        self._previous_bank_index = -1

    def render(self, clean, rng=None):
        """Select a bank window, map its values to signed residuals, add and clip."""
        clean = np.asarray(clean, dtype=np.float32)
        if clean.ndim != 3 or clean.shape[2] != 3 or not np.isfinite(clean).all():
            raise ValueError("Input must be finite H×W×3 RGB")
        self.warmup(clean.shape)
        rng = self._window_rng if rng is None else rng
        height, width, _ = clean.shape
        candidates = [index for index in range(4) if index != self._previous_bank_index]
        bank_index = candidates[int(rng.integers(len(candidates)))]
        row_offset = int(rng.integers(2 * height))
        column_offset = int(rng.integers(2 * width))
        noisy = np.empty_like(clean)
        self._render_noise_window(
            self._noise_bank[bank_index], clean, row_offset, column_offset, noisy
        )
        self._previous_bank_index = bank_index
        return noisy

    def _render_noise_window(self, noise_field, clean, row_offset, column_offset, noisy):
        from numba import get_num_threads, set_num_threads

        previous_threads = get_num_threads()
        try:
            set_num_threads(self.workers)
            self._lookup_and_add_noise(
                noise_field, clean, self._residual_lookup,
                self._gaussian_knot_start, self._gaussian_knot_step,
                row_offset, column_offset, noisy,
            )
        finally:
            set_num_threads(previous_threads)

    @staticmethod
    @njit(cache=True, parallel=True, nogil=True)
    def _lookup_and_add_noise(
        noise_field, clean, residual_lookup, gaussian_knot_start, gaussian_knot_step,
        row_offset, column_offset, noisy,
    ):
        """Fuse cyclic window reading, bilinear lookup, addition and clipping."""
        height, width, channels = clean.shape
        field_height, field_width, _ = noise_field.shape
        num_intensity_knots = residual_lookup.shape[1]
        num_gaussian_knots = residual_lookup.shape[2]

        def interpolate_residual(clean_intensity, gaussian_value, channel):
            intensity_position = min(max(clean_intensity, 0.0), 1.0) * (num_intensity_knots - 1)
            intensity_index = min(int(intensity_position), num_intensity_knots - 2)
            intensity_weight = intensity_position - intensity_index
            # Extend the first/last line segment beyond the Gaussian knot range.
            gaussian_position = (gaussian_value - gaussian_knot_start) / gaussian_knot_step
            gaussian_index = min(max(int(np.floor(gaussian_position)), 0), num_gaussian_knots - 2)
            gaussian_weight = gaussian_position - gaussian_index
            lookup = residual_lookup[channel]
            lower = lookup[intensity_index, gaussian_index] + gaussian_weight * (
                lookup[intensity_index, gaussian_index + 1] - lookup[intensity_index, gaussian_index]
            )
            upper = lookup[intensity_index + 1, gaussian_index] + gaussian_weight * (
                lookup[intensity_index + 1, gaussian_index + 1] - lookup[intensity_index + 1, gaussian_index]
            )
            return lower + intensity_weight * (upper - lower)

        for row in prange(height):
            field_row = row + row_offset
            if field_row >= field_height:
                field_row -= field_height
            for column in range(width):
                field_column = column + column_offset
                if field_column >= field_width:
                    field_column -= field_width
                for channel in range(channels):
                    clean_intensity = float(clean[row, column, channel])
                    gaussian_value = float(noise_field[field_row, field_column, channel])
                    residual = interpolate_residual(clean_intensity, gaussian_value, channel)
                    noisy[row, column, channel] = min(max(clean_intensity + residual, 0.0), 1.0)

    @property
    def bank_bytes(self):
        return sum(field.nbytes for field in self._noise_bank)
