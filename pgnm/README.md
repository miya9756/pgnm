# pgnm — the method

A CPU implementation of PGNM (Sec. 3 of the paper), independent of every
dataset and experiment in this repository. It needs NumPy, SciPy and Numba.

```bash
pip install ./pgnm            # or: pip install './pgnm[images]' for PNG I/O
```

```python
import numpy as np
from pgnm import PGNM, fit, load, save

params = fit(pairs)                     # aligned float RGB (clean, noisy) pairs in [0, 1]
save("look.json", params)
model = PGNM(load("presets/sidd/S6_800.json"), workers=2)
noisy = model.render(clean, np.random.default_rng(0))   # float32 RGB in [0, 1]
```

`pairs` may be a list or, for large training sets, a function returning a fresh
iterator: fitting passes over the data several times. Pass `progress=print` to
see sample counts. Deliver 8-bit images by clipping, multiplying by 255 and
rounding half up. A small command line exists too:

```bash
python -m pgnm fit pairs.npz look.json
python -m pgnm render clean.png noisy.png --params look.json --seed 0
```

## The model in two equations

For independent standard Gaussian fields z_i (one per input channel),

```text
t_c(p) = sum_i sum_u A_ci(u) z_i(p - u)                 latent field
y_c(p) = clip(x_c(p) + g_c(x_c(p), t_c(p)), 0, 1)       noisy image
```

`A` is a full 25×25×3×3 convolution tensor whose nine filters create spatial
and color dependence together; each output channel has unit energy, so every
latent value stays standard Gaussian. `g` is a signed-residual lookup: for
channel c, clean intensity x and latent value t it returns the residual quantile
at probability Phi(t). `A` shapes dependence, `g` supplies the distribution and
strength; nothing else is fitted.

| Stored array | Shape | Meaning |
|---|---|---|
| `kernels` | (25, 25, 3, 3) | offset (height, width), input channel, output channel |
| `marginal.t` | (25,) | latent knots -3, -2.75, ..., 3 |
| `marginal.g` | (3, 256, 25) | residual by channel, clean intensity code and latent knot |
| `marginal.step` | 1.0 | residual histogram bin width in 8-bit codes |

Presets are JSON files with schema `pgnm/5`; `load` and `save` validate them.

## Reading `model.py` (inference)

1. `PGNM._build_noise_field` draws padded independent float32 Gaussian RGB and
   applies the nine filters by valid FFT convolution (ordinary convolution).
2. `PGNM.warmup` builds four such fields of size 2H×2W (seeds 6100–6103 unless
   `bank_seeds` says otherwise) once per frame size: the template bank.
3. `PGNM.render` picks a template other than the previous one and a random cyclic
   window offset; these are the only per-frame random draws.
4. `PGNM._lookup_and_add_noise` reads the window, interpolates g bilinearly in
   clean intensity and latent value (extending the outer segments beyond ±3),
   adds the residual and clips, in one compiled loop.

`reset_window_selection()` forgets the previous template so that independent
items do not depend on their order; the paper calls it before every scored
image or block.

## Reading `fitting.py` (fitting)

1. `flat_pixel_contrast` / `flat_pixel_threshold`: a pixel is flat when the
   largest RGB standard deviation in its reflected 7×7 window is at most twice
   the 2nd percentile of that contrast (bounded to [0.003, 0.030]).
2. `fit_pilot_strength`: a provisional strength curve at 17 intensities from
   flat pixels, corrected for clipping. It only normalizes residuals before
   correlations are measured; it is not part of the model.
3. `normalize_residuals`: residual / pilot strength on flat pixels whose clean
   intensity is at least 2.5 pilot sigmas from 0 and 1 in every channel (the
   clipping guard). If no pixel of the whole training set passes the guard,
   the flat pixels alone are used.
4. `measure_cross_correlation`: all nine channel-pair correlations for lags up
   to 16 pixels, never across image boundaries.
5. `factor_cross_channel_kernels`: per frequency, the positive semidefinite
   square root of the correlation spectrum (lags trusted to 8, tapered to 16),
   cropped to 25×25 and normalized: the tensor A.
6. `measure_residual_histogram`, `pool_residual_histograms`,
   `build_residual_lookup`: over all pixels, including texture and clipping,
   per channel and clean intensity code, a histogram of residual codes, pooled
   with neighboring intensities until it has 2,000 samples; inverting its
   cumulative distribution at Phi(t) gives g.
