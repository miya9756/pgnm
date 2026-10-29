"""PyTorch port of the PGNM renderer, used only for the GPU rows of Tab. 4.

pgnm.PGNM runs on the CPU. This port evaluates the same equations with
PyTorch so that they can be timed on a GPU:

  field  F = K * z         z white Gaussian, (H+24) x (W+24) x 3, "valid" support
  noisy  y = clip(x + g_c(x, F), 0, 1)

with the 25x25x3x3 filter tensor K and the bilinear residual lookup g of a
preset (3 channels x 256 intensity knots x 25 Gaussian knots, extended
linearly beyond the first and last Gaussian knot). Bank playback keeps four
2H x 2W fields and reads one cyclic H x W window per frame with PGNM's rule:
a bank field other than the previous one, then a row and a column offset.

The arithmetic is float32; the CPU renderer filters and interpolates in
float64. Given the same white noise, the two latent fields differ by about
1e-5; given the same field, the noisy images agree after 8-bit delivery
(python -m experiments.cost port-check). The port draws its white noise
with PyTorch's generator, so its samples are not those of the CPU renderer.
"""
import numpy as np
import torch


class TorchPGNM:
    def __init__(self, preset, device, bank_seeds=(6100, 6101, 6102, 6103), window_seed=2026):
        kernels = np.asarray(preset["kernels"], np.float32)  # (height, width, input, output)
        # scipy's convolution flips the filter and conv2d does not, so flip it here.
        weight = kernels.transpose(3, 2, 0, 1)[:, :, ::-1, ::-1]
        self.weight = torch.from_numpy(np.ascontiguousarray(weight)).to(device)
        self.margin = kernels.shape[0] - 1, kernels.shape[1] - 1
        lookup = np.asarray(preset["marginal"]["g"], np.float32)  # (channel, intensity, gaussian)
        self.channels, self.intensity_knots, self.gaussian_knots = lookup.shape
        self.lookup = torch.from_numpy(lookup.reshape(self.channels, -1)).to(device)
        knots = preset["marginal"]["t"]
        self.knot_start, self.knot_step = float(knots[0]), float(knots[1] - knots[0])
        self.device = device
        self.bank_seeds = tuple(bank_seeds)
        self.bank = []
        self.previous = -1
        self.window_rng = np.random.default_rng(window_seed)

    def filter(self, noise):
        """(1, 3, H+24, W+24) white noise -> (3, H, W) latent field."""
        return torch.nn.functional.conv2d(noise, self.weight)[0]

    def new_field(self, height, width, generator):
        noise = torch.randn((1, 3, height + self.margin[0], width + self.margin[1]),
                            generator=generator, device=self.device)
        return self.filter(noise)

    def warmup(self, height, width):
        """Build the bank: four 2H x 2W fields, one per seed."""
        self.bank = [self.new_field(2 * height, 2 * width, torch.Generator(device=self.device).manual_seed(seed))
                     for seed in self.bank_seeds]

    def window(self, height, width):
        candidates = [index for index in range(4) if index != self.previous]
        index = candidates[int(self.window_rng.integers(len(candidates)))]
        rows = (torch.arange(height, device=self.device)
                + int(self.window_rng.integers(2 * height))) % (2 * height)
        columns = (torch.arange(width, device=self.device)
                   + int(self.window_rng.integers(2 * width))) % (2 * width)
        self.previous = index
        return self.bank[index].index_select(1, rows).index_select(2, columns)

    def add_noise(self, field, clean):
        """field, clean: (3, H, W) float32 on the device -> noisy (3, H, W)."""
        position = clean.clamp(0, 1) * (self.intensity_knots - 1)
        index = position.long().clamp(max=self.intensity_knots - 2)
        weight = position - index
        gaussian = (field - self.knot_start) / self.knot_step
        g_index = gaussian.floor().clamp(0, self.gaussian_knots - 2).long()
        g_weight = gaussian - g_index
        flat = (index * self.gaussian_knots + g_index).reshape(self.channels, -1)

        def table(offset):
            return torch.gather(self.lookup, 1, flat + offset).reshape(clean.shape)

        n = self.gaussian_knots
        lower = table(0) + g_weight * (table(1) - table(0))
        upper = table(n) + g_weight * (table(n + 1) - table(n))
        return (clean + lower + weight * (upper - lower)).clamp(0, 1)

    def render(self, clean):
        """Bank playback for a (3, H, W) clean image; call warmup(H, W) first."""
        return self.add_noise(self.window(clean.shape[1], clean.shape[2]), clean)
