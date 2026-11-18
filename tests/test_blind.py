"""Smoke test of single-image estimation (seconds, no data, no torch)."""
import unittest

import numpy as np

from blind import estimate
from blind.denoisers import dncnn, load_torch_state, nlm
from pgnm import load
from pgnm.parameters import validate


class SingleImageEstimation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        y, x = np.mgrid[:128, :128] / 127
        cls.clean = estimate.to_uint8(np.stack([x, y, (x + y) / 2], -1) * .8 + .1)
        preset = load("presets/sidd/N6_800.json")
        cls.noisy = estimate.to_uint8(estimate.render_fresh(preset, estimate.to_float(cls.clean), rng))

    def test_checkpoint_reads_without_torch(self):
        state = load_torch_state("checkpoints/denoise/real.pt")
        self.assertEqual(state["dncnn.0.weight"].shape, (64, 3, 3, 3))
        out = dncnn("real")(self.noisy[:32, :32])
        self.assertEqual(out.shape, (32, 32, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_estimators_return_valid_presets(self):
        denoise = nlm(20)
        for name, estimator in estimate.ESTIMATORS.items():
            options = dict(iterations=1, simulations=1) if name == "calibrated" else {}
            validate(estimator(self.noisy, denoise, **options))
        validate(estimate.oracle(self.noisy, self.clean))

    def test_rescale_lookup_scales_spread(self):
        lookup = np.tile(np.linspace(-.1, .1, 25), (3, 256, 1))
        scaled = estimate.rescale_lookup(lookup, np.full((3, 16), 2.0))
        np.testing.assert_allclose(scaled, 2 * lookup, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
