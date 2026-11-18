"""Smoke test of the texture harness on tiny images (seconds, no data)."""
import unittest

import numpy as np

from textures import measures
from textures.generators import TEXTURES, samples
from textures.models import Pyramid, SingleScale, to_uint8


class TextureHarness(unittest.TestCase):
    def test_generators_are_seeded_uint8(self):
        for name in TEXTURES:
            a, b = samples(name, 1, 0, size=64), samples(name, 1, 0, size=64)
            self.assertEqual(a.dtype, np.uint8)
            self.assertEqual(a.shape, (1, 64, 64, 3))
            np.testing.assert_array_equal(a, b)

    def test_models_fit_sample_and_score(self):
        train, reference = samples("gauss_short", 2, 1, size=128), samples("gauss_short", 2, 2, size=128)
        models = [SingleScale.fit(train), SingleScale.fit(train, gaussianize=True),
                  Pyramid.fit(train, levels=2, predict=True, gaussianize=True)]
        for model in models:
            rng = np.random.default_rng(0)
            generated = np.stack([to_uint8(model.sample((128, 128, 3), rng)) for _ in range(2)])
            self.assertEqual(generated.shape, reference.shape)
            scores = dict(**measures.compare(reference, generated), **measures.describe(generated[:1]))
            self.assertTrue(all(np.isfinite(v) for v in scores.values() if v is not None))

    def test_gaussian_texture_has_no_tv_deficit(self):
        rng = np.random.default_rng(0)
        image = np.clip(128 + 30 * rng.standard_normal((128, 128, 3)), 0, 255).astype(np.uint8)
        _, deficit = measures.sharpness_index(image, rng)
        self.assertLess(abs(deficit), 0.01)


if __name__ == "__main__":
    unittest.main()
