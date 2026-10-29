"""Self-checks that need no dataset: python -m unittest discover -s tests"""
import json
from pathlib import Path
import unittest

import numpy as np

from experiments import config
from experiments.seeds import bank_seeds, film_pgnm_seed, film_seed, stream_seed, to_uint8
from metrics import kl, residual, statistics

ROOT = Path(__file__).resolve().parents[1]


class Seeds(unittest.TestCase):
    def test_recorded_streams(self):
        # Seeds of the first block / image behind Tab. 1 and Tab. 3.
        self.assertEqual(stream_seed("cvpr-sidd", "S6_800", 0, 0, 0), 783654400)
        self.assertEqual(stream_seed("score", "S6_800", 0, 0, 0), 1765450600)
        self.assertEqual(film_seed("film3-reference", 0.1, "01", 0), 1776266922)
        self.assertEqual(film_pgnm_seed(0.1, "01", 0), 3364950043)
        self.assertEqual(bank_seeds(1), (6104, 6105, 6106, 6107))

    def test_delivery_rounds_half_up(self):
        np.testing.assert_array_equal(to_uint8([-.1, .5 / 255, 1.5 / 255, 1.2]), [0, 1, 2, 255])


class Presets(unittest.TestCase):
    def test_presets_match_the_splits(self):
        from pgnm import load
        sidd, film = config.dataset("sidd"), config.dataset("film")
        self.assertEqual(sorted(p.stem for p in (ROOT / "presets/sidd").glob("*.json")), sorted(sidd["classes"]))
        for name in sidd["classes"]:
            self.assertEqual(np.asarray(load(ROOT / f"presets/sidd/{name}.json")["kernels"]).shape, (25, 25, 3, 3))
        for radius in film["radii"]:
            load(ROOT / "presets/film" / ("r%03d.json" % round(1000 * radius)))
        self.assertEqual(len(sidd["sets"]["S39"]), 39)
        self.assertEqual(len(sidd["sets"]["S19"]), 19)
        self.assertEqual(len(film["test_images"]), 60)

    def test_render_is_reproducible(self):
        from pgnm import PGNM, load
        model = PGNM(load(ROOT / "presets/sidd/S6_800.json"), workers=1)
        clean = np.linspace(0, 1, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
        first = model.render(clean, np.random.default_rng(0))
        model.reset_window_selection()
        second = model.render(clean, np.random.default_rng(0))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(0 <= first.min() <= first.max() <= 1)


class Checkpoints(unittest.TestCase):
    def test_denoisers_load(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from experiments.denoise.noise import SOURCES
        from experiments.denoise.train import DnCNN
        for source in SOURCES:
            DnCNN().load_state_dict(torch.load(ROOT / f"checkpoints/denoise/{source}.pt", map_location="cpu"),
                                    strict=True)


class Metrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.clean = rng.integers(20, 230, (2, 64, 64, 3), dtype=np.uint8)
        self.noisy = np.clip(self.clean + rng.normal(0, 6, self.clean.shape), 0, 255).astype(np.uint8)

    def test_identical_outputs_score_zero(self):
        state = residual.prepare_rgb(self.clean, self.noisy)
        scores = residual.compare_rgb(state, self.noisy)
        self.assertEqual(scores["conditional_w1"], 0)
        self.assertEqual(scores["spatial_correlation_error"], 0)
        self.assertEqual(scores["color_correlation_error"], 0)
        self.assertEqual(kl.kl(self.noisy.astype(int) - self.clean, self.noisy.astype(int) - self.clean), 0)

    def test_stronger_noise_scores_worse(self):
        state = residual.prepare_rgb(self.clean, self.noisy)
        louder = np.clip(self.clean + 3 * (self.noisy.astype(float) - self.clean), 0, 255).astype(np.uint8)
        self.assertGreater(residual.compare_rgb(state, louder)["conditional_w1"], 1)

    def test_gray_identity(self):
        mu = self.clean[0, ..., 0] / 255
        state = residual.prepare_gray(mu, self.noisy[0, ..., 0])
        scores = residual.compare_gray(state, residual.summarize_gray(state, self.noisy[0, ..., 0]))
        self.assertEqual(scores["conditional_w1"], 0)
        self.assertEqual(scores["spatial_correlation_error"], 0)

    def test_source_weights_count_cameras_equally(self):
        labels = [dict(camera="A", iso=100), dict(camera="A", iso=100), dict(camera="A", iso=200),
                  dict(camera="B", iso=100)]
        np.testing.assert_allclose(statistics.source_weights([0, 1, 2, 3], labels), [.125, .125, .25, .5])
        values = np.ones((4, 5))
        self.assertAlmostEqual(statistics.level(values, [1, 1, 1, 1]), 1)
        design = statistics.draws([1, 1, 1, 1], ["s1", "s1", "s2", "s3"], B=200)
        result = statistics.paired_difference(values, values, [1, 1, 1, 1], design)
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["interval"], [0, 0])


class Splits(unittest.TestCase):
    def test_fit_keys_name_existing_crops(self):
        split = config.dataset("sidd")
        for name, group in split["classes"].items():
            camera, iso = name.split("_")
            for key in group["fit_keys"][:20]:
                capture, image, index = key.split("/")
                self.assertEqual(capture.split("_")[2], camera)
                self.assertEqual(int(capture.split("_")[3]), int(iso))
                self.assertIn(image, ("010", "011"))
        json.dumps(split)


if __name__ == "__main__":
    unittest.main()
