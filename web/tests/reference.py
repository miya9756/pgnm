"""Reference outputs of pgnm/ for the web engine tests: python web/tests/reference.py OUT_DIR

For a camera and a film preset, writes the padded Gaussian noise PGNM draws,
the latent field it builds from that noise (float32) and the delivered 8-bit
image rendered from that field at window offset 0, all from one 8-bit clean
image (`clean`, RGB bytes). Film presets render the expected image
mu = clip(B * x, 0, 255) / 255 of their radius (experiments.film.expected_image).
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import correlate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pgnm import PGNM, load  # noqa: E402

HEIGHT, WIDTH, PAD = 261, 487, 24
PRESETS = {"camera": "presets/sidd/S6_800.json", "film": "presets/film/r100.json"}


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 256, (HEIGHT, WIDTH, 3)).astype(np.uint8)
    codes[:, : WIDTH // 2] = np.linspace(0, 255, WIDTH // 2).round().astype(np.uint8)[None, :, None]
    codes.tofile(out / "clean")
    blur = np.asarray(json.loads((ROOT / "presets/film/blur.json").read_text())["r100"])
    cleans = {
        "camera": codes.astype(np.float32) / 255,
        "film": np.stack([np.clip(correlate(codes[..., c].astype(np.float64), blur, mode="reflect"), 0, 255) / 255
                          for c in range(3)], axis=-1).astype(np.float32),
    }
    for key, path in PRESETS.items():
        model = PGNM(load(ROOT / path), workers=1)
        noise = np.random.default_rng(11).standard_normal((HEIGHT + PAD, WIDTH + PAD, 3), dtype=np.float32)
        field = model._build_noise_field((HEIGHT, WIDTH, 3), np.random.default_rng(11)).astype(np.float32)
        noisy = np.empty_like(cleans[key])
        model._render_noise_window(field, cleans[key], 0, 0, noisy)
        delivered = np.floor(np.clip(noisy.astype(np.float64), 0, 1) * 255 + 0.5).astype(np.uint8)
        noise.astype("<f4").tofile(out / f"noise_{key}")
        field.astype("<f4").tofile(out / f"field_{key}")
        delivered.tofile(out / f"noisy_{key}")
    (out / "meta.json").write_text(json.dumps({"height": HEIGHT, "width": WIDTH, "presets": PRESETS,
                                               "blur": {"film": "r100"}}))


if __name__ == "__main__":
    main(sys.argv[1])
