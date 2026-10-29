"""Build the PGNM web studio into a static site.

    pip install './pgnm[images]'
    python web/build.py _site            # then: python -m http.server -d _site

Copies the app, converts the shipped presets to float32 binaries
(kernels[25][25][3][3] followed by marginal.g[3][256][25]) and renders each
preset's swatch with the reference implementation in pgnm/.
"""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import correlate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgnm import PGNM, load  # noqa: E402

WEB = ROOT / "web"
STATIC = ["index.html", "style.css", "js/app.js", "js/engine.js", "js/viewer.js",
          "js/pool.js", "js/worker.js", "js/scene.js"]
CAMERAS = [("S6", "Galaxy S6"), ("GP", "Pixel"), ("IP", "iPhone 7"), ("N6", "Nexus 6"), ("G4", "LG G4")]
FILM_RADII = ["r025", "r050", "r075", "r100", "r200", "r400", "r600", "r800"]
SWATCH = 72
KNOTS = np.arange(-3, 3.0001, 0.25)


def camera_presets():
    """SIDD presets (Tab. 1), plus the denoising presets (Tab. 2) for other ISOs."""
    found = {}
    for folder in ("denoise", "sidd"):
        for path in sorted((ROOT / "presets" / folder).glob("*.json")):
            found[path.stem] = path
    rows = []
    for code, label in CAMERAS:
        isos = sorted(int(name.split("_")[1]) for name in found if name.split("_")[0] == code)
        rows.append({"label": label, "presets": [
            {"id": f"{code}_{iso}", "label": str(iso), "title": f"{label} · ISO {iso}",
             "path": found[f"{code}_{iso}"]} for iso in isos]})
    return rows


def film_presets():
    blurs = json.loads((ROOT / "presets/film/blur.json").read_text())
    grain = [{"id": key, "label": f"{int(key[1:]) / 1000:g}", "title": f"Grain {int(key[1:]) / 1000:g}",
              "path": ROOT / f"presets/film/{key}.json", "blur": blurs[key]} for key in FILM_RADII]
    nocturne = [{"id": "nocturne", "label": "Color", "title": "Nocturne",
                 "path": ROOT / "presets/bandwidth/pgnm.json"}]
    return [{"label": "Grain", "presets": grain}, {"label": "Nocturne", "presets": nocturne}]


def swatch(params, blur):
    ramp = np.linspace(0.25, 0.75, SWATCH)
    clean = np.broadcast_to(ramp[None, :, None], (SWATCH, SWATCH, 3)).astype(np.float64)
    if blur is not None:
        weights = np.asarray(blur, np.float64)
        clean = np.stack([np.clip(correlate(255 * clean[..., c], weights, mode="reflect"), 0, 255) / 255
                          for c in range(3)], axis=-1)
    noisy = PGNM(params, workers=1).render(clean.astype(np.float32), np.random.default_rng(2026))
    return np.floor(np.clip(noisy.astype(np.float64), 0, 1) * 255 + 0.5).astype(np.uint8)


def build(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for name in STATIC:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEB / name, out / name)
    (out / "presets").mkdir()
    (out / "swatches").mkdir()
    groups = [{"id": "camera", "label": "Camera", "default": "S6_800", "rows": camera_presets()},
              {"id": "film", "label": "Film", "default": "r100", "rows": film_presets()}]
    for group in groups:
        for row in group["rows"]:
            for preset in row["presets"]:
                params = load(preset.pop("path"))
                if not np.allclose(params["marginal"]["t"], KNOTS):
                    raise ValueError(f"{preset['id']}: unexpected latent knots")
                kernels = np.asarray(params["kernels"], "<f4")
                lookup = np.asarray(params["marginal"]["g"], "<f4")
                np.concatenate([kernels.ravel(), lookup.ravel()]).tofile(out / f"presets/{preset['id']}.bin")
                Image.fromarray(swatch(params, preset.get("blur"))).save(out / f"swatches/{preset['id']}.png")
                print(preset["id"], flush=True)
    (out / "presets.json").write_text(json.dumps({"groups": groups}, indent=1) + "\n")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ROOT / "_site")
