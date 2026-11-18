"""Fit, sample and score every texture: `python -m textures.run [--textures a b ...]`.

Per texture: 8 training samples (seed 1) fit each model; 8 reference samples
(seed 2) are what the models are scored against; 8 more independent samples
(seed 3, "oracle") show each distance's own sampling noise. A model is as good
as it can be on a statistic when its distance reaches the oracle's.
Writes runs/textures/{results.json, results.md, comparison.jpg}.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import measures
from .generators import TEXTURES, samples
from .models import Pyramid, SingleScale, to_uint8

MODELS = dict(
    pgnm=lambda batch: SingleScale.fit(batch),
    pgnm_ns=lambda batch: SingleScale.fit(batch, gaussianize=True),
    pyramid=lambda batch: Pyramid.fit(batch),
    predictive=lambda batch: Pyramid.fit(batch, predict=True, gaussianize=True),
    matched=lambda batch: Pyramid.fit(batch, predict=True, gaussianize=True, match="final"),
)
COUNT, SIZE, CROP = 8, 512, 192


def run(names, out):
    out.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for name in names:
        start = time.time()
        train, reference, oracle = (samples(name, COUNT, seed, SIZE) for seed in (1, 2, 3))
        batches = dict(reference=reference, oracle=oracle)
        labels = dict(reference="Reference", oracle="Oracle (new reference samples)")
        for key, make in MODELS.items():
            model = make(train)
            rng = np.random.default_rng([7, list(TEXTURES).index(name), list(MODELS).index(key)])
            batches[key] = np.stack([to_uint8(model.sample((SIZE, SIZE, 3), rng)) for _ in range(COUNT)])
            labels[key] = model.label
        results[name] = {key: dict(label=labels[key], **measures.describe(batch[:4]),
                                   **({} if key == "reference" else measures.compare(reference, batch)))
                         for key, batch in batches.items()}
        rows.append(np.concatenate([batches[key][0, :CROP, :CROP]
                                    for key in ("reference", *MODELS)], 1))
        print(f"{name}: {time.time() - start:.0f} s", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results.md").write_text(table(results))
    Image.fromarray(np.concatenate(rows, 0)).save(out / "comparison.jpg", quality=92)
    print(table(results))


def _fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def table(results):
    lines = []
    for name, entries in results.items():
        lines += [f"### {name}", "",
                  "| | W1 | ACF ≤8 | ACF 9–48 | color | TV deficit | SI | xscale value | xscale magnitude |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, e in entries.items():
            lines.append("| " + " | ".join([
                e["label"], _fmt(e.get("w1"), 2), _fmt(e.get("acf_short")), _fmt(e.get("acf_long")),
                _fmt(e.get("color")), _fmt(e["tv_deficit"]), f"{e['sharpness_index']:.3g}",
                _fmt(e["xscale_value"], 2), _fmt(e["xscale_magnitude"], 2)]) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--textures", nargs="+", default=list(TEXTURES), choices=list(TEXTURES))
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("PGNM_RUNS", "runs")) / "textures")
    arguments = parser.parse_args()
    run(arguments.textures, arguments.out)
