"""Where datasets, external baseline code and outputs live.

Every location can be set with an environment variable; the defaults expect a
``data/`` and an ``external/`` folder next to this repository's README.

    PGNM_SIDD_MEDIUM      SIDD-Medium sRGB, the folder that contains Data/
    PGNM_SIDD_VALIDATION  folder with ValidationGtBlocksSrgb.mat and
                          ValidationNoisyBlocksSrgb.mat
    PGNM_FILM             GrainNet's NeuralFilmGrainRendering training data
    PGNM_EXTERNAL         fetched baseline repositories, weights and tools
                          (python -m baselines.fetch)
    PGNM_RUNS             all generated outputs, scores and tables
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "datasets"


def _path(variable, default):
    return Path(os.environ.get(variable, str(default))).expanduser()


def sidd_medium():
    return _path("PGNM_SIDD_MEDIUM", REPO / "data/SIDD_Medium_Srgb")


def sidd_validation():
    return _path("PGNM_SIDD_VALIDATION", REPO / "data/SIDD_Validation")


def film_data():
    return _path("PGNM_FILM", REPO / "data/NeuralFilmGrainRendering")


def external(*parts):
    return _path("PGNM_EXTERNAL", REPO / "external").joinpath(*parts)


def runs(*parts):
    return _path("PGNM_RUNS", REPO / "runs").joinpath(*parts)


def dataset(name):
    """Read one split description from datasets/<name>.json."""
    return json.loads((DATASETS / (name + ".json")).read_text())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def save_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)
