"""Validate and serialize the converged PGNM's direct-filter parameters."""
import copy
import json
from pathlib import Path

import numpy as np

SCHEMA = "pgnm/5"


def _numeric_array(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError
        array = raw.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain numbers") from None
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def validate(value):
    """Return an independent, JSON-ready preset after checking its invariants.

    Historical presets need refitting: the direct tensor and shared correlation
    mask are a different estimator, so silently relabeling them is not a migration.
    """
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(
            "Expected schema pgnm/5. Historical pgnm/1–4 presets require refitting; "
            "use the archived implementation to reproduce historical results."
        )
    if value.get("method") != "ours" or value.get("name") != "PGNM":
        raise ValueError("Expected method 'ours' and name 'PGNM'")
    if "alpha" in value:
        raise ValueError("PGNM always uses alpha=1; remove the alpha parameter")
    unknown = set(value) - {"schema", "method", "name", "kernels", "marginal", "provenance"}
    if unknown:
        raise ValueError(f"Unknown PGNM parameters: {', '.join(sorted(unknown))}")
    params = copy.deepcopy(value)
    kernels = _numeric_array(params.get("kernels"), "kernels")
    if (
        kernels.ndim != 4
        or kernels.shape[2:] != (3, 3)
        or any(size < 1 or size % 2 == 0 for size in kernels.shape[:2])
    ):
        raise ValueError("kernels must have odd positive (height, width, 3 input, 3 output) shape")
    if not np.allclose(np.sum(kernels * kernels, axis=(0, 1, 2)), 1, rtol=0, atol=1e-5):
        raise ValueError("kernels must have unit squared energy per output channel")
    marginal = params.get("marginal")
    if not isinstance(marginal, dict) or set(marginal) != {"t", "g", "step"}:
        raise ValueError("marginal must contain exactly t, g and step")
    knots = _numeric_array(marginal["t"], "marginal.t")
    if (
        knots.shape != (25,)
        or not np.all(np.diff(knots) > 0)
        or not np.allclose(np.diff(knots), knots[1] - knots[0], rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("marginal.t must contain 25 increasing, uniformly spaced Gaussian knots")
    lookup = _numeric_array(marginal["g"], "marginal.g")
    if lookup.shape != (3, 256, 25) or np.any(np.abs(lookup) > 1):
        raise ValueError("marginal.g must have shape (3, 256, 25) with |residual| <= 1")
    if np.any(np.diff(lookup, axis=2) < -1e-9):
        raise ValueError("marginal.g must be non-decreasing along the Gaussian axis")
    if isinstance(marginal["step"], bool) or marginal["step"] != 1.0:
        raise ValueError("marginal.step must be 1.0 residual code unit")
    provenance = params.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    params.update(
        kernels=kernels.tolist(),
        marginal={"t": knots.tolist(), "g": lookup.tolist(), "step": 1.0},
        provenance=provenance,
    )
    try:
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("provenance must contain finite JSON values") from None
    return params


def load(path):
    """Read and validate a pgnm/5 JSON preset."""
    return validate(json.loads(Path(path).read_text()))


def save(path, params):
    """Validate and write a portable pgnm/5 JSON preset."""
    Path(path).write_text(json.dumps(validate(params), allow_nan=False) + "\n")
