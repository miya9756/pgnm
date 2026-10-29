"""Random streams behind every reported number.

Each generated image has its own seed, derived by hashing a fixed label, so any
single image can be regenerated without the others. The literal labels below
(for example ``"pgnm-protocol3"``) are part of the recorded streams: changing
one changes every sample drawn from it.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Five generator runs; run r uses replicate year 2026 + r in the hashed label.
RUNS = range(5)
REPLICATE_LABELS = (2026, 2027, 2028, 2029, 2030)


def stream_seed(namespace, setting, source, run, subitem=0):
    """Seed for one item, e.g. ("score", "S6_800", image, run, block)."""
    payload = "|".join(map(str, ("pgnm-protocol3", namespace, setting, source,
                                 REPLICATE_LABELS[run], subitem)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**31 - 1)


def bank_seeds(run):
    """The four noise-field seeds of PGNM's template bank in generator run r."""
    return tuple(6100 + 4 * run + j for j in range(4))


def radius_key(radius):
    """0.025 -> 'r025': file and seed label of a film-grain radius."""
    return "r%03d" % round(1000 * radius)


def film_seed(namespace, radius, image, run):
    """Seed of a film-grain baseline or Newson reference render."""
    return 2 + stream_seed(namespace, radius_key(radius), image, run) % (2**31 - 3)


def film_pgnm_seed(radius, image, run):
    """Window-selection seed of PGNM on one film-grain test image."""
    label = f"cvpr-film-pgnm:{radius:.3f}:{image}:{run}"
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")


def to_uint8(image):
    """Delivered 8-bit observation: clip to [0, 1], then round half up."""
    return np.floor(np.clip(np.asarray(image, np.float64), 0, 1) * 255 + .5).astype(np.uint8)
