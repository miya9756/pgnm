"""One adapter per published generator: build(device) -> gen(clean, cond, seed).

``gen`` receives a clean uint8 RGB image (H, W, 3), a conditioning dict
(camera, iso, ... or radius) and the item's seed, and returns the noisy image
as float in [0, 1]; callers deliver it with experiments.seeds.to_uint8. Each
adapter runs inside its upstream environment (baselines/environments.sh) and
imports upstream code from $PGNM_EXTERNAL/<name>/src.
"""
import importlib
import random
import sys

import numpy as np

from experiments.config import external


def upstream(name, *sub):
    """Put $PGNM_EXTERNAL/<name>/src[/sub] first on sys.path and return it."""
    path = str(external(name, "src", *sub))
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def seed_all(seed):
    """Seed Python, NumPy and torch before every item."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build(method, device):
    return importlib.import_module("baselines.adapters." + method).build(device)
