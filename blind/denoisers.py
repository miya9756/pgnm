"""Blind denoisers used as the only available "clean" reference.

All take and return uint8 H×W×3 RGB and know nothing about the noise.

* `nlm(h)`: OpenCV non-local means with a fixed filtering strength `h`. A small
  `h` leaves noise, a large one wipes texture: a strength sweep stands in for
  denoisers of different aggressiveness. Like most patch and CNN denoisers it
  leaves part of the low-frequency (chroma) noise in place.
* `dncnn(source)`: the paper's DnCNN checkpoints (Tab. 2), evaluated in NumPy so
  that no PyTorch install is needed. `real` was trained on real SIDD noise of all
  cameras and ISOs, so it is a blind real-noise denoiser for these presets.
"""
from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def nlm(h):
    def denoise(noisy):
        return cv2.fastNlMeansDenoisingColored(noisy, None, h, h, 7, 21)
    denoise.__name__ = f"nlm{h}"
    return denoise


# DnCNN in NumPy ------------------------------------------------------------------

def load_torch_state(path):
    """Read a torch.save'd state dict of float tensors without importing torch."""
    archive = zipfile.ZipFile(path)
    prefix = archive.namelist()[0].split("/")[0]
    dtypes = {"FloatStorage": np.float32, "LongStorage": np.int64, "HalfStorage": np.float16}

    def rebuild(storage, offset, size, stride, *_):
        array = np.lib.stride_tricks.as_strided(
            storage[offset:], shape=size, strides=[s * storage.itemsize for s in stride])
        return np.array(array)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if module == "torch" and name in dtypes:
                return name
            if module == "collections" and name == "OrderedDict":
                from collections import OrderedDict
                return OrderedDict
            raise pickle.UnpicklingError(f"unexpected class {module}.{name}")

        def persistent_load(self, pid):
            _, storage_type, key, _, _ = pid
            raw = archive.read(f"{prefix}/data/{key}")
            return np.frombuffer(raw, dtype=dtypes[storage_type])

    return Unpickler(archive.open(f"{prefix}/data.pkl")).load()


def _conv3x3(x, weight, bias):
    """x: (H, W, Cin) float32; weight: (Cout, Cin, 3, 3); zero padding 1."""
    height, width, channels = x.shape
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    out = np.empty((height, width, weight.shape[0]), np.float32)
    matrix = weight.transpose(2, 3, 1, 0).reshape(9 * channels, -1)   # (dy, dx, cin) x cout
    for start in range(0, height, 64):                                # row tiles bound memory
        stop = min(start + 64, height)
        columns = np.concatenate([padded[start + dy:stop + dy, dx:dx + width]
                                  for dy in range(3) for dx in range(3)], axis=-1)
        out[start:stop] = columns @ matrix + bias
    return out


def dncnn(source="real"):
    state = load_torch_state(ROOT / "checkpoints" / "denoise" / f"{source}.pt")
    layers = []
    for index in sorted({int(key.split(".")[1]) for key in state if key.endswith(".weight")}):
        weight, bias = state[f"dncnn.{index}.weight"], state[f"dncnn.{index}.bias"]
        if weight.ndim == 4:
            layers.append(("conv", weight.astype(np.float32), bias.astype(np.float32)))
        else:   # batch norm (eval mode, eps 1e-4) folded into scale and shift
            scale = weight / np.sqrt(state[f"dncnn.{index}.running_var"] + 1e-4)
            shift = bias - state[f"dncnn.{index}.running_mean"] * scale
            layers.append(("bn", scale.astype(np.float32), shift.astype(np.float32)))

    def denoise(noisy):
        x = noisy.astype(np.float32) / 256.0
        h = x
        for position, (kind, a, b) in enumerate(layers):
            if kind == "conv":
                h = _conv3x3(h, a, b)
            else:
                h = h * a + b
            if position < len(layers) - 1 and (kind == "bn" or position == 0):
                np.maximum(h, 0, out=h)                               # ReLU after conv0 and every BN
        out = np.clip(x - h, 0, 1) * 256.0
        return np.clip(np.floor(out + .5), 0, 255).astype(np.uint8)

    denoise.__name__ = f"dncnn_{source}"
    return denoise
