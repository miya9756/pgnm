"""DISTS and LPIPS on delivered gray film images, as GrainNet evaluates them.

Both use the published packages (DISTS_pytorch, lpips with AlexNet) on CPU
with two threads, inside the common 3-pixel border. DISTS receives the gray
image replicated to RGB; LPIPS receives the 1xHxW gray tensor in [0, 1], which
its per-channel input scaling broadcasts to three channels, exactly as in
GrainNet's evaluate.py. Backbone features of an image are computed once and
reused for all of its pairs; the distance formulas themselves are untouched.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np


class _Memo:
    """Cache a deterministic feature function by input identity or content."""

    def __init__(self, function, torch, by_content=False, size=10):
        self.function, self.torch, self.by_content, self.size = function, torch, by_content, size
        self.cache = OrderedDict()

    def __call__(self, tensor):
        key = (hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest(), tuple(tensor.shape)
               ) if self.by_content else id(tensor)
        hit = self.cache.get(key)
        if hit is not None and (self.by_content or hit[0] is tensor):
            self.cache.move_to_end(key)
            return hit[1]
        with self.torch.no_grad():
            features = self.function(tensor)
        self.cache[key] = (tensor, features)
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return features


class Appearance:
    def __init__(self, threads=2):
        import torch
        import lpips
        from DISTS_pytorch import DISTS
        torch.set_num_threads(threads)
        self.torch = torch
        self.dists = DISTS().eval()
        self.lpips = lpips.LPIPS(net="alex")
        self.dists.forward_once = _Memo(self.dists.forward_once, torch)
        # LPIPS rescales its input into a new tensor on every call: key by content.
        self.lpips.net.forward = _Memo(self.lpips.net.forward, torch, by_content=True)
        self.tensors = OrderedDict()

    def _inputs(self, image, border):
        key = (id(image), border)
        hit = self.tensors.get(key)
        if hit is None or hit[0] is not image:
            x = image[border:-border, border:-border].astype(float) / 255
            rgb = self.torch.tensor(np.repeat(x[..., None], 3, -1)).permute(2, 0, 1)[None].float()
            hit = self.tensors[key] = (image, rgb, self.torch.tensor(x).float()[None])
            while len(self.tensors) > 10:
                self.tensors.popitem(last=False)
        return hit[1:]

    def __call__(self, reference, generated, border=3):
        reference_rgb, reference_gray = self._inputs(reference, border)
        generated_rgb, generated_gray = self._inputs(generated, border)
        with self.torch.no_grad():
            return dict(dists=float(self.dists(reference_rgb, generated_rgb)),
                        lpips=float(self.lpips(reference_gray, generated_gray).detach().item()))
