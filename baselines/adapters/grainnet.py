"""GrainNet (Lesne et al., CGF 2025): grayscale film grain at radius r.

A gray input (R = G = B) is rendered once with seed 3*seed and replicated, as
GrainNet's own editing script renders each channel with its own grain.
"""
import os

import numpy as np

from . import upstream


def build(device):
    import torch
    root = upstream("grainnet")
    import net as nets

    model = nets.GrainNet(activation="tanh", block_nb=2)
    model.load_state_dict(torch.load(os.path.join(root, "models", "GrainNet", "default", "grainnet.pt"),
                                     map_location="cpu"), strict=True)
    model = model.to(device).eval()

    def gen(image, cond, seed):
        gray = bool((image[..., 0] == image[..., 1]).all() and (image[..., 1] == image[..., 2]).all())
        channels = [image[..., 0]] if gray else [image[..., i] for i in range(3)]
        radius = torch.full((1, 1), float(cond["radius"]), device=device)
        out = []
        with torch.no_grad():
            for i, channel in enumerate(channels):
                x = torch.from_numpy(channel.astype(np.float32) / 255.0)[None, None].to(device)
                out.append(model(x, radius, seed=seed * 3 + i)[0, 0].cpu().numpy())
        return np.stack(out * 3 if gray else out, -1)

    return gen
