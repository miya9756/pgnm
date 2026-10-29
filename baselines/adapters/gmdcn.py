"""GMDCN (Song et al. 2023): residual from [RGB, crc32(camera)/2^32, ISO/1e4].

Our implementation of the published model, trained on SIDD-Medium. Its source
tree (package noise_synthesizer) and checkpoint are not part of this
repository; place them at $PGNM_EXTERNAL/gmdcn/src.
"""
import os

import numpy as np

from . import upstream


def build(device):
    import torch
    root = upstream("gmdcn")
    from noise_synthesizer.config import NoiseSynthExperimentConfig, build_generator
    from noise_synthesizer.generate import _build_condition_map

    generator = build_generator(NoiseSynthExperimentConfig.from_yaml(
        os.path.join(root, "configs", "gmdcn.yaml")).generator)
    generator.load_state_dict(torch.load(os.path.join(root, "models_zoo", "sidd_gmdcn.pth"),
                                         map_location="cpu"), strict=True)
    generator = generator.to(device).eval()

    def gen(image, cond, seed):
        x = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        condition = _build_condition_map(cond["camera"], float(cond["iso"]), x.shape[2], x.shape[3], device)
        with torch.no_grad():
            return (x + generator(torch.cat([x, condition], 1))).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    return gen
