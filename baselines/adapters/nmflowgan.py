"""NM-FlowGAN (Han & Yu, 2023): official released noise modeler, camera + ISO.

The checkpoint works in codes 0..255 and clips its sample to [0, 256];
returning sample/255 (in float64, so no value crosses a half-code boundary)
lets the common 8-bit delivery reproduce upstream's floor(sample + 0.5).
ISO values outside upstream's five keep upstream's own fallback to the ISO 100
embedding.
"""
import os

import numpy as np

from . import upstream
from experiments.config import external

CAMERAS = ("IP", "GP", "S6", "N6", "G4")


def build(device):
    import torch
    import yaml
    root = upstream("nmflowgan")
    from core.model.NMFlowGAN import NMFlowGANGenerator

    with open(os.path.join(root, "conf", "NMFlowGAN_NoiseModeler.yaml")) as handle:
        config = yaml.safe_load(handle)
    model = NMFlowGANGenerator(kwargs_unet=config["MODEL"]["UNET"], kwargs_flow=config["MODEL"]["NMFLOW"])
    checkpoint = torch.load(str(external("nmflowgan", "NMFlowGAN_NoiseModeler.pth")), map_location="cpu",
                            weights_only=True)
    model.load_state_dict(checkpoint["model_weight"]["generator"], strict=True)
    model = model.to(device).eval()

    def gen(image, cond, seed):
        x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        inputs = {"clean": x, "ISO-level": torch.tensor([float(cond["iso"])], device=device),
                  "smartphone-code": torch.tensor([float(CAMERAS.index(cond["camera"]))], device=device)}
        with torch.no_grad():
            y = model.sample(inputs)
        return np.ascontiguousarray(y[0].permute(1, 2, 0).cpu().numpy(), dtype=np.float64) / 255.0

    return gen
