"""NAFlow (Kim et al., ICLR 2024), class-conditional sampling at native temperature.

z = sigma_c * eps + mu_c for the class CAM_ISO (the nearest ISO of the same
camera if the class is absent), decoded with the clean image plus the 3/255
condition jitter of NAFlow's own sampling; output clamped to [0, 1].
"""
import os

import numpy as np

from . import upstream
from experiments.config import external


def build(device):
    import torch
    root = upstream("naflow", "naflow")
    from omegaconf import OmegaConf
    from archs.naflow import NAFlow

    config = OmegaConf.load(os.path.join(root, "configs", "models", "naflow.yaml"))
    net = NAFlow((config.params if "params" in config else config).network_G_config)
    state = torch.load(str(external("naflow", "weights", "naflow_sidd.ckpt")), map_location="cpu")["state_dict"]
    net.load_state_dict({k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}, strict=True)
    net = net.to(device).eval()
    classes = list(net.classes)

    def nearest_class(camera, iso):
        wanted = "%s_%05d" % (camera, int(iso))
        if wanted in classes:
            return wanted
        isos = [int(c.split("_")[1]) for c in classes if c.startswith(camera + "_")]
        return "%s_%05d" % (camera, min(isos, key=lambda s: (abs(np.log(s / float(iso))), s)))

    def gen(image, cond, seed):
        x = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        label = nearest_class(cond["camera"], cond["iso"])
        with torch.no_grad():
            eps = torch.randn(1, net.flowUpsamplerNet.C, x.shape[2] // 8, x.shape[3] // 8, device=device)
            z = net.cov_shifts[label].view(1, -1, 1, 1) * eps + net.mean_shifts[label].view(1, -1, 1, 1)
            condition = (x + torch.randn_like(x) * 3.0 / 255).clamp(0, 1)
            y, _ = net(cond=condition, z=z, eps_std=1.0, reverse=True, label=[label])
        return y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    gen.classes = classes
    return gen
