"""sRGB-Flow (Kousha et al., CVPR 2022): clean + camera + ISO -> noisy.

Upstream expects BGR codes 0..255 and a (camera, ISO) slot of its 5x5 table;
only 15 slots were trained (the others have all-zero gain rows). A request uses
the same camera's nearest trained ISO on a log scale, read from the epoch-60
checkpoint. Sampling at eps_std 1 (native temperature).
"""
import os

import numpy as np

from . import upstream

ISOS = (100, 400, 800, 1600, 3200)
CAMERAS = ["IP", "GP", "S6", "N6", "G4"]
ARCHITECTURE = "CL2|CAC|CAC|CL2|CAC|CAC|CL2|CAC|CAC|CL2|CAC|CAC"


def build(device):
    import torch
    upstream("srgb_flow")
    root = upstream("srgb_flow", "sRGB_noise_modeling")
    from noise_model import NoiseModel

    initial = {"c_i": 1.0, "beta1_i": -5.0, "beta2_i": 0.0, "gain_params_i": np.full(5, -5.0),
               "cam_params_i": np.ones((3, 5))}
    state = torch.load(os.path.join(root, "experiments/sidd/our_model/saved_models/epoch_60_noise_model_net.pth"),
                       map_location="cpu")["state_dict"]
    rows = None  # CL2 tables: row iso_index * 5 + camera_index; zero rows were never trained
    for value in state.values():
        if value.ndim == 2 and value.shape[0] == 25:
            used = value.abs().sum(1) > 0
            rows = used if rows is None else rows | used
    trained = {(CAMERAS[i % 5], ISOS[i // 5]) for i in range(25) if rows[i]}
    models = {}

    def mapped(camera, iso):
        return min((i for c, i in trained if c == camera), key=lambda s: (abs(np.log(s / float(iso))), s))

    def gen(image, cond, seed):
        shape = image.shape[:2]
        if shape not in models:
            model = NoiseModel((3,) + tuple(shape), ARCHITECTURE, 1, initial, True, "cuda", False)
            model.load_state_dict(state)
            models[shape] = model.to("cuda").eval()
        x = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1]).astype(np.float32)).permute(2, 0, 1)[None].cuda()
        iso = torch.full_like(x, float(mapped(cond["camera"], cond["iso"])))
        camera = torch.full_like(x, float(CAMERAS.index(cond["camera"])))
        with torch.no_grad():
            y = models[shape].sample(eps_std=torch.tensor(1.0, device="cuda"), clean=x, iso=iso, cam=camera)
        return np.ascontiguousarray(y[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]) / 255.0

    gen.mapped = mapped
    return gen
