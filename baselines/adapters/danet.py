"""DANet generator (Yue et al., ECCV 2020): clean RGB -> noisy RGB, no conditioning.

networks.UNetG(3, wf=32, depth=5) with model_states/DANet.pt['G'], padded to a
multiple of 2^5 as upstream does; output clamped to [0, 1].
"""
import os

import numpy as np

from . import upstream


def build(device):
    import torch
    root = upstream("danet")
    from networks import UNetG, sample_generator
    from utils import PadUNet

    net = UNetG(3, wf=32, depth=5)
    net.load_state_dict(torch.load(os.path.join(root, "model_states", "DANet.pt"), map_location="cpu")["G"])
    net = net.to(device).eval()

    def gen(image, cond, seed):
        x = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            padded = PadUNet(x, dep_U=5)
            return padded.pad_inverse(sample_generator(net, padded.pad())).clamp_(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    return gen
