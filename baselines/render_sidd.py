"""Render a published generator on the SIDD Validation blocks (Tab. 1).

    python -m experiments.sidd export-clean                 # once, base environment
    python -m baselines.render_sidd --method naflow         # in the NAFlow environment

Every block gets its own seed, stream_seed("score", class, image, run, block),
and every random generator is reseeded with it before the block is rendered.
Released checkpoints are used at their native settings without retraining.
sRGB-Flow and NM-FlowGAN cover the camera-ISO classes of 19 sources only.
"""
import argparse

import numpy as np

from experiments import config
from experiments.seeds import RUNS, stream_seed, to_uint8
from .adapters import build, seed_all

SOURCES = dict(naflow="S39", gmdcn="S39", danet="S39", srgb_flow="S19", nmflowgan="S19")


def conditioning(method, camera, iso):
    cond = dict(camera=camera, iso=int(iso))
    if method == "srgb_flow":
        cond["epoch"] = 60
    return cond


def render(method, images=None, runs=RUNS, device="cuda"):
    import torch
    torch.set_num_threads(2)
    split = config.dataset("sidd")
    clean = np.load(config.runs("sidd/validation_clean.npy"), mmap_mode="r")
    generator = build(method, device)
    for image in images or split["sets"][SOURCES[method]]:
        label = split["validation"][image]
        name = "%s_%s" % (label["camera"], label["iso"])
        for run in runs:
            target = config.runs("sidd/generated", method, "r%d" % run, "img%02d.npy" % image)
            if target.exists():
                continue
            blocks = []
            for block, x in enumerate(clean[image]):
                seed = stream_seed("score", name, image, run, block)
                seed_all(seed)
                blocks.append(to_uint8(generator(np.asarray(x), conditioning(method, label["camera"], label["iso"]), seed)))
            config.save_array(target, np.stack(blocks))
            print("rendered", method, image, run, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=tuple(SOURCES))
    parser.add_argument("--images", nargs="+", type=int)
    parser.add_argument("--runs", nargs="+", type=int, default=list(RUNS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render(args.method, args.images, args.runs, args.device)


if __name__ == "__main__":
    main()
