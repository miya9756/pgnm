"""Fit aligned RGB arrays or render an RGB image with the converged PGNM."""
import argparse
from pathlib import Path

import numpy as np

from . import PGNM, fit, load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit", help="NPZ clean/noisy arrays, shaped N×H×W×3 in [0,1]")
    fitting.add_argument("pairs", type=Path)
    fitting.add_argument("output", type=Path)
    fitting.add_argument("--kernel-size", type=int, default=25)
    fitting.add_argument("--threshold", type=float)
    rendering = commands.add_parser("render", help="Render one RGB image")
    rendering.add_argument("input", type=Path)
    rendering.add_argument("output", type=Path)
    rendering.add_argument("--params", type=Path, required=True)
    rendering.add_argument("--seed", type=int, default=2026)
    rendering.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "fit":
        with np.load(args.pairs, allow_pickle=False) as data:
            clean, noisy = data["clean"], data["noisy"]
            if clean.ndim != 4 or clean.shape != noisy.shape or clean.shape[-1] != 3:
                raise ValueError("clean and noisy must have matching (N,H,W,3) shapes")
            params = fit(list(zip(clean, noisy)), kernel_size=args.kernel_size,
                         threshold=args.threshold,
                         progress=lambda message: print(message, flush=True))
        save(args.output, params)
    else:
        from PIL import Image

        with Image.open(args.input) as image:
            clean = np.asarray(image.convert("RGB"), np.float32) / 255
        model = PGNM(load(args.params), workers=args.workers)
        noisy = model.render(clean, np.random.default_rng(args.seed))
        codes = np.floor(np.clip(noisy.astype(np.float64), 0, 1) * 255 + 0.5).astype(np.uint8)
        Image.fromarray(codes).save(args.output)


if __name__ == "__main__":
    main()
