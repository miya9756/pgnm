"""Render GrainNet on the 60 film-grain test images (Tab. 3).

    python -m baselines.render_film            # in the GrainNet environment

GrainNet takes the clean image x (no blur B) and the radius; its seed is
film_seed("film3-lesne", radius, image, run), set everywhere before each image.
"""
import argparse

import numpy as np

from experiments import config, film
from experiments.seeds import RUNS, film_seed, to_uint8
from .adapters import build, seed_all


def render(radii=film.RADII, images=None, device="cuda"):
    generator = build("grainnet", device)
    for radius in radii:
        for run in RUNS:
            for image in images or film.split()["test_images"]:
                target = film.output_path("grainnet", radius, run, image)
                if target.exists():
                    continue
                clean = film.gray(film.test_image_path(image))
                seed = film_seed("film3-lesne", radius, image, run)
                seed_all(seed)
                output = generator(np.repeat(clean[..., None], 3, -1), dict(radius=radius), seed)
                config.save_array(target, to_uint8(output[..., 0]))
            print("rendered grainnet", film.radius_key(radius), run, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radii", nargs="+", type=float, choices=film.RADII, default=list(film.RADII))
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render(args.radii, args.images, args.device)


if __name__ == "__main__":
    main()
