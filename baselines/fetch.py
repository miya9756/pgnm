"""Fetch the published baselines into $PGNM_EXTERNAL at the commits we used.

    python -m baselines.fetch                 # all sources below
    python -m baselines.fetch naflow danet    # selected ones
    python -m baselines.fetch --check         # verify weight checksums only

Each repository is cloned to $PGNM_EXTERNAL/<name>/src and checked out at the
pinned commit. Weights hosted outside Git must be downloaded from the listed
page (several hosts need a browser); every weight file is checked against its
SHA-256. Licences differ and most forbid redistribution, so nothing upstream
is copied into this repository.

GMDCN is our implementation of Song et al. (2023) and is not public; its source
tree goes to $PGNM_EXTERNAL/gmdcn/src with models_zoo/sidd_gmdcn.pth.

Newson's renderer is the authors' CUDA implementation. The build copies its
sources, makes a value of -randomizeSeed above 1 the seed itself (upstream
seeds from the clock, so no render could be repeated) and compiles with nvcc
from CUDA 11.8 (the code uses texture references, removed in CUDA 12).
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from experiments.config import external

SOURCES = {
    "naflow": dict(repo="https://github.com/dongjinkim9/NAFlow", commit="533e1eac581af5fbdf79fb9223bbd471c47d9293",
                   weights={"weights/naflow_sidd.ckpt": (
                       "2bd36b5fb079726d3c6de8aa34e3270a1b7ed2d9e4881d21ba55c5103511fe7b",
                       "https://1drv.ms/u/c/85cf5b7f538e2007/EQcgjlN_W88ggIU66gAAAAABfatiu1qFux_8HtaF4Ovliw")}),
    "danet": dict(repo="https://github.com/zsyOAOA/DANet", commit="ea4d714168150e48c66ef0439ea1966342512b0a",
                  weights={"src/model_states/DANet.pt": (
                      "2618b942c06cc74a6eeffdd193b2ea9ca811a09df4ce5c77cf21d3bd412bed3d", "in repository")}),
    "srgb_flow": dict(repo="https://github.com/SamsungLabs/Noise2NoiseFlow",
                      commit="54cc3c11e4fc55600b3550c0300a54e9e5dde4be",
                      weights={"src/sRGB_noise_modeling/experiments/sidd/our_model/saved_models/"
                               "epoch_60_noise_model_net.pth": (
                                   "ae507c0d62fff3dd6584b0235631596d5b9b5853022d0b24055b23356a5f62f7",
                                   "in repository")}),
    "nmflowgan": dict(repo="https://github.com/YoungJooHan/NM-FlowGAN", commit="63983e2c2c5e3280fadda58fe55893f3a6cce3b6",
                      weights={"NMFlowGAN_NoiseModeler.pth": (
                          "859271f5636c8809d1666b89530f6f7cf5b265008f7b0e3da5744585f1218cfa",
                          "https://drive.google.com/file/d/1WaJg2ykB5k1KTd53zm7S9_WPSvg37Sfy")}),
    "grainnet": dict(repo="https://github.com/Gwilherm-LESNE/Neural_Film_Grain_Rendering",
                     commit="865c4940040531db5f9308177b077dcfe3ea122e",
                     weights={"src/models/GrainNet/default/grainnet.pt": (
                         "862428869d39254629c5063c7788f6e90ebb861e6b1f6f9a061105e4fdcf7482", "in repository")}),
    "newson": dict(repo="https://github.com/alasdairnewson/film_grain_rendering_gpu",
                   commit="5695ff2abea762c90740a0ae309336dc9be4cd95", weights={}),
    "gmdcn": dict(repo=None, commit=None, weights={"src/models_zoo/sidd_gmdcn.pth": (
        "ccff42634cec832876af90b58e2c0abd5ee7d9135285d6735b3f9dedfbc9f6c6", "authors' implementation")}),
}
SEED_PATCH = ("        if (filmGrainOptions.randomizeSeed > 0) {",
              "        if (filmGrainOptions.randomizeSeed > 1) {\n            seed = filmGrainOptions.randomizeSeed;\n"
              "        } else if (filmGrainOptions.randomizeSeed > 0) {")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def clone(name):
    spec, target = SOURCES[name], external(name, "src")
    if spec["repo"] is None:
        print("%-10s place the source tree at %s" % (name, target))
        return
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", spec["repo"], str(target)], check=True)
    subprocess.run(["git", "checkout", "--quiet", spec["commit"]], cwd=target, check=True)
    print("%-10s %s" % (name, spec["commit"][:12]))


def check_weights(name):
    ok = True
    for relative, (digest, where) in SOURCES[name]["weights"].items():
        path = external(name, relative)
        if not path.is_file():
            print("%-10s missing %s (%s)" % (name, path, where))
            ok = False
        elif sha256(path) != digest:
            print("%-10s CHECKSUM MISMATCH %s" % (name, path))
            ok = False
    return ok


def build_newson(nvcc_prefix=None):
    """Patched build copy of the CUDA renderer: external/newson/build/bin/..."""
    source, build = external("newson", "src"), external("newson", "build")
    shutil.rmtree(build, ignore_errors=True)
    (build / "obj").mkdir(parents=True)
    (build / "bin").mkdir()
    shutil.copytree(source / "src", build / "src")
    shutil.copy(source / "makefile", build / "makefile")
    path = build / "src" / "film_grain_rendering.cu"
    text = path.read_text()
    if text.count(SEED_PATCH[0]) != 1:
        raise RuntimeError("Unexpected Newson source")
    path.write_text(text.replace(*SEED_PATCH))
    prefix = Path(nvcc_prefix or os.environ.get("CONDA_PREFIX", "/usr/local/cuda"))
    subprocess.run(["make", "-s", "-C", str(build),
                    f"NVCC={prefix}/bin/nvcc -arch=sm_86 -I{prefix}/include", f"CC=gcc -I{prefix}/include",
                    f"LIB=-L{prefix}/lib -Xlinker -rpath -Xlinker {prefix}/lib -lpng -ltiff -lglut -lGL"], check=True)
    print("newson     built", build / "bin" / "film_grain_rendering_main")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="any of: " + ", ".join(SOURCES))
    parser.add_argument("--check", action="store_true", help="only verify weight files")
    parser.add_argument("--nvcc-prefix", help="CUDA 11.8 prefix for the Newson build")
    args = parser.parse_args()
    names = args.names or list(SOURCES)
    unknown = set(names) - set(SOURCES)
    if unknown:
        parser.error("unknown source: " + ", ".join(sorted(unknown)))
    if not args.check:
        for name in names:
            clone(name)
        if "newson" in names:
            build_newson(args.nvcc_prefix)
    missing = [name for name in names if not check_weights(name)]
    print("all weights verified" if not missing else "incomplete: " + ", ".join(missing))


if __name__ == "__main__":
    main()
