"""Fetch and build the video codecs of the delivery experiment (Tab. 5).

    python -m baselines.codecs            # clone and build all three into $PGNM_EXTERNAL
    python -m baselines.codecs --check    # print the version of each built tool

libaom v3.6.0 with its examples (``noise_model`` estimates AV1 film-grain
tables), Fraunhofer VVenC v1.14.0 and VVdeC v3.2.0. Each repository is cloned
at its release tag, checked against the commit below and built in Release mode
with the system compilers (GCC 11.4 and CMake 3.22 on Ubuntu 22.04 for the
paper):

    libaom  cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_EXAMPLES=1 -DENABLE_TESTS=0
                  -DENABLE_DOCS=0 -DENABLE_TOOLS=0, targets aomenc aomdec noise_model
    VVenC   cmake -DCMAKE_BUILD_TYPE=Release (static), target vvencFFapp
    VVdeC   cmake -DCMAKE_BUILD_TYPE=Release (static), target vvdecapp

The experiment also needs ffmpeg 4.4 on the PATH (Ubuntu 22.04's 4.4.2 package;
set PGNM_FFMPEG to use another binary). It decodes the JPEG 2000 and ProRes
sources and converts the grain version to RGB.
"""
from __future__ import annotations

import argparse
import os
import subprocess

from experiments.config import external

CODECS = {
    "libaom": dict(repo="https://aomedia.googlesource.com/aom", tag="v3.6.0",
                   commit="3c65175b1972da4a1992c1dae2365b48d13f9a8d", source="libaom/src", build="libaom/build",
                   options=("-DENABLE_EXAMPLES=1", "-DENABLE_TESTS=0", "-DENABLE_DOCS=0", "-DENABLE_TOOLS=0"),
                   targets=("aomenc", "aomdec", "noise_model")),
    "vvenc": dict(repo="https://github.com/fraunhoferhhi/vvenc", tag="v1.14.0",
                  commit="9428ea8636ae7f443ecde89999d16b2dfc421524", source="vvenc", build="vvenc/build",
                  options=(), targets=("vvencFFapp",)),
    "vvdec": dict(repo="https://github.com/fraunhoferhhi/vvdec", tag="v3.2.0",
                  commit="649f0b2fafee977c998d7e4d674f8b88d952e3a3", source="vvdec", build="vvdec/build",
                  options=(), targets=("vvdecapp",)),
}
AOMENC = external("libaom/build/aomenc")
AOMDEC = external("libaom/build/aomdec")
NOISE_MODEL = external("libaom/build/examples/noise_model")
VVENC = external("vvenc/bin/release-static/vvencFFapp")
VVDEC = external("vvdec/bin/release-static/vvdecapp")
FFMPEG = os.environ.get("PGNM_FFMPEG", "ffmpeg")


def build(name):
    spec = CODECS[name]
    source, build_dir = external(spec["source"]), external(spec["build"])
    if not (source / ".git").exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", spec["tag"], spec["repo"],
                        str(source)], check=True)
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != spec["commit"]:
        raise RuntimeError(f"{name}: {source} is at {commit}, expected {spec['commit']}")
    subprocess.run(["cmake", "-S", str(source), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
                    *spec["options"]], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 2),
                    "--target", *spec["targets"]], check=True)
    print("%-7s %s built" % (name, spec["tag"]))


def check():
    for tool, flag, marker in ((AOMENC, "--help", "AV1 Encoder"), (AOMDEC, "--help", "AV1 Decoder"),
                               (VVENC, "--version", "version"), (VVDEC, "--version", "version")):
        result = subprocess.run([str(tool), flag], capture_output=True, text=True)
        lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if marker in line]
        print("%-12s %s" % (tool.name, lines[0] if lines else "MISSING"))
    print("%-12s %s" % ("noise_model", "found" if NOISE_MODEL.is_file() else "MISSING"))
    print("%-12s %s" % ("ffmpeg", subprocess.run([FFMPEG, "-version"], capture_output=True,
                                                 text=True).stdout.splitlines()[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="any of: " + ", ".join(CODECS))
    parser.add_argument("--check", action="store_true", help="only print the versions of the built tools")
    args = parser.parse_args()
    unknown = set(args.names) - set(CODECS)
    if unknown:
        parser.error("unknown codec: " + ", ".join(sorted(unknown)))
    if not args.check:
        for name in args.names or CODECS:
            build(name)
    check()


if __name__ == "__main__":
    main()
