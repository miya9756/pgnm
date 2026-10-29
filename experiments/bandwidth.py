"""Grain synthesis for video delivery (Tab. 5).

    python -m baselines.codecs                    # libaom, VVenC, VVdeC (once)
    python -m experiments.bandwidth prepare       # the 100-frame clip pair (downloads ~0.5 GB)
    python -m experiments.bandwidth fit           # PGNM and AV1 grain models (optional)
    python -m experiments.bandwidth encode        # the eight bitstreams behind Tab. 5
    python -m experiments.bandwidth decode        # decode, add grain, keep the delivered RGB
    python -m experiments.bandwidth score         # W1 and correlation errors, all 100 frames
    python -m experiments.bandwidth appearance    # DISTS, LPIPS (grainnet environment)
    python -m experiments.bandwidth table         # Tab. 5

Netflix's Nocturne (CC BY 4.0) comes as a clean master and as a version with
synthetic film grain added to the same master; the grain version is the
reference. The clip is the shot of the AOM test material whose first frame has
median luma closest to 128 (master frames 2370-2539), first 100 frames (1.67 s
at 60 fps). Clean frames are read from the shot's AOM archive (JPEG 2000,
12-bit), grain frames from Netflix's ProRes file, both over HTTP. Both are
cropped to the center 1920x1080 of the 4K frame and mapped from HDR (PQ, P3)
to 8-bit SDR by one fixed tone curve. Every encoder sees BT.709 limited-range
8-bit 4:2:0.

Outputs are named arm/rate, e.g. pgnm/55. Rows of Tab. 5:
* grain coded: the grain version encoded by libaom at Q 15 (av1, and
  av1_nofilter without alt-ref and keyframe filtering) and by VVenC at QP 17
  (vvc, and vvc_nofilter without MCTF); the setting with lower DISTS is shown.
* grain synthesized after decoding: the clean master encoded by libaom, plus
  AV1's grain table (av1_grain: libaom's noise_model on clip frames 0, 10,
  ..., 90 of the clean and grain versions; aomdec adds the grain) or PGNM
  (pgnm, fitted on the same frames). VVC's grain (vvc_grain) comes from
  VVenC's own film-grain analysis of the grain version, which takes no clean
  input, with two-pass rate control at 935 kbps, the total rate of PGNM at
  Q 55; VVdeC adds the grain.

PGNM renders decoded frame t with bank seeds 6100-6103 and window seed 9000+t,
frames in order. Its rate includes its model: the zlib-compressed float64
kernel and lookup arrays (161,269 bytes, 774 kbps on this clip). Scores are
against the grain version: conditional W1 and the spatial and color
correlation errors (metrics.residual, the clean master as clean image) on all
100 frames, DISTS and LPIPS on frames 0, 4, ..., 96; each is averaged over
frames. `score` needs the base environment (numba), `appearance` torch,
DISTS_pytorch and lpips (bash baselines/environments.sh grainnet); the paper
scored on CPU, except DISTS/LPIPS of av1_nofilter, vvc, vvc_nofilter and
vvc_grain on GPU (--device cuda; within 1e-4 of CPU).

The text's matched-DISTS saving (at least 90%) comes from the full rate
ladders in the supplementary material: with --ladder the stages also run Q 15-63
(libaom) and QP 17-47 (VVenC) for the coded arms and PGNM, and `table` prints
PGNM's smallest saving over the four coded arms. The nlmeans-denoised variant
(85%) is not included.

Files: runs/bandwidth/{clip,fits,bitstreams,delivered,scores,appearance}/.
Stages skip outputs that exist; listing outputs restricts a stage to them.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request
import zipfile
import zlib

import numpy as np

from baselines.codecs import AOMDEC, AOMENC, FFMPEG, NOISE_MODEL, VVDEC, VVENC
from . import config
from .seeds import bank_seeds, to_uint8

BUCKET = "https://s3.amazonaws.com/download.opencontent.netflix.com/"
GRAIN_URL = BUCKET + "Nocturne/nocturne_filmgrain/nocturne_60fps_uhd_filmgrain_hdr.mov"
SHOT_URL = BUCKET + "aom_test_materials/HDR/nocturne/{start}-{end}.zip"
# The eight Nocturne shots of the AOM test material (inclusive master frame ranges).
SHOTS = ((2370, 2539), (8540, 9009), (9010, 9349), (17140, 17709), (18013, 18315),
         (23820, 24322), (27740, 28109), (32660, 32799))
FPS, FRAMES = 60, 100
HEIGHT, WIDTH, TOP, LEFT = 1080, 1920, 540, 960
FIT_FRAMES = tuple(range(0, FRAMES, 10))
APPEARANCE_FRAMES = tuple(range(0, FRAMES, 4))
WINDOW_SEED = 9000

# arm: (codec, encoded clip, grain added after decoding, extra encoder options)
ARMS = {
    "av1": ("av1", "grain", None, ()),
    "av1_nofilter": ("av1", "grain", None, ("--arnr-strength=0", "--enable-keyframe-filtering=0")),
    "vvc": ("vvc", "grain", None, ()),
    "vvc_nofilter": ("vvc", "grain", None, ("--MCTF", "0", "--BIM", "0")),
    "av1_grain": ("av1", "clean", "av1", ()),
    "vvc_grain": ("vvc", "grain", "vvc", ("--fga", "1")),
    "pgnm": ("av1", "clean", "pgnm", ()),
}
# Tab. 5: (encoder, grain, outputs); with two outputs the one with lower DISTS is shown.
# 935k: two-pass target, the total rate of PGNM at Q 55 (935 kbps).
TABLE = (("AV1, Q 15", "coded", ("av1/15", "av1_nofilter/15")),
         ("VVC, QP 17", "coded", ("vvc/17", "vvc_nofilter/17")),
         ("AV1, Q 31", "AV1", ("av1_grain/31",)),
         ("VVC, 2-pass", "VVC", ("vvc_grain/935k",)),
         ("AV1, Q 31", "PGNM", ("pgnm/31",)),
         ("AV1, Q 55", "PGNM", ("pgnm/55",)))
LADDERS = dict(av1=(15, 23, 31, 39, 47, 55, 63), vvc=(17, 22, 27, 32, 37, 42, 47))
CODED = ("av1", "av1_nofilter", "vvc", "vvc_nofilter")

# libaom's film-grain example settings (examples/noise_model.c), constant quality.
AOMENC_OPTIONS = ("--good", "--cpu-used=4", "--lag-in-frames=25", "--auto-alt-ref=2", "--threads=2",
                  "--bit-depth=8", "--input-bit-depth=8", "--i420", "-w", str(WIDTH), "-h", str(HEIGHT),
                  f"--fps={FPS}/1", f"--limit={FRAMES}")
VVENC_OPTIONS = ("--SourceWidth", str(WIDTH), "--SourceHeight", str(HEIGHT), "--InputBitDepth", "8",
                 "--InputChromaFormat", "420", "-fr", str(FPS), "-f", str(FRAMES), "--preset", "medium",
                 "--PerceptQPA", "1", "--Threads", "2")


def outputs(ladder=False):
    """Every encoded output, 'arm/rate', of Tab. 5 (and of the full rate ladders)."""
    keys = [key for _, _, group in TABLE for key in group]
    if ladder:
        keys += [f"{arm}/{q}" for arm in CODED + ("pgnm",) for q in LADDERS[ARMS[arm][0]]]
    return list(dict.fromkeys(keys))


def root(*parts):
    return config.runs("bandwidth", *parts)


# HDR to SDR, identical for both versions --------------------------------------------

_P3_TO_709 = np.array([[1.2249401, -0.2249404, 0.0],
                       [-0.0420569, 1.0420571, 0.0],
                       [-0.0196376, -0.0786361, 1.0982735]])
_REFERENCE_WHITE, _PEAK, _EXPOSURE = 203.0, 4000.0, 2.0


def pq_eotf(signal):
    """SMPTE ST 2084: code value in [0, 1] -> cd/m^2."""
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    e = np.power(np.clip(signal, 0, 1), 1 / m2)
    return 10000 * np.power(np.maximum(e - c1, 0) / (c2 - c3 * e), 1 / m1)


def _hable(x):
    a, b, c, d, e, f = .15, .50, .10, .20, .02, .30
    return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f


def tone_map(pq_rgb):
    """PQ P3-D65 RGB in [0, 1] -> BT.709 SDR uint8 RGB.

    Linear BT.709 light relative to the 203 cd/m^2 reference white; a Hable
    filmic curve (exposure 2, white at the 4000 cd/m^2 peak) on max(R, G, B)
    scales all three channels alike, which keeps hue; then the inverse BT.1886
    curve (gamma 2.4).
    """
    linear = np.maximum(pq_eotf(np.asarray(pq_rgb, np.float64)) @ _P3_TO_709.T, 0) / _REFERENCE_WHITE
    peak = linear.max(axis=-1, keepdims=True)
    white = _hable(_EXPOSURE * _PEAK / _REFERENCE_WHITE)
    scale = np.divide(_hable(_EXPOSURE * peak) / white, peak, out=np.zeros_like(peak), where=peak > 0)
    return to_uint8(np.power(np.clip(linear * scale, 0, 1), 1 / 2.4))


def luma(rgb):
    return np.asarray(rgb, np.float64) @ np.array([.2126, .7152, .0722])


def crop(frame):
    return frame[TOP:TOP + HEIGHT, LEFT:LEFT + WIDTH]


# Codec domain: BT.709 limited-range 8-bit Y'CbCr 4:2:0, center-sited chroma ---------

_KR, _KB = .2126, .0722
_KG = 1 - _KR - _KB
FRAME_BYTES = HEIGHT * WIDTH * 3 // 2


def rgb_to_i420(rgb):
    """One HxWx3 uint8 RGB frame -> I420 bytes (Y, then 2x2-mean Cb and Cr)."""
    x = np.asarray(rgb, np.float64) / 255
    y = _KR * x[..., 0] + _KG * x[..., 1] + _KB * x[..., 2]
    cb = (x[..., 2] - y) / (2 * (1 - _KB))
    cr = (x[..., 0] - y) / (2 * (1 - _KR))
    h, w = y.shape
    pool = lambda c: c.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    planes = [16 + 219 * y, 128 + 224 * pool(cb), 128 + 224 * pool(cr)]
    return b"".join(np.clip(np.floor(p + .5), 0, 255).astype(np.uint8).tobytes() for p in planes)


def _upsample(c):
    """Bilinear 2x upsampling of center-sited chroma, edges replicated."""
    def axis(a, ax):
        prev = np.concatenate([np.take(a, [0], ax), np.take(a, range(a.shape[ax] - 1), ax)], ax)
        nxt = np.concatenate([np.take(a, range(1, a.shape[ax]), ax), np.take(a, [-1], ax)], ax)
        even, odd = .25 * prev + .75 * a, .75 * a + .25 * nxt
        return np.stack([even, odd], ax + 1).reshape(a.shape[:ax] + (2 * a.shape[ax],) + a.shape[ax + 1:])
    return axis(axis(c, 0), 1)


def i420_to_rgb(frame):
    """I420 bytes of one frame -> HxWx3 uint8 RGB."""
    data = np.frombuffer(frame, np.uint8)
    n, m = HEIGHT * WIDTH, HEIGHT * WIDTH // 4
    y = (data[:n].reshape(HEIGHT, WIDTH) - 16.) / 219
    cb = _upsample((data[n:n + m].reshape(HEIGHT // 2, WIDTH // 2) - 128.) / 224)
    cr = _upsample((data[n + m:n + 2 * m].reshape(HEIGHT // 2, WIDTH // 2) - 128.) / 224)
    r = y + 2 * (1 - _KR) * cr
    b = y + 2 * (1 - _KB) * cb
    g = (y - _KR * r - _KB * b) / _KG
    return to_uint8(np.stack([r, g, b], -1))


def read_i420(path, ten_bit=False):
    """All frames of an I420 file; 10-bit little-endian (VVdeC) becomes 8-bit as (v + 2) >> 2."""
    if ten_bit:
        data = np.minimum((np.fromfile(path, "<u2").astype(np.uint32) + 2) >> 2, 255).astype(np.uint8).tobytes()
    else:
        data = Path(path).read_bytes()
    if len(data) != FRAMES * FRAME_BYTES:
        raise ValueError(f"{path}: expected {FRAMES} frames")
    return [data[i * FRAME_BYTES:(i + 1) * FRAME_BYTES] for i in range(FRAMES)]


# Sources ----------------------------------------------------------------------------

class _RemoteFile(io.RawIOBase):
    """Seekable read-only view of a URL by HTTP range requests (one zip member at a time)."""

    def __init__(self, url):
        self.url, self.position = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def readinto(self, buffer):
        if self.position >= self.size:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        data = urllib.request.urlopen(request).read()
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


class Shot:
    """The clean frames of one AOM shot archive; member frameNNNNNNNN.j2k is frame start + N."""

    def __init__(self, start, end):
        self.archive = zipfile.ZipFile(io.BufferedReader(_RemoteFile(SHOT_URL.format(start=start, end=end)),
                                                         buffer_size=1 << 20))
        self.prefix = f"HDR/nocturne/{start}-{end}/"

    def frame(self, k):
        """Frame k of the shot as 3840x2160 PQ RGB in [0, 1] (12-bit codes / 4095)."""
        data = self.archive.read(f"{self.prefix}frame{k:08d}.j2k")
        raw = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-f", "j2k_pipe", "-i", "-", "-f", "rawvideo",
                              "-pix_fmt", "rgb48le", "-"], input=data, capture_output=True, check=True).stdout
        raw = np.frombuffer(raw, "<u2").reshape(2160, 3840, 3)
        if not np.all(raw % 16 == 0):     # ffmpeg returns the 12-bit codes shifted left by 4
            raise ValueError("Expected 12-bit codes")
        return (raw >> 4).astype(np.float64) / 4095


def grain_frames(first, count):
    """Master frames first..first+count-1 of the grain version, center crop, PQ RGB in [0, 1].

    ffmpeg crops the 10-bit 4:2:2 frame and converts it with its BT.709
    limited-range tag. It seeks a few frames early; frames are kept by their own
    timestamps, which must cover the range exactly.
    """
    lead = min(3, first)
    convert = (f"crop={WIDTH}:{HEIGHT}:{LEFT}:{TOP},scale=in_color_matrix=bt709:in_range=tv:"
               "flags=accurate_rnd+full_chroma_int+bitexact,format=rgb48le,showinfo")
    result = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-threads", "2", "-ss", f"{(first - lead) / FPS:.6f}",
                             "-copyts", "-i", GRAIN_URL, "-map", "0:v:0", "-frames:v", str(count + lead),
                             "-vsync", "passthrough", "-vf", convert, "-f", "rawvideo", "-"],
                            capture_output=True, check=True)
    raw = np.frombuffer(result.stdout, "<u2").reshape(-1, HEIGHT, WIDTH, 3)
    # showinfo may report one frame more than -frames:v lets through to the output.
    stamps = [round(float(t) * FPS) for t in re.findall(r"pts_time:\s*([0-9.]+)", result.stderr.decode())]
    stamps = stamps[:len(raw)]
    keep = [i for i, index in enumerate(stamps) if first <= index < first + count]
    if len(stamps) != len(raw) or [stamps[i] for i in keep] != list(range(first, first + count)):
        raise ValueError(f"grain version: expected frames {first}..{first + count - 1}, got {stamps}")
    return raw[keep].astype(np.float64) / 65535


def select_shot():
    """The shot whose first frame (center crop, SDR) has median luma closest to 128."""
    medians = []
    for start, end in SHOTS:
        medians.append(float(np.median(luma(tone_map(crop(Shot(start, end).frame(0)))))))
        print("shot %d-%d: median luma %.2f" % (start, end, medians[-1]), flush=True)
    best = min(range(len(SHOTS)), key=lambda i: (abs(medians[i] - 128), SHOTS[i][0]))
    return SHOTS[best], medians


def _mse(a, b):
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


def prepare():
    """Clean and grain uint8 RGB clips, their I420 files and the 10 fitting frames."""
    folder = root("clip")
    if (folder / "grain_fit.yuv").exists():
        return
    (start, end), medians = select_shot()
    shot = Shot(start, end)
    clean, grain = [], []
    for k in range(0, FRAMES, 10):
        clean += [tone_map(crop(shot.frame(k + i))) for i in range(10)]
        grain += [tone_map(frame) for frame in grain_frames(start + k, 10)]
        print("read frames %d-%d" % (start + k, start + k + 9), flush=True)
    clean, grain = np.stack(clean), np.stack(grain)
    # Each grain frame must be closest to the clean frame of the same index.
    for t in range(FRAMES):
        errors = {o: _mse(grain[t], clean[t + o]) for o in (-1, 0, 1) if 0 <= t + o < FRAMES}
        if min(errors, key=errors.get) != 0:
            raise RuntimeError(f"grain frame {t} is not aligned with the clean frame: {errors}")
    config.save_array(folder / "clean.npy", clean)
    config.save_array(folder / "grain.npy", grain)
    for name, frames in (("clean", clean), ("grain", grain)):
        coded = [rgb_to_i420(frame) for frame in frames]
        (folder / f"{name}.yuv").write_bytes(b"".join(coded))
        (folder / f"{name}_fit.yuv").write_bytes(b"".join(coded[t] for t in FIT_FRAMES))
    config.write_json(folder / "shot.json", dict(shot=[start, end], frames=[start, start + FRAMES - 1],
                                                 median_luma={f"{a}-{b}": m for (a, b), m in zip(SHOTS, medians)}))


def clip(name):
    return np.load(root("clip", name + ".npy"), mmap_mode="r")


# Grain models -------------------------------------------------------------------------

def fit(out=None):
    """PGNM on clip frames 0, 10, ..., 90 (RGB / 255), and AV1's table on the same frames."""
    import pgnm
    out = Path(out or root("fits"))
    out.mkdir(parents=True, exist_ok=True)
    clean, grain = clip("clean"), clip("grain")

    def pairs():
        for t in FIT_FRAMES:
            yield np.asarray(clean[t], np.float32) / 255, np.asarray(grain[t], np.float32) / 255
    if not (out / "pgnm.json").exists():
        pgnm.save(out / "pgnm.json", pgnm.fit(pairs, progress=print))
    if not (out / "av1_grain.tbl").exists():
        # fps 6: the ten fitting frames are 1/6 s apart in the clip.
        subprocess.run([str(NOISE_MODEL), f"--input={root('clip/grain_fit.yuv')}",
                        f"--input-denoised={root('clip/clean_fit.yuv')}", f"--output-grain-table={out / 'av1_grain.tbl'}",
                        f"--width={WIDTH}", f"--height={HEIGHT}", "--fps=6/1", "--i420"], check=True)


def model_bytes(params):
    """PGNM's side information: zlib (level 9) of the float64 kernel and lookup arrays."""
    kernels = np.asarray(params["kernels"], np.float64)
    lookup = np.asarray(params["marginal"]["g"], np.float64)
    return len(zlib.compress(kernels.tobytes() + lookup.tobytes(), 9))


# Encoding and decoding -----------------------------------------------------------------

def bitstream(key):
    arm, rate = key.split("/")
    return root("bitstreams", arm, rate + (".266" if ARMS[arm][0] == "vvc" else ".ivf"))


def encode_command(key, presets):
    arm, rate = key.split("/")
    codec, source, grain, options = ARMS[arm]
    source = root("clip", source + ".yuv")
    if codec == "vvc":
        rate_options = ("--TargetBitrate", rate, "--Passes", "2") if rate.endswith("k") else ("-q", rate)
        return [str(VVENC), "-i", str(source), *VVENC_OPTIONS, *rate_options, *options, "-b", str(bitstream(key))]
    table = (f"--film-grain-table={Path(presets) / 'av1_grain.tbl'}",) if grain == "av1" else ()
    return [str(AOMENC), *AOMENC_OPTIONS, "--end-usage=q", f"--cq-level={rate}", *table, *options,
            "-o", str(bitstream(key)), str(source)]


def encode(keys, presets):
    for key in keys:
        target = bitstream(key)
        if target.with_suffix(".json").exists():    # written once the encoder has finished
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        command = encode_command(key, presets)
        subprocess.run(command, check=True, capture_output=True)
        size = target.stat().st_size
        # IVF: a 32-byte file header and a 12-byte header per frame; VVC: Annex-B stream.
        payload = size if ARMS[key.split("/")[0]][0] == "vvc" else size - 32 - 12 * FRAMES
        config.write_json(target.with_suffix(".json"), dict(command=command, bytes=size, payload_bytes=payload,
                                                            kbps=payload * 8 / (FRAMES / FPS) / 1000))
        print("encoded", key, flush=True)


def decode_command(key, output):
    """Decoder call; codec grain is applied only where the arm synthesizes it."""
    arm = key.split("/")[0]
    codec, _, grain, _ = ARMS[arm]
    if codec == "vvc":
        return [str(VVDEC), "-b", str(bitstream(key)), "-o", str(output), "-t", "2", "-v", "1",
                "-fg", "1" if grain == "vvc" else "0"]
    skip = () if grain == "av1" else ("--skip-film-grain",)
    return [str(AOMDEC), *skip, "--rawvideo", "--i420", "-o", str(output), str(bitstream(key))]


def delivered_path(key):
    return root("delivered", key + ".npy")


def decode(keys, presets):
    """Delivered uint8 RGB frames of every output, with grain where the arm adds it."""
    from pgnm import PGNM, load
    root().mkdir(parents=True, exist_ok=True)
    for key in keys:
        target = delivered_path(key)
        if target.exists():
            continue
        with tempfile.TemporaryDirectory(dir=root()) as tmp:
            decoded = Path(tmp) / "decoded.yuv"
            subprocess.run(decode_command(key, decoded), check=True, capture_output=True)
            frames = read_i420(decoded, ten_bit=ARMS[key.split("/")[0]][0] == "vvc")
        delivered = np.empty((FRAMES, HEIGHT, WIDTH, 3), np.uint8)
        if ARMS[key.split("/")[0]][2] == "pgnm":
            model = PGNM(load(Path(presets) / "pgnm.json"), workers=2, bank_seeds=bank_seeds(0))
            for t, frame in enumerate(frames):   # in order: each frame avoids the previous bank field
                clean = i420_to_rgb(frame).astype(np.float32) / 255
                delivered[t] = to_uint8(model.render(clean, np.random.default_rng(WINDOW_SEED + t)))
        else:
            for t, frame in enumerate(frames):
                delivered[t] = i420_to_rgb(frame)
        config.save_array(target, delivered)
        print("decoded", key, flush=True)


# Scores --------------------------------------------------------------------------------

def _pending(keys, folder):
    return [key for key in keys if not root(folder, key + ".json").exists() and delivered_path(key).exists()]


def score(keys):
    """Grain fidelity of every pending output on all frames, frame by frame."""
    from metrics import residual
    pending = _pending(keys, "scores")
    if not pending:
        return
    clean, grain = clip("clean"), clip("grain")
    delivered = {key: np.load(delivered_path(key), mmap_mode="r") for key in pending}
    frames = {key: {} for key in pending}
    for t in range(FRAMES):
        state = residual.prepare_rgb(clean[t], grain[t])
        for key in pending:
            frames[key][t] = residual.compare_rgb(state, np.asarray(delivered[key][t]))
        print("scored frame", t, flush=True)
    for key in pending:
        config.write_json(root("scores", key + ".json"), dict(frames=frames[key]))


class _Features:
    """Reuse a backbone's features for an input seen recently (the reference frame).

    DISTS gets the same tensor object again; LPIPS rescales its input into a new
    tensor on every call, so it is recognized by its bytes. Distances are untouched.
    """

    def __init__(self, function, by_content=False):
        self.function, self.by_content, self.cache = function, by_content, []

    def __call__(self, tensor):
        key = tensor.detach().cpu().numpy().tobytes() if self.by_content else tensor
        for index, (seen, features) in enumerate(self.cache):
            if (seen == key) if self.by_content else (seen is key):
                self.cache.append(self.cache.pop(index))
                return features
        features = self.function(tensor)
        self.cache = self.cache[-1:] + [(key, features)]
        return features


def appearance(keys, device="cpu"):
    """DISTS and LPIPS (AlexNet), GrainNet's definitions, of RGB frames in [0, 1] against the grain version."""
    import torch
    import lpips
    from DISTS_pytorch import DISTS
    pending = _pending(keys, "appearance")
    if not pending:
        return
    torch.set_num_threads(2)
    dists, perceptual = DISTS().eval().to(device), lpips.LPIPS(net="alex").to(device)
    dists.forward_once = _Features(dists.forward_once)
    perceptual.net.forward = _Features(perceptual.net.forward, by_content=True)

    def tensor(image):
        return torch.tensor(np.asarray(image, np.float32) / 255).permute(2, 0, 1)[None].contiguous().to(device)
    grain = clip("grain")
    delivered = {key: np.load(delivered_path(key), mmap_mode="r") for key in pending}
    frames = {key: {} for key in pending}
    for t in APPEARANCE_FRAMES:
        reference = tensor(grain[t])
        for key in pending:
            generated = tensor(delivered[key][t])
            with torch.no_grad():
                frames[key][t] = dict(dists=float(dists(reference, generated)),
                                      lpips=float(perceptual(reference, generated).item()))
        print("appearance frame", t, flush=True)
    for key in pending:
        config.write_json(root("appearance", key + ".json"), dict(frames=frames[key], device=device))


# Tab. 5 ----------------------------------------------------------------------------------

def summary(key, presets):
    """Total rate and frame means of one output."""
    arm = key.split("/")[0]
    kbps = config.read_json(bitstream(key).with_suffix(".json"))["kbps"]
    if ARMS[arm][2] == "pgnm":
        kbps += model_bytes(config.read_json(Path(presets) / "pgnm.json")) * 8 / 1000 / (FRAMES / FPS)
    fidelity = config.read_json(root("scores", key + ".json"))["frames"]
    looks = config.read_json(root("appearance", key + ".json"))["frames"]
    row = dict(kbps=kbps)
    for metric in ("conditional_w1", "spatial_correlation_error", "color_correlation_error"):
        row[metric] = float(np.mean([fidelity[str(t)][metric] for t in range(FRAMES)]))
    for metric in ("dists", "lpips"):
        row[metric] = float(np.mean([looks[str(t)][metric] for t in APPEARANCE_FRAMES]))
    return row


def table_rows(rows):
    """Tab. 5 from per-output summaries; each coded row keeps its lower-DISTS setting."""
    result = []
    for encoder, grain, keys in TABLE:
        key = min(keys, key=lambda k: rows[k]["dists"])
        result.append(dict(rows[key], encoder=encoder, grain=grain, output=key))
    return result


def _matched_rate(points, target):
    """Rate at which a coded arm reaches ``target`` (lower is better): linear in log-rate
    between adjacent ladder points, first crossing from the lowest rate."""
    points = sorted(points)
    if points[0][1] <= target:
        return None, "coded arm better at every rate"
    for (r0, m0), (r1, m1) in zip(points, points[1:]):
        if m1 <= target:
            fraction = (target - m0) / (m1 - m0)
            return float(np.exp(np.log(r0) + fraction * (np.log(r1) - np.log(r0)))), "matched"
    return float(points[-1][0]), "lower bound"     # never reached: the highest rate bounds the saving


def matched_savings(rows, metric="dists"):
    """Per PGNM quantizer, the smallest saving at equal ``metric`` over the four coded arms."""
    savings = {}
    for q in LADDERS["av1"]:
        pgnm = rows[f"pgnm/{q}"]
        candidates = []
        for arm in CODED:
            points = [(rows[f"{arm}/{r}"]["kbps"], rows[f"{arm}/{r}"][metric]) for r in LADDERS[ARMS[arm][0]]]
            rate, kind = _matched_rate(points, pgnm[metric])
            candidates.append((None if rate is None else 1 - pgnm["kbps"] / rate, arm, kind))
        savings[q] = min(candidates, key=lambda c: -np.inf if c[0] is None else c[0])   # None: no saving
    return savings


def table(presets, ladder=False):
    rows = {key: summary(key, presets) for key in outputs(ladder)}
    result = dict(rows=rows, table=table_rows(rows))
    if ladder:
        result["dists_matched_saving"] = {q: dict(saving=s, arm=a, kind=k)
                                          for q, (s, a, k) in matched_savings(rows).items()}
    config.write_json(root("table.json"), result)
    return result


def print_table(result):
    print("%-12s %-7s %7s %7s %7s %6s %8s" % ("Encoder", "Grain", "Mbps", "DISTS", "LPIPS", "W1", "Spatial"))
    for row in result["table"]:
        mbps = row["kbps"] / 1000
        print("%-12s %-7s %7s %7.3f %7.3f %6.2f %8.3f" % (
            row["encoder"], row["grain"], "%.1f" % mbps if mbps > 10 else "%.2f" % mbps, row["dists"],
            row["lpips"], row["conditional_w1"], row["spatial_correlation_error"]))
    if "dists_matched_saving" in result:
        values = [entry["saving"] for entry in result["dists_matched_saving"].values()]
        if None in values:
            print("A coded arm reaches PGNM's DISTS at a lower rate for some quantizer; see table.json")
        else:
            print("PGNM saving at matched DISTS, smallest over the coded arms: %.1f-%.1f%%"
                  % (100 * min(values), 100 * max(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "fit", "encode", "decode", "score", "appearance", "table"))
    parser.add_argument("outputs", nargs="*", help="restrict to these outputs, e.g. pgnm/55 (default: all)")
    parser.add_argument("--ladder", action="store_true", help="also every rate of the coded arms and PGNM")
    parser.add_argument("--presets", type=Path, default=config.REPO / "presets/bandwidth",
                        help="folder with pgnm.json and av1_grain.tbl (default: presets/bandwidth)")
    parser.add_argument("--device", default="cpu", help="torch device for `appearance` (default: cpu)")
    args = parser.parse_args()
    keys = args.outputs or outputs(args.ladder)
    unknown = [key for key in keys if key.split("/")[0] not in ARMS or "/" not in key]
    if unknown:
        parser.error("unknown output: " + ", ".join(unknown))
    if args.stage == "prepare":
        prepare()
    elif args.stage == "fit":
        fit()
    elif args.stage == "encode":
        encode(keys, args.presets)
    elif args.stage == "decode":
        decode(keys, args.presets)
    elif args.stage == "score":
        score(keys)
    elif args.stage == "appearance":
        appearance(keys, args.device)
    else:
        print_table(table(args.presets, args.ladder))


if __name__ == "__main__":
    main()
