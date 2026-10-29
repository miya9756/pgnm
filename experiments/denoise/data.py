"""Training images, patches and random streams of the denoising experiment (Tab. 2).

The training set is NM-FlowGAN's denoiser split: 123 SIDD-Medium scene
instances with both captures, 246 sRGB pairs, which are all Medium instances of
the 15 camera-ISO classes NM-FlowGAN supports. ``prepare_store`` decodes them
once into two uint8 files (clean.u8 and noisy.u8, about 10 GB each) that are read
as memory maps.

Every image is cut into 96x96 patches at stride 48 on NM-FlowGAN's grid, which
gives 1,473,956 patches per epoch. A training batch is a pure function of
(seed, epoch, batch): the epoch permutation and the rotation and flip of each
patch come from hashed per-patch seeds, so the number of data workers and
resuming a run never change the sample stream, and every training-noise source
sees the same patches in the same order with the same augmentation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .. import config

# TRAIN_DENOISER_INDICES of NM-FlowGAN's prep.py (inherited from Noise2NoiseFlow).
TRAIN_INDICES = (
    1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 27, 28,
    29, 30, 32, 33, 34, 35, 38, 39, 40, 42, 43, 44, 45, 47, 48, 51, 52, 54, 55, 57, 59, 60,
    62, 63, 66, 75, 77, 81, 86, 87, 88, 90, 94, 98, 101, 102, 104, 105, 110, 111, 113, 114,
    115, 116, 117, 118, 122, 125, 126, 127, 129, 132, 133, 134, 135, 136, 137, 138, 140, 142,
    147, 149, 150, 151, 152, 154, 155, 156, 159, 160, 161, 163, 164, 165, 166, 167, 169, 172,
    175, 177, 178, 179, 180, 181, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196,
    197, 198, 199,
)
CLASSES = (
    "G4_100", "G4_800", "GP_100", "GP_800", "IP_100", "IP_400", "IP_800", "IP_1600",
    "N6_100", "N6_800", "S6_100", "S6_400", "S6_800", "S6_1600", "S6_3200",
)
CAPTURES = ("010", "011")
PATCH, STRIDE = 96, 48
PATCHES = 1_473_956

# Literal prefix of every random stream of this experiment.
STREAM = "CVPR-DENOISE/1"


def seed(*parts):
    """63-bit seed hashed from printable parts, e.g. seed("augment", 1, epoch, position)."""
    payload = "|".join(map(str, (STREAM,) + parts))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") >> 1


def rng(*parts):
    return np.random.default_rng(seed(*parts))


def class_name(instance):
    """'0001_001_S6_00100_00060_3200_L' -> 'S6_100'."""
    fields = instance.split("_")
    return "%s_%d" % (fields[2], int(fields[3]))


def training_images():
    """The 246 training pairs in store order, with class and PNG paths."""
    data = config.sidd_medium() / "Data"
    instances = sorted(p.name for p in data.iterdir() if p.is_dir())
    chosen = [i for i in instances if int(i.split("_")[0]) in TRAIN_INDICES]
    # The split is exactly the Medium instances of the 15 classes.
    if len(chosen) != len(TRAIN_INDICES) or chosen != [i for i in instances if class_name(i) in CLASSES]:
        raise RuntimeError("SIDD-Medium copy does not give NM-FlowGAN's denoiser split")
    return [dict(instance=instance, capture=capture, class_name=class_name(instance),
                 clean=data / instance / f"{instance[:4]}_GT_SRGB_{capture}.PNG",
                 noisy=data / instance / f"{instance[:4]}_NOISY_SRGB_{capture}.PNG")
            for instance in chosen for capture in CAPTURES]


def read_rgb(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(image[..., ::-1])


# Patch grid and store ------------------------------------------------------------

def grid_starts(size):
    """NM-FlowGAN's crop loop: 0, 48, ... below ``size``, clamped to size - 96.

    Starts past the last full patch repeat the edge patch; those repeats are part
    of the epoch, as upstream.
    """
    return [min(start, size - PATCH) for start in range(0, size, STRIDE)]


def patch_grid(shapes):
    """int32 rows (image, top, left) of every patch of images with these (height, width)."""
    rows = []
    for image, (height, width) in enumerate(shapes):
        tops, lefts = grid_starts(height), grid_starts(width)
        rows.append(np.stack([np.full(len(tops) * len(lefts), image), np.repeat(tops, len(lefts)),
                              np.tile(lefts, len(tops))], 1))
    return np.concatenate(rows).astype(np.int32)


def store_path(path=None):
    return Path(path or config.runs("denoise/store"))


def prepare_store(path=None):
    """Decode the 246 training pairs into clean.u8 and noisy.u8 (RGB, row-major)."""
    store = store_path(path)
    if (store / "index.json").exists():
        return
    store.mkdir(parents=True, exist_ok=True)
    images, offset = [], 0
    with open(store / "clean.u8.tmp", "wb") as clean_file, open(store / "noisy.u8.tmp", "wb") as noisy_file:
        for number, image in enumerate(training_images()):
            clean, noisy = read_rgb(image["clean"]), read_rgb(image["noisy"])
            if clean.shape != noisy.shape:
                raise RuntimeError("Clean/noisy shape mismatch: " + image["instance"])
            clean_file.write(clean.tobytes())
            noisy_file.write(noisy.tobytes())
            images.append(dict(instance=image["instance"], capture=image["capture"],
                               class_name=image["class_name"], shape=list(clean.shape[:2]), offset=offset))
            offset += clean.size
            print("stored", number + 1, "of 246", image["instance"], image["capture"], flush=True)
    if len(patch_grid([image["shape"] for image in images])) != PATCHES:
        raise RuntimeError("Unexpected number of training patches")
    for name in ("clean", "noisy"):
        (store / (name + ".u8.tmp")).replace(store / (name + ".u8"))
    config.write_json(store / "index.json", dict(images=images, bytes=offset))


class Store:
    """Read-only view of the decoded training pairs; memory maps open lazily per worker."""

    def __init__(self, path=None):
        self.path = store_path(path)
        index = json.loads((self.path / "index.json").read_text())
        for name in ("clean", "noisy"):
            if (self.path / (name + ".u8")).stat().st_size != index["bytes"]:
                raise RuntimeError("Store file has the wrong size: " + name)
        images = index["images"]
        self.shapes = [tuple(image["shape"]) for image in images]
        self.offsets = np.array([image["offset"] for image in images], np.int64)
        self.widths = np.array([shape[1] for shape in self.shapes], np.int64)
        self.class_index = np.array([CLASSES.index(image["class_name"]) for image in images])
        self.grid = patch_grid(self.shapes)
        self._maps = None

    def maps(self):
        if self._maps is None:
            self._maps = tuple(np.memmap(self.path / (name + ".u8"), np.uint8, "r") for name in ("clean", "noisy"))
        return self._maps

    def patch(self, row):
        """(clean, noisy) uint8 96x96x3 patches of one grid row."""
        image, top, left = (int(value) for value in row)
        width = int(self.widths[image])
        start = int(self.offsets[image]) + (top * width + left) * 3
        return [np.array(np.lib.stride_tricks.as_strided(
                    array[start:], shape=(PATCH, PATCH * 3), strides=(width * 3, 1))).reshape(PATCH, PATCH, 3)
                for array in self.maps()]


def augment(image, rotation, flip):
    image = np.rot90(image, rotation, axes=(0, 1))
    return np.ascontiguousarray(image[:, ::-1] if flip else image)


class EpochBatches:
    """Map-style dataset whose item i is batch ``start + i`` of one epoch.

    Each patch gets a uniform rotation by k*90 degrees and a horizontal flip,
    applied to the clean patch and its real noisy patch before any noise is
    synthesized. ``cpu_noise`` (PGNM) renders here, in the data workers.
    """

    def __init__(self, store, seed, epoch, batch_size, start=0, limit=None, cpu_noise=None):
        self.store, self.seed, self.epoch = store, int(seed), int(epoch)
        self.batch_size, self.start, self.cpu_noise = int(batch_size), int(start), cpu_noise
        self.permutation = rng("permutation", self.seed, self.epoch).permutation(len(store.grid))
        self.batches = -(-len(store.grid) // self.batch_size)
        if limit is not None:
            self.batches = min(self.batches, int(limit))

    def __len__(self):
        return self.batches - self.start

    def __getitem__(self, item):
        number = self.start + int(item)
        if not self.start <= number < self.batches:
            raise IndexError(item)
        clean, real, classes, synthetic = [], [], [], []
        for position in range(number * self.batch_size,
                              min((number + 1) * self.batch_size, len(self.permutation))):
            row = self.store.grid[self.permutation[position]]
            draw = rng("augment", self.seed, self.epoch, position)
            rotation, flip = int(draw.integers(4)), int(draw.integers(2))
            pair = [augment(image, rotation, flip) for image in self.store.patch(row)]
            index = self.store.class_index[row[0]]
            clean.append(pair[0])
            real.append(pair[1])
            classes.append(index)
            if self.cpu_noise is not None:
                synthetic.append(self.cpu_noise.render(pair[0], CLASSES[index], (self.seed, self.epoch, position)))
        batch = dict(clean=np.stack(clean), real=np.stack(real), classes=np.asarray(classes, np.int64),
                     batch=number)
        if synthetic:
            batch["synthetic"] = np.stack(synthetic)
        return batch


# PGNM fitting crops -------------------------------------------------------------

def crop_origins(size, crop=256):
    """A 256-pixel grid plus one last crop flush with the far edge (as experiments.sidd)."""
    origins = list(range(0, size - crop + 1, crop))
    if size - (origins[-1] + crop) > 0:
        origins.append(size - crop)
    return origins


def fit_pairs(name):
    """Callable yielding the (clean, noisy) float32 256x256 crops of one class's training images.

    Crops follow image order, then row-major grid order. The crop list used for
    the paper's presets named every crop twice, so every crop is passed twice in a
    row; this weights all crops equally and reproduces those presets exactly.
    """
    images = [image for image in training_images() if image["class_name"] == name]

    def pairs():
        for image in images:
            clean, noisy = read_rgb(image["clean"]), read_rgb(image["noisy"])
            for y in crop_origins(clean.shape[0]):
                for x in crop_origins(clean.shape[1]):
                    crop = tuple(a[y:y + 256, x:x + 256].astype(np.float32) / 255 for a in (clean, noisy))
                    yield crop
                    yield crop
    return pairs
