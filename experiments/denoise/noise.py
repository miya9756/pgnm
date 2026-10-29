"""Training-noise sources of Tab. 2: clean patch + camera-ISO class -> noisy patch.

    real        the patch's own real noisy observation (reference)
    pgnm        PGNM, one preset per class fitted on the same 246 training images
    gmdcn       GMDCN, the SIDD checkpoint of Tab. 1, with its camera-ISO condition map
    naflow      NAFlow, class-conditional sampling of the released checkpoint
    naflow_nas  NAFlow-NAS: noise-aware sampling, which encodes the patch's real
                noisy observation, weights all class Gaussians by likelihood and
                samples from the mixture
    nmflowgan   NM-FlowGAN, released sampler with its camera and ISO inputs

Real and PGNM run on the CPU in the data workers on uint8 patches. The neural
generators run batched on the training GPU in float32 (TF32 off) on inputs
code/255 and return floats in [0, 1]. Every noisy patch, real or synthetic, is
delivered by the same rule: clip to [0, 1] and round half up to 8 bits. All
generators use their native settings. Upstream code and weights come from
$PGNM_EXTERNAL, loaded as in baselines/adapters.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .. import config
from ..seeds import to_uint8
from .data import CLASSES, fit_pairs, rng

SOURCES = dict(real="Real noise (reference)", gmdcn="GMDCN", nmflowgan="NM-FlowGAN",
               naflow="NAFlow", naflow_nas="NAFlow-NAS", pgnm="PGNM (ours)")


def split_class(name):
    camera, iso = name.split("_")
    return camera, int(iso)


class RealNoise:
    where = "cpu"


class PGNMNoise:
    """PGNM's convolution and residual lookup with a fresh latent field for every patch.

    The renderer's four-field bank is a device for video-rate rendering; across
    44 million training patches it would repeat four small fields. Each patch
    instead draws an independent field of exactly its size from the seed of its
    (seed, epoch, position); the field and the lookup are the renderer's own
    steps (PGNM.render without the bank window).
    """
    where = "cpu"

    def __init__(self, presets=None):
        self.presets = Path(presets or config.REPO / "presets/denoise")
        missing = [name for name in CLASSES if not (self.presets / (name + ".json")).is_file()]
        if missing:
            raise FileNotFoundError("Missing PGNM presets in %s: %s" % (self.presets, ", ".join(missing)))
        self._models = {}

    def model(self, name):
        if name not in self._models:  # built lazily, once per worker process
            from pgnm import PGNM, load
            self._models[name] = PGNM(load(self.presets / (name + ".json")), workers=1)
        return self._models[name]

    def render(self, clean, name, key):
        """uint8 (96, 96, 3) clean patch -> uint8 noisy patch."""
        model = self.model(name)
        field = model._build_noise_field(clean.shape, rng("pgnm-field", *key)).astype(np.float32)
        image = np.ascontiguousarray(clean, np.float32) / np.float32(255)
        noisy = np.empty_like(image)
        model._render_noise_window(field, image, 0, 0, noisy)
        return to_uint8(noisy)


class GMDCNNoise:
    where = "device"

    def __init__(self, device):
        import torch
        from baselines.adapters import upstream
        root = upstream("gmdcn")
        from noise_synthesizer.config import NoiseSynthExperimentConfig, build_generator
        from noise_synthesizer.generate import _build_condition_map

        generator = build_generator(NoiseSynthExperimentConfig.from_yaml(
            os.path.join(root, "configs", "gmdcn.yaml")).generator)
        generator.load_state_dict(torch.load(os.path.join(root, "models_zoo", "sidd_gmdcn.pth"),
                                             map_location="cpu"), strict=True)
        self.generator = generator.to(device).eval()
        self.condition_map, self.device = _build_condition_map, device

    def __call__(self, clean, real, names):
        import torch
        height, width = clean.shape[2:]
        maps = torch.cat([self.condition_map(*split_class(name), height, width, self.device) for name in names])
        with torch.no_grad():
            return (clean + self.generator(torch.cat([clean, maps], 1))).clamp(0, 1)


class NAFlowNoise:
    where = "device"

    def __init__(self, device, noise_aware=False):
        import torch
        from baselines.adapters import upstream
        root = upstream("naflow", "naflow")
        from omegaconf import OmegaConf
        from archs.naflow import NAFlow

        model_config = OmegaConf.load(os.path.join(root, "configs", "models", "naflow.yaml"))
        model_config = (model_config.params if "params" in model_config else model_config).network_G_config
        net = NAFlow(model_config)
        state = torch.load(str(config.external("naflow", "weights", "naflow_sidd.ckpt")),
                           map_location="cpu")["state_dict"]
        net.load_state_dict({k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()},
                            strict=True)
        self.net = net.to(device).eval()
        self.jitter_std = float(model_config["add_cond_noise_std"]) / 255.0  # 3/255
        self.noise_aware = bool(noise_aware)

    def jitter(self, clean):
        import torch
        return (clean + torch.randn_like(clean) * self.jitter_std).clamp(0, 1)

    def noise_aware_sampling(self, z):
        """NAFlow's noise_aware_sampling evaluated separately for every patch of the batch.

        Softmax over the per-pixel log-likelihoods of all 34 class Gaussians
        (temperature 1), then a draw from the Gaussian with the probability-weighted
        class means and standard deviations.
        """
        import torch
        eps = torch.randn_like(z)
        names = list(self.net.mean_shifts.keys())
        means = torch.stack([self.net.mean_shifts[name] for name in names])
        stds = torch.stack([self.net.cov_shifts[name] for name in names])
        log_likelihood = torch.stack([self.net.logp(self.net.mean_shifts[name], self.net.cov_shifts[name].pow(2), z)
                                      for name in names])
        weights = torch.softmax(log_likelihood / (z.shape[2] * z.shape[3]), dim=0)
        mean = torch.einsum("kb,kc->bc", weights, means)
        std = torch.einsum("kb,kc->bc", weights, stds)
        return std[:, :, None, None] * eps + mean[:, :, None, None]

    def __call__(self, clean, real, names):
        import torch
        labels = ["%s_%05d" % split_class(name) for name in names]
        with torch.no_grad():
            if self.noise_aware:
                condition = self.jitter(clean)
                z, _, _ = self.net(input=real, cond=condition, reverse=False, label=labels)
                z = self.noise_aware_sampling(z)
            else:  # latent first, then the condition jitter
                eps = torch.randn(len(labels), self.net.flowUpsamplerNet.C, clean.shape[2] // 8,
                                  clean.shape[3] // 8, device=clean.device)
                mean = torch.stack([self.net.mean_shifts[label] for label in labels])
                std = torch.stack([self.net.cov_shifts[label] for label in labels])
                z = std[:, :, None, None] * eps + mean[:, :, None, None]
                condition = self.jitter(clean)
            output, _ = self.net(cond=condition, z=z, eps_std=1.0, reverse=True, label=labels)
        return output.clamp(0, 1)


class NMFlowGANNoise:
    """Codes 0..255 in, sample clipped to [0, 256] out; sample/255 reproduces upstream's floor(sample + 0.5)."""
    where = "device"

    def __init__(self, device):
        import torch
        import yaml
        from baselines.adapters import upstream
        from baselines.adapters.nmflowgan import CAMERAS
        root = upstream("nmflowgan")
        from core.model.NMFlowGAN import NMFlowGANGenerator

        with open(os.path.join(root, "conf", "NMFlowGAN_NoiseModeler.yaml")) as handle:
            model_config = yaml.safe_load(handle)
        model = NMFlowGANGenerator(kwargs_unet=model_config["MODEL"]["UNET"],
                                   kwargs_flow=model_config["MODEL"]["NMFLOW"])
        checkpoint = torch.load(str(config.external("nmflowgan", "NMFlowGAN_NoiseModeler.pth")),
                                map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_weight"]["generator"], strict=True)
        self.model, self.cameras = model.to(device).eval(), CAMERAS

    def __call__(self, clean, real, names):
        import torch
        labels = [split_class(name) for name in names]
        inputs = {
            "clean": torch.floor(clean * 255 + 0.5),
            "ISO-level": torch.tensor([float(iso) for _, iso in labels], device=clean.device),
            "smartphone-code": torch.tensor([float(self.cameras.index(camera)) for camera, _ in labels],
                                            device=clean.device),
        }
        with torch.no_grad():
            return self.model.sample(inputs).double() / 255.0


def build(source, device=None, presets=None):
    if source == "real":
        return RealNoise()
    if source == "pgnm":
        return PGNMNoise(presets)
    if source == "gmdcn":
        return GMDCNNoise(device)
    if source in ("naflow", "naflow_nas"):
        return NAFlowNoise(device, noise_aware=source == "naflow_nas")
    if source == "nmflowgan":
        return NMFlowGANNoise(device)
    raise ValueError("Unknown training-noise source " + repr(source))


# 8-bit conversions on the GPU ----------------------------------------------------

_UNIT = {}


def unit(codes):
    """uint8 codes -> float32 code/255, rounded exactly as NumPy rounds it.

    CUDA divides by a scalar through its reciprocal, which differs by one ulp for
    about a third of the codes; a lookup table keeps the inputs identical to the
    adapters, which divide in NumPy.
    """
    import torch
    table = _UNIT.get(codes.device)
    if table is None:
        table = _UNIT[codes.device] = torch.from_numpy(np.arange(256, dtype=np.float32) / np.float32(255)).to(codes.device)
    return table[codes.long()]


def quantize(image):
    """experiments.seeds.to_uint8 on (B, 3, H, W) torch images in [0, 1]."""
    import torch
    return torch.floor(image.double().clamp(0, 1) * 255 + 0.5).to(torch.uint8)


# PGNM presets ------------------------------------------------------------------

def fit(classes=None, out=None):
    """Fit one PGNM preset per class on every 256x256 crop of its training images (CPU)."""
    from pgnm import fit as fit_pgnm, validate
    out = Path(out or config.runs("denoise/presets"))
    for name in classes or CLASSES:
        target = out / (name + ".json")
        if target.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        params = validate(fit_pgnm(fit_pairs(name), progress=lambda message: print(name, message, flush=True)))
        # Same layout as presets/denoise, so a refit can be compared byte for byte.
        target.write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
        print("fitted", name, flush=True)
