"""Train one DnCNN on one training-noise source (Tab. 2).

    python -m experiments.denoise train --source pgnm --workers 8

The recipe is NM-FlowGAN's denoiser configuration: DnCNN with 18 3x3
convolutions of 64 features, batch normalization (momentum 0.9, eps 1e-4) on
the 16 middle layers and a residual output; input noisy code/256, output times
256, mean squared error against the clean code; Adam (1e-4, betas 0.9/0.999)
halving its learning rate after every fifth epoch; 30 epochs of 11,516 batches
of 128 patches. Initialization, patch order and augmentation depend on the
training seed only, so all sources share them; GPU noise draws are seeded per
(source, seed, epoch, batch). Everything runs in float32 with TF32 disabled.

Writes runs/denoise/<source>/model.pt (the epoch-30 weights, the reported
model), epoch checkpoints, and last.pt, from which an interrupted run resumes.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import config
from .data import CLASSES, EpochBatches, Store, seed as stream_seed
from .noise import build, quantize, unit

SEED, EPOCHS, BATCH = 1, 30, 128
LEARNING_RATE, BETAS, HALVE_EVERY = 1e-4, (0.9, 0.999), 5
LEVELS = 256.0  # NM-FlowGAN's 8-bit scaling of DnCNN inputs and outputs
LOG_EVERY = 100


class DnCNN(nn.Module):
    """Layer order, batch-norm settings and default initialization of NM-FlowGAN's DnCNN."""

    def __init__(self, channels=3, num_of_layers=18, features=64):
        super().__init__()
        layers = [nn.Conv2d(channels, features, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=True),
                       nn.BatchNorm2d(features, momentum=0.9, eps=1e-4, affine=True),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)


def learning_rate(epoch):
    return LEARNING_RATE * 0.5 ** ((epoch - 1) // HALVE_EVERY)


def full_float32():
    """TF32 would change generator outputs by about 1e-4; cuDNN autotuning is kept."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


def noisy_batch(source, noise, batch, seed, epoch, device):
    """(clean, noisy) float code images (B, 3, 96, 96) on the device."""
    def nchw(key):
        return batch[key].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()

    clean_u8, real_u8 = nchw("clean"), nchw("real")
    if source == "real":
        noisy = real_u8.float()
    elif noise.where == "cpu":
        noisy = nchw("synthetic").float()
    else:
        torch.manual_seed(stream_seed("renoise", source, seed, epoch, int(batch["batch"])))
        names = [CLASSES[i] for i in batch["classes"].tolist()]
        noisy = quantize(noise(unit(clean_u8), unit(real_u8), names)).float()
    return clean_u8.float(), noisy


def training_step(model, optimizer, clean, noisy):
    loss = F.mse_loss(model(noisy / LEVELS) * LEVELS, clean)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _one_thread(worker):
    torch.set_num_threads(1)


def train(source, seed=SEED, workers=8, device="cuda", presets=None, store=None, out=None,
          epochs=EPOCHS, batch_size=BATCH, batches_per_epoch=None):
    """Train (or resume) one run. ``batches_per_epoch`` shortens epochs for a quick test."""
    full_float32()
    device = torch.device(device)
    out = Path(out or config.runs("denoise", source))
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        print("already trained:", out / "model.pt")
        return
    store = Store(store)
    noise = build(source, device, presets)
    torch.manual_seed(stream_seed("init", seed))
    model = DnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS)
    epoch, next_batch = 1, 0
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        if (state["source"], state["seed"]) != (source, seed):
            raise RuntimeError("last.pt belongs to another run: " + str(out))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, next_batch = state["epoch"], state["next_batch"]

    def save(epoch, next_batch):
        state = dict(source=source, seed=seed, epoch=epoch, next_batch=next_batch,
                     model=model.state_dict(), optimizer=optimizer.state_dict())
        torch.save(state, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", out / "last.pt")

    while epoch <= epochs:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch)
        dataset = EpochBatches(store, seed, epoch, batch_size, start=next_batch, limit=batches_per_epoch,
                               cpu_noise=noise if source == "pgnm" else None)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda", worker_init_fn=_one_thread,
                            prefetch_factor=4 if workers else None)
        model.train()
        losses, tick, saved = [], time.monotonic(), time.monotonic()
        for batch in loader:
            clean, noisy = noisy_batch(source, noise, batch, seed, epoch, device)
            losses.append(training_step(model, optimizer, clean, noisy))
            number = int(batch["batch"])
            if (number + 1) % LOG_EVERY == 0 or number + 1 == dataset.batches:
                loss = torch.stack(losses).double().mean().item()
                if not np.isfinite(loss):
                    raise RuntimeError("Training diverged in epoch %d, batch %d" % (epoch, number))
                print("epoch %d batch %d/%d loss %.4f %.2f it/s" % (
                    epoch, number + 1, dataset.batches, loss, len(losses) / (time.monotonic() - tick)), flush=True)
                losses, tick = [], time.monotonic()
            if time.monotonic() - saved > 1800:  # resumable every 30 minutes
                save(epoch, number + 1)
                saved = time.monotonic()
        torch.save(model.state_dict(), out / "checkpoints" / ("epoch%02d.pt" % epoch))
        epoch, next_batch = epoch + 1, 0
        save(epoch, next_batch)
    torch.save(model.state_dict(), out / "model.pt")
    print("trained", out / "model.pt", flush=True)
