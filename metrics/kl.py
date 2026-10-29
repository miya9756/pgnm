"""Residual-histogram KL divergence used by NM-FlowGAN and sRGB-Flow.

Signed residuals in 8-bit code units (output - clean) are counted in 130 bins of
width 4 between -260 and 260, one count is added to every bin, and
KL(real || generated) is taken in nats over the pooled channels and pixels.
The paper reports the mean over all 64 nonoverlapping 32x32 tiles of each
256x256 Validation block, then over blocks; the whole-block value is also kept.
"""
from __future__ import annotations

import numpy as np

EDGES = np.arange(-260, 261, 4)


def _histogram(codes):
    counts = np.histogram(codes, EDGES)[0]
    # The denominator counts every sample, as upstream does.
    return (counts + 1) / (codes.size + len(counts))


def kl(real_codes, generated_codes):
    real, generated = _histogram(real_codes), _histogram(generated_codes)
    return float(np.sum(real * np.log(real / generated)))


def block_and_tile_kl(clean, real, generated, tile=32):
    """Mean KL over blocks and over each block's tiles, for (B, H, W, 3) uint8."""
    clean = np.asarray(clean, np.int16)
    real_codes = np.asarray(real, np.int16) - clean
    generated_codes = np.asarray(generated, np.int16) - clean
    blocks, tiles = [], []
    for r, g in zip(real_codes, generated_codes):
        blocks.append(kl(r, g))
        height, width = r.shape[:2]
        tiles.append(float(np.mean([kl(r[y:y + tile, x:x + tile], g[y:y + tile, x:x + tile])
                                    for y in range(0, height, tile) for x in range(0, width, tile)])))
    return dict(kl_block256=float(np.mean(blocks)), kl_tiles32=float(np.mean(tiles)))
