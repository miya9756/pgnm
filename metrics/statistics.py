"""Source weighting and scene-bootstrap intervals for SIDD tables.

SIDD scores are means over five generator runs per source image. Sources are
weighted so that every camera counts equally, every ISO equally within its
camera and every image equally within its camera-ISO class.

Intervals (the ablation, Tab. 6) resample physical scenes with
replacement, keeping the source weights fixed, and resample whole generator
runs; differences between two variants use the same draws for both.
"""
from __future__ import annotations

import numpy as np


def source_weights(sources, labels):
    """Equal cameras, equal ISOs within a camera, equal images within a class."""
    tree = {}
    for source in sources:
        label = labels[int(source)]
        tree.setdefault(label["camera"], {}).setdefault(str(label["iso"]), []).append(source)
    weight = {source: 1 / (len(tree) * len(classes) * len(images))
              for classes in tree.values() for images in classes.values() for source in images}
    return np.array([weight[source] for source in sources])


def _cells(values):
    """(source, run) -> (source, run, 1), the layout the draws are defined on."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected complete finite (source, run) scores")
    return x[:, :, None]


def level(values, weights):
    """Weighted mean over sources of the per-source mean over runs."""
    w = np.asarray(weights, dtype=np.float64)
    return float((w / w.sum()) @ _cells(values).mean((1, 2)))


def draws(weights, clusters, runs=5, B=100_000, seed=11):
    """Bootstrap design: scene multiplicities and resampled generator runs."""
    w = np.asarray(weights, dtype=np.float64)
    labels = [str(c) for c in clusters]
    unique = list(dict.fromkeys(labels))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=B).astype(np.int32)
    run_ids = rng.integers(0, runs, size=(B, runs), dtype=np.int16)
    rng.integers(0, 1, size=(B, 1), dtype=np.int16)  # one reference observation; kept for the stream
    return dict(weights=w / w.sum(), cluster_index=np.array([unique.index(c) for c in labels], np.int32),
                cluster_counts=counts, run_ids=run_ids, runs=runs, B=B)


def bootstrap_levels(values, design, chunk=2048):
    """Level of each of the B bootstrap draws."""
    x = _cells(values)
    if x.shape[1] not in (1, design["runs"]):
        raise ValueError("Scores need one column per generator run")
    samples = np.empty(design["B"])
    for start in range(0, design["B"], chunk):
        stop = min(start + chunk, design["B"])
        w = design["cluster_counts"][start:stop, design["cluster_index"]] * design["weights"]
        w /= w.sum(1)[:, None]
        ids = design["run_ids"][start:stop]
        run_share = (np.ones((stop - start, 1)) if x.shape[1] == 1 else
                     np.stack([(ids == j).mean(1) for j in range(x.shape[1])], 1))
        per_source = np.einsum("ngr,bg,br->bn", x, run_share, np.ones((stop - start, 1)), optimize=True)
        samples[start:stop] = (w * per_source).sum(1)
    return samples


def interval(samples, alpha=.05):
    return np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)]).tolist()


def paired_difference(variant, reference, weights, design, alpha=.05):
    """Level difference variant - reference and its 95% bootstrap interval."""
    w = np.asarray(weights, dtype=np.float64)
    difference = float((w / w.sum()) @ (_cells(variant).mean((1, 2)) - _cells(reference).mean((1, 2))))
    samples = bootstrap_levels(variant, design) - bootstrap_levels(reference, design)
    return dict(difference=difference, interval=interval(samples, alpha))
