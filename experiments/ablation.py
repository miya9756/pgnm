"""Ablation of PGNM's fitted components (Tab. 6).

    python -m experiments.ablation fit --part sidd       # 5 variants x 16 camera-ISO classes
    python -m experiments.ablation render --part sidd
    python -m experiments.ablation score --part sidd
    python -m experiments.ablation fit --part film       # 3 kernel sizes x 8 radii
    python -m experiments.ablation render --part film
    python -m experiments.ablation score --part film     # needs torch, DISTS_pytorch and lpips
    python -m experiments.ablation table                 # Tab. 6

Full PGNM is the PGNM row of Tab. 1 and Tab. 3, so run the PGNM render and
score stages of experiments.sidd and experiments.film first. Each variant is
pgnm.fit on the same training pairs with one fitted component changed:

  no_masks    The pilot strength curve and the correlation use every pixel:
              no flatness selection and no clipping guard. The quantile lookup
              never uses these masks and stays as in full PGNM.
  kernel_k    Convolution support k x k instead of 25 x 25: the same spectral
              square root, cropped, with each output's energy renormalized
              (exactly pgnm.fit(pairs, kernel_size=k)). k = 1 keeps the color
              mixing but no spatial correlation.
  gaussian    The quantile lookup g(x, t) is replaced by mu(x) + sigma(x) t,
              the mean and standard deviation of the same pooled residual
              distribution that the lookup inverts, limited to the pgnm/5
              range |g| <= 1. Kernels are full PGNM's.

The SIDD part has all variants, the film-grain part the kernel sizes. Variants
are rendered by the unchanged renderer from the random streams of Tab. 1 and
Tab. 3 (same bank seeds and window seeds), so run r of every variant starts
from the same white noise and differences come from the fitted parameters.
Scores use the same metrics and, for film grain, the same Newson references.

Table: full PGNM's level, then each variant's change relative to it (lower is
better for every column). SIDD (39 sources): a change is a tie when the 95%
interval of the level difference includes zero; the interval resamples the
physical scenes (100,000 draws, seed 11) and whole generator runs, with the
same draws for both variants. Film grain (levels averaged over the eight
radii): a change is worse or better when more than half of the 60 test images
move in that direction at every radius, otherwise a tie.

Outputs: runs/ablation/<sidd|film>/{presets,generated,scores}/<variant>/...
and runs/ablation/table.json.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean

import numpy as np

from . import config, film, sidd
from .seeds import RUNS, radius_key

SIDD_VARIANTS = dict(no_masks="No fitting masks", kernel_1="Kernel 1x1", kernel_5="Kernel 5x5",
                     kernel_11="Kernel 11x11", gaussian="Gaussian conditional")
FILM_VARIANTS = dict(kernel_1="Kernel 1x1", kernel_5="Kernel 5x5", kernel_11="Kernel 11x11")
SIDD_COLUMNS = sidd.COLUMNS   # KL on 32x32 tiles, conditional W1, spatial and color correlation error
FILM_COLUMNS = film.COLUMNS   # conditional W1, spatial correlation error, DISTS, LPIPS


def root(part):
    return config.runs("ablation", part)


def kernel_size(variant):
    return int(variant.split("_")[1]) if variant.startswith("kernel_") else 25


# Fitting -------------------------------------------------------------------------

def gaussian_lookup(histogram):
    """Tabulate g(x, t) = mu(x) + sigma(x) t on the lookup's own knots.

    mu and sigma^2 are the moments of the pooled residual distribution that
    pgnm.fitting.build_residual_lookup inverts: the same all-pixel histogram
    and intensity pooling, residual codes spread uniformly over their one-code
    bin (hence the 1/12). The renderer interpolates linearly in t, so the table
    is exactly Gaussian. It is limited to |g| <= 1 like every pgnm/5 lookup;
    this matters only for a few near-black red and blue rows of S6 at ISO 800
    and 3200 (10 of the 12,288 table rows over all classes).
    """
    from pgnm import fitting
    probabilities = fitting.pool_residual_histograms(histogram)
    codes = np.arange(511, dtype=np.float64) - 255
    mean = probabilities @ codes
    variance = probabilities @ (codes * codes) - mean * mean + 1 / 12
    lookup = (mean[..., None] + np.sqrt(variance)[..., None] * fitting.GAUSSIAN_KNOTS) / 255
    return np.clip(lookup, -1, 1)


def _correlation(pairs, pilot, threshold=None):
    """pgnm.fit's correlation pass: flat pixels inside the clipping guard.

    With ``threshold=None`` every pixel is used, without the clipping guard.
    """
    from pgnm import fitting

    def samples():
        for clean, noisy in pairs():
            clean, noisy = np.asarray(clean, np.float64), np.asarray(noisy, np.float64)
            if threshold is None:
                every = np.ones(clean.shape[:2], bool)
                yield fitting.normalize_residuals(clean, noisy, every, pilot)[0], every
            else:
                flat = fitting.flat_pixel_contrast(clean) <= threshold
                yield fitting.normalize_residuals(clean, noisy, flat, pilot)  # (field, flat AND guard)
    return fitting.measure_cross_correlation(samples())[0]


def _preset(variant, kernels, lookup, **provenance):
    from pgnm import fitting, validate
    from pgnm.parameters import SCHEMA
    return validate(dict(
        schema=SCHEMA, method="ours", name="PGNM", kernels=kernels.tolist(),
        marginal=dict(t=fitting.GAUSSIAN_KNOTS.tolist(), g=lookup.tolist(), step=1.0),
        provenance=dict(status="fitted", variant=variant, kernel_size=int(kernels.shape[0]), **provenance)))


def fit_sidd_variants(pairs):
    """Full PGNM and every SIDD variant of one class, from shared data passes.

    The steps are pgnm.fit's own estimators: flatness threshold, pilot
    strength on flat pixels, correlation on flat pixels inside the clipping
    guard, spectral factorization, all-pixel quantile lookup. pgnm.fit's
    fallback for an empty clipping guard never occurs on SIDD. The kernel
    sizes crop the factorization of the same correlation.
    """
    from pgnm import fitting
    threshold = fitting.flat_pixel_threshold(pairs)
    histogram = fitting.measure_residual_histogram(pairs)
    lookup = fitting.build_residual_lookup(histogram)
    masked = _correlation(pairs, fitting.fit_pilot_strength(pairs, threshold), threshold)
    unmasked = _correlation(pairs, fitting.fit_pilot_strength(pairs, np.inf))

    def kernels(correlation, size=25):
        return fitting.factor_cross_channel_kernels(correlation, size)[0]

    full = kernels(masked)
    variants = dict(full=(full, lookup), no_masks=(kernels(unmasked), lookup),
                    gaussian=(full, gaussian_lookup(histogram)))
    for name in SIDD_VARIANTS:
        if name.startswith("kernel_"):
            variants[name] = (kernels(masked, kernel_size(name)), lookup)
    return {name: _preset(name, k, g, flat_threshold=threshold) for name, (k, g) in variants.items()}


def fit_sidd(classes=None, variants=SIDD_VARIANTS):
    from pgnm import load, save
    for name in classes or sorted(sidd.split()["classes"]):
        targets = {v: root("sidd") / "presets" / v / (name + ".json") for v in variants}
        if all(t.exists() for t in targets.values()):
            continue
        presets = fit_sidd_variants(sidd.training_pairs(name))
        for variant, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            save(target, presets[variant])
        shipped = load(config.REPO / "presets/sidd" / (name + ".json"))
        same = (presets["full"]["kernels"] == shipped["kernels"] and
                presets["full"]["marginal"] == shipped["marginal"])
        print("fitted", name, "(full PGNM refit equals presets/sidd: %s)" % same, flush=True)


def fit_film(radii=film.RADII, variants=FILM_VARIANTS):
    from pgnm import fit, save
    for radius in radii:
        targets = {v: root("film") / "presets" / v / (radius_key(radius) + ".json") for v in variants}
        pending = [v for v, target in targets.items() if not target.exists()]
        if not pending:
            continue
        pairs = film.training_pairs(radius, film.blurs()[radius_key(radius)])
        for variant in pending:
            targets[variant].parent.mkdir(parents=True, exist_ok=True)
            save(targets[variant], fit(pairs, kernel_size=kernel_size(variant)))
            print("fitted", radius_key(radius), variant, flush=True)


# Rendering and scoring: the Tab. 1 and Tab. 3 stages under the variant's name ------

def render_sidd(images=None, variants=SIDD_VARIANTS):
    for variant in variants:
        sidd.render(images, presets=root("sidd") / "presets" / variant, root=root("sidd"), method=variant)


def score_sidd(images=None, variants=SIDD_VARIANTS):
    sidd.score(images, methods=list(variants), root=root("sidd"))


def render_film(radii=film.RADII, images=None, variants=FILM_VARIANTS):
    for variant in variants:
        film.render(radii, images, methods=("pgnm",), presets=root("film") / "presets" / variant,
                    root=root("film"), pgnm_name=variant)


def score_film(radii=film.RADII, images=None, variants=FILM_VARIANTS):
    film.score(radii, images, methods=list(variants), root=root("film"))


# Table ---------------------------------------------------------------------------

def _sidd_scores(folder, sources):
    """{column: (source, run) array} of one method's score files."""
    runs = [config.read_json(Path(folder) / ("img%02d.json" % i))["runs"] for i in sources]
    return {column: np.array([[r[column] for r in item] for item in runs], np.float64)
            for column, _ in SIDD_COLUMNS}


def sidd_table(full_scores=None, scores=None):
    from metrics.statistics import draws, level, paired_difference, source_weights
    data = sidd.split()
    sources, labels = data["sets"]["S39"], data["validation"]
    weights = source_weights(sources, labels)
    design = draws(weights, [labels[i]["scene"] for i in sources], runs=len(RUNS), B=100_000, seed=11)
    full = _sidd_scores(full_scores or config.runs("sidd/scores/pgnm"), sources)
    result = dict(full={column: level(full[column], weights) for column, _ in SIDD_COLUMNS})
    for variant in SIDD_VARIANTS:
        values = _sidd_scores(Path(scores or root("sidd") / "scores") / variant, sources)
        result[variant] = {}
        for column, _ in SIDD_COLUMNS:
            change = paired_difference(values[column], full[column], weights, design)
            low, high = change["interval"]
            verdict = "tie" if low <= 0 <= high else "worse" if change["difference"] > 0 else "better"
            result[variant][column] = dict(
                level=level(values[column], weights), difference=change["difference"],
                interval=change["interval"], percent=100 * change["difference"] / result["full"][column],
                verdict=verdict)
    return result


def _film_means(folder):
    """{(radius, image): {column: mean over the 25 output/reference pairs}}."""
    means = {}
    for radius in film.RADII:
        for image in film.split()["test_images"]:
            pairs = config.read_json(Path(folder) / radius_key(radius) / (image + ".json"))["pairs"]
            means[radius, image] = {column: fmean(p[column] for p in pairs) for column, _ in FILM_COLUMNS}
    return means


def film_table(full_scores=None, scores=None):
    images = list(film.split()["test_images"])

    def level(means, column):  # images equally per radius, then radii equally (as Tab. 3)
        return fmean(fmean(means[r, i][column] for i in images) for r in film.RADII)

    full = _film_means(full_scores or config.runs("film/scores/pgnm"))
    result = dict(full={column: level(full, column) for column, _ in FILM_COLUMNS})
    for variant in FILM_VARIANTS:
        means = _film_means(Path(scores or root("film") / "scores") / variant)
        result[variant] = {}
        for column, _ in FILM_COLUMNS:
            higher = [sum(means[r, i][column] > full[r, i][column] for i in images) for r in film.RADII]
            lower = [sum(means[r, i][column] < full[r, i][column] for i in images) for r in film.RADII]
            verdict = ("worse" if all(n > len(images) / 2 for n in higher) else
                       "better" if all(n > len(images) / 2 for n in lower) else "tie")
            value = level(means, column)
            result[variant][column] = dict(
                level=value, percent=100 * (value - result["full"][column]) / result["full"][column],
                images_higher=higher, images_lower=lower, verdict=verdict)
    return result


def table(parts=("sidd", "film")):
    result = {part: (sidd_table if part == "sidd" else film_table)() for part in parts}
    config.write_json(root("table.json"), result)
    return result


def level_text(value):
    """Three significant digits, at most four decimals: 0.0219, 1.32, 0.0094."""
    decimals = min(4, max(0, 2 - math.floor(math.log10(abs(value)))))
    return "%.*f" % (decimals, value)


def percent_text(value):
    """+1.2%, -8.5%, +46%, +586%: one decimal below 10%."""
    return ("%+.1f%%" if abs(value) < 10 else "%+.0f%%") % value


def print_table(result):
    for part, title, columns, variants in (
            ("sidd", "SIDD (39 sources)", SIDD_COLUMNS, SIDD_VARIANTS),
            ("film", "Film grain (8 radii)", FILM_COLUMNS, FILM_VARIANTS)):
        if part not in result:
            continue
        print("%-22s" % title + "".join("%15s" % label for _, label in columns))
        print("  %-20s" % "Full PGNM" + "".join("%15s" % level_text(result[part]["full"][c]) for c, _ in columns))
        for variant, label in variants.items():
            cells = [result[part][variant][c] for c, _ in columns]
            print("  %-20s" % label + "".join("%15s" % ("%s %s" % (percent_text(cell["percent"]), cell["verdict"]))
                                              for cell in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("fit", "render", "score", "table"))
    parser.add_argument("--part", choices=("sidd", "film"), nargs="+", default=["sidd", "film"])
    parser.add_argument("--variants", nargs="+", choices=tuple(SIDD_VARIANTS), help="default: all of the part")
    parser.add_argument("--classes", nargs="+", help="SIDD camera-ISO classes to fit, e.g. GP_10000")
    parser.add_argument("--images", nargs="+", help="SIDD Validation images (e.g. 14) or film test images (e.g. 01)")
    parser.add_argument("--radii", nargs="+", type=float, choices=film.RADII, default=list(film.RADII))
    args = parser.parse_args()
    if args.stage == "table":
        print_table(table(args.part))
        return
    for part in args.part:
        known = SIDD_VARIANTS if part == "sidd" else FILM_VARIANTS
        variants = [v for v in args.variants or known if v in known]
        if part == "sidd":
            images = [int(i) for i in args.images] if args.images else None
            if args.stage == "fit":
                fit_sidd(args.classes, variants)
            elif args.stage == "render":
                render_sidd(images, variants)
            else:
                score_sidd(images, variants)
        elif args.stage == "fit":
            fit_film(args.radii, variants)
        elif args.stage == "render":
            render_film(args.radii, args.images, variants)
        else:
            score_film(args.radii, args.images, variants)


if __name__ == "__main__":
    main()
