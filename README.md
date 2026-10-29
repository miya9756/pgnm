# Parametric Gaussian Noise Mixer (PGNM): Unified Synthesis of Camera Noise and Film Grain

PGNM synthesizes noise from two fitted arrays: a full RGB convolution tensor
that turns independent Gaussian noise into a correlated latent field, and an
empirical lookup that maps each latent value, given the clean intensity, to a
signed residual. Both are estimated directly from clean/noisy image pairs, with
no neural network and no physical image-formation model.

This repository contains the paper ([PGNM.pdf](PGNM.pdf), a technical report
with its supplementary material), the method (`pgnm/`), the presets fitted for
the paper (`presets/`) and the code behind every table and figure of the main
paper.

## Install and try

```bash
pip install -e .                       # Python >= 3.10: numpy, scipy, numba, opencv, pillow, matplotlib
python -m unittest discover -s tests   # no data needed
```

```python
import numpy as np
from PIL import Image
from pgnm import PGNM, load

clean = np.asarray(Image.open("photo.png").convert("RGB"), np.float32) / 255
model = PGNM(load("presets/sidd/S6_800.json"))      # Galaxy S6 camera noise at ISO 800
noisy = model.render(clean, np.random.default_rng(0))
Image.fromarray(np.floor(np.clip(noisy, 0, 1) * 255 + .5).astype(np.uint8)).save("noisy.png")
```

Film-grain presets expect the clean image after its radius's 7×7 blur B
(`presets/film/blur.json`, see `experiments.film.expected_image`). How the
model works and how to read its code: [pgnm/README.md](pgnm/README.md).

## Layout

| Path                     | Contents                                                                          |
| ------------------------ | --------------------------------------------------------------------------------- |
| `PGNM.pdf`             | the paper: main part and supplementary material                                   |
| `pgnm/`                | the method: `model.py` (inference), `fitting.py`, `parameters.py`           |
| `presets/sidd/`        | 16 presets, one per SIDD camera–ISO class (Tab. 1)                               |
| `presets/film/`        | 8 presets for Newson radii 0.025–0.8, their blur B and the Zhang23 fits (Tab. 3) |
| `presets/denoise/`     | 15 presets fitted on the denoising training images (Tab. 2)                       |
| `presets/bandwidth/`   | PGNM and AV1 grain parameters fitted on the video clip (Tab. 5)                   |
| `checkpoints/denoise/` | the six trained DnCNNs of Tab. 2, one per training-noise source                   |
| `datasets/`            | exact training keys, held-out captures, test images and scored source sets        |
| `metrics/`             | conditional W1, spatial/color correlation error, KL, DISTS/LPIPS, scene bootstrap |
| `experiments/`         | one module per table: fit, render, score, table stages                            |
| `baselines/`           | adapters for the published generators, our Zhang23, fetching and environments     |
| `figures/`             | one script per figure                                                             |

## Paper to code

| Paper                               | Command                                                                                                   | Needs                                               |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Tab. 1 camera noise on SIDD         | `python -m experiments.sidd render`, `score`, `table`                                               | SIDD; baselines below                               |
| Fig. 5 SIDD visual comparison       | `python -m figures.visual_comparison sidd`                                                              | Tab. 1 outputs                                      |
| Tab. 2 denoising                    | `python -m experiments.denoise store`, `train --source S`, `evaluate --source S`, `table`         | SIDD, one GPU (30–47 h per source)                 |
| Tab. 2 from the shipped models      | `python -m experiments.denoise evaluate --source S --checkpoint checkpoints/denoise/S.pt`, `table`    | SIDD Validation, one GPU (minutes)                  |
| Tab. 3 simulated film grain         | `python -m experiments.film references`, `render`, `score`, `table`                               | GrainNet data, Newson renderer (GPU)                |
| Fig. 6 film-grain visual comparison | `python -m figures.visual_comparison film`                                                              | Tab. 3 outputs                                      |
| Tab. 4 model size and speed         | `python -m experiments.cost parameters --method M`, `time --method M --device D`, `table`           | checkpoints; GPU for the GPU rows                   |
| Tab. 5 video delivery               | `python -m experiments.bandwidth prepare`, `encode`, `decode`, `score`, `appearance`, `table` | codecs (`python -m baselines.codecs`), ffmpeg 4.4 |
| Tab. 6 ablation                     | `python -m experiments.ablation fit\|render\|score --part sidd\|film`, `table`                           | Tab. 1 and Tab. 3 PGNM scores                       |
| Fig. 1 teaser                       | `python -m figures.teaser --photo your.jpg`                                                             | any photo; Linux Biolinum fonts                     |
| Fig. 2 inference pipeline           | drawn by hand, no script                                                                                  | —                                                  |
| Fig. 3 conditional residual density | `python -m figures.residual_density`                                                                    | SIDD-Medium                                         |
| Fig. 4 conditional quantile mapping | `python -m figures.quantile_mapping`                                                                    | —                                                  |
| Fig. 7 joint PCA of the presets     | `python -m figures.joint_pca`                                                                           | presets only                                        |

Every stage writes under `runs/` and skips work that already exists, so stages
can be interrupted and resumed, or run per image (`--images`) or per class
(`--classes`) in parallel. `fit` stages are optional: the shipped presets are
the ones they produce. Each module's docstring describes the experiment.

## Data

Set these variables or place the data under `data/`:

| Variable                 | Dataset                                                                     |
| ------------------------ | --------------------------------------------------------------------------- |
| `PGNM_SIDD_MEDIUM`     | SIDD-Medium sRGB, the folder containing `Data/` (fitting)                 |
| `PGNM_SIDD_VALIDATION` | `ValidationGtBlocksSrgb.mat`, `ValidationNoisyBlocksSrgb.mat` (scoring) |
| `PGNM_FILM`            | GrainNet's `NeuralFilmGrainRendering` training pairs                      |
| `PGNM_EXTERNAL`        | fetched baseline code, weights and tools (default `external/`)            |
| `PGNM_RUNS`            | all outputs (default `runs/`)                                             |

SIDD crops are cut on the fly from the full Medium images (`datasets/sidd.json`
explains the keys). GrainNet's 60 test images come with its repository.
`experiments.bandwidth prepare` downloads the 100 frames of Netflix's
*Nocturne* (CC BY 4.0) it needs from Netflix's open-content bucket.

## Baselines

```bash
python -m baselines.fetch                      # clone at the pinned commits, verify weights
bash baselines/environments.sh naflow          # one environment per code base
python -m experiments.sidd export-clean        # clean Validation blocks for those environments
python -m baselines.render_sidd --method naflow    # from the repository root, in that environment
python -m baselines.render_film                # GrainNet, in its environment
```

Published checkpoints are used at their native settings. NM-FlowGAN runs in the
NAFlow environment. GMDCN is our implementation of Song et al. (2023); its code
is not included. Zhang23 is implemented in `baselines/zhang23.py`.

## Randomness

Every generated image or block has its own seed, a hash of a fixed label and
the item (`experiments/seeds.py`), so any item can be regenerated alone. PGNM
uses four template seeds per generator run and one window draw per item. With
the same software versions, reruns give identical outputs: this code reproduces
the paper's presets, outputs and scores exactly. The exception is DnCNN
training for Tab. 2, which uses cuDNN autotuning as the reported runs did and
is therefore reproducible up to training noise; evaluating the shipped
checkpoints reproduces Tab. 2 exactly.

## License

© 2026 Mingyang Song. The code, presets, checkpoints and paper in this
repository are licensed under [CC BY-NC-ND 4.0](LICENSE)
(https://creativecommons.org/licenses/by-nc-nd/4.0/): you may use them and share
them unmodified for non-commercial purposes, with attribution; you may adapt them
for your own use, but not share adapted versions. The datasets, the published
baselines and the *Nocturne* clip are not part of this repository and keep their
own licenses; so do the third-party images shown in the paper's figures.
