"""Real-noise denoising (Tab. 2): one DnCNN per training-noise source.

A denoiser trained on synthetic pairs works on real images only as far as the
synthesized noise resembles real noise. As in NM-FlowGAN's denoiser setup, a
DnCNN is trained for 30 epochs on 96x96 patches from 246 SIDD-Medium images
(its 15 camera-ISO classes), once per training-noise source: real pairs, PGNM
(fitted per class on the same images), GMDCN, NM-FlowGAN, NAFlow and NAFlow-NAS.
Noise is drawn anew for every patch and epoch and quantized to 8 bits; all
sources share initialization, patch order and augmentation (training seed 1).
The epoch-30 model is scored by PSNR and SSIM on all 1,280 SIDD Validation blocks.

    python -m experiments.denoise store                    # decode the 246 training pairs (~20 GB)
    python -m experiments.denoise fit                      # optional: refit presets/denoise into
                                                           #   runs/denoise/presets (CPU; train --presets)
    python -m experiments.denoise train --source pgnm      # one GPU, see run times below
    python -m experiments.denoise evaluate --source pgnm   # SIDD Validation, a few minutes
    python -m experiments.denoise table                    # Tab. 2

The six trained models of Tab. 2 are in checkpoints/denoise/<source>.pt;
`evaluate --source S --checkpoint checkpoints/denoise/S.pt` scores one in minutes.

Sources: real, pgnm, gmdcn, nmflowgan, naflow, naflow_nas. All six run in one
environment with torch 2.1.2 (CUDA 11.8), NumPy 1.26, SciPy, Numba, OpenCV,
scikit-image, OmegaConf, PyYAML, einops and loguru; the neural generators load
their upstream code and weights from $PGNM_EXTERNAL (python -m baselines.fetch).

Run times on one RTX 3090 (8 CPU cores, 7 data workers), 345,480 iterations:
about 30 h for Real and PGNM, 36 h NAFlow, 39 h NM-FlowGAN, 40 h NAFlow-NAS and
47 h GMDCN, with up to 12 GB of GPU memory. PGNM renders on the CPU in the data
workers (about 6.5 ms per patch). Fitting all 15 PGNM presets takes 4 to 8 hours
on two CPU threads. cuDNN autotuning is on, as for the reported runs, so a rerun
reproduces Tab. 2 up to training noise rather than bit for bit.

Modules: data (split, patch store, batches, random streams), noise (the six
sources and PGNM fitting), train (DnCNN and its training loop), evaluate
(scores and table).
"""
