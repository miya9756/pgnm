#!/usr/bin/env bash
# One conda environment per upstream code base, as used for the paper.
#   bash baselines/environments.sh NAME     NAME: base naflow gmdcn danet srgb_flow grainnet newson denoise
# Environments go to $PGNM_ENVS (default external/envs). Pins follow each upstream's own
# environment file; the few changes needed for an RTX A6000 (sm_86, driver 535, CUDA <= 12.2)
# are noted. NM-FlowGAN runs in the naflow environment. Keep ~/.local out of every environment.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ENVS=${PGNM_ENVS:-${PGNM_EXTERNAL:-$HERE/external}/envs}
TORCH=https://download.pytorch.org/whl
export PYTHONNOUSERSITE=1 PIP_USER=0 PIP_DISABLE_PIP_VERSION_CHECK=1
NAME=${1:?usage: environments.sh NAME}
PREFIX=$ENVS/$NAME
make_env() { [ -x "$PREFIX/bin/python" ] || conda create -q -y -p "$PREFIX" "python=$1" pip >/dev/null; }
P=$PREFIX/bin/python

case $NAME in
base)  # PGNM, metrics, SIDD tables, figures (no GPU needed)
  make_env 3.11
  $P -m pip install -q -e "$HERE" ;;
naflow)  # NAFlow environments.yaml: py3.11, torch 2.1.2, CUDA 11.8, lightning 2.1.4; also NM-FlowGAN
  make_env 3.11
  $P -m pip install -q torch==2.1.2 torchvision==0.16.2 --index-url $TORCH/cu118
  $P -m pip install -q lightning==2.1.4 pytorch-lightning==2.1.4 scikit-image scipy einops pillow pandas \
    "opencv-python<4.12" timm matplotlib natsort tensorboard omegaconf rich pyyaml "numpy<2" "setuptools<70" ;;
gmdcn)  # our GMDCN implementation: py3.11; torch 2.5.1+cu121
  make_env 3.11
  $P -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url $TORCH/cu121
  $P -m pip install -q "einops>=0.8" "lmdb>=1.4" "loguru>=0.7" "lpips>=0.1.4" "matplotlib>=3.7" "numpy>=1.24" \
    "opencv-python>=4.8" "Pillow>=10.0" "pytorch-lightning>=2.5" "timm>=1.0" "tqdm>=4.64" pyyaml
  $P -m pip install -q --no-deps -e "${PGNM_EXTERNAL:-$HERE/external}/gmdcn/src" ;;
danet)  # DANet environment.yml: py3.7.4, torch 1.3.1+cu100; CUDA 10.0 cannot target sm_86 -> torch 1.8.1+cu111
  make_env 3.7
  $P -m pip install -q torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f $TORCH/torch_stable.html
  $P -m pip install -q numpy==1.16.5 scipy==1.3.1 scikit-image==0.15.0 opencv-python==4.1.1.26 pillow==6.1.0 \
    h5py==2.10.0 commentjson==0.8.3 matplotlib==3.1.1 ;;
srgb_flow)  # Noise2NoiseFlow requirements.txt: torch 1.11.0 (cu113 wheels support sm_86)
  make_env 3.8
  $P -m pip install -q torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url $TORCH/cu113
  $P -m pip install -q opencv-python==4.5.5.64 h5py==3.6.0 scikit-image==0.19.2 scipy matplotlib scikit-learn \
    tensorboard==2.8.0 tensorboardX "protobuf>=3.20,<3.21" "numpy<1.24" ;;
grainnet)  # GrainNet filmgrain.yml: py3.10, torch 2.5.0 (cu121 build); also DISTS/LPIPS film scoring
  make_env 3.10
  $P -m pip install -q torch==2.5.0 torchvision==0.20.0 --index-url $TORCH/cu121
  $P -m pip install -q numpy==2.2.1 scipy==1.14.1 pandas==2.2.3 matplotlib==3.10.0 imageio==2.36.1 lpips==0.1.4 \
    dists-pytorch==0.1 tqdm==4.67.1 pillow==11.0.0 tensorboard==2.18.0 ;;
denoise)  # Tab. 2: DnCNN training with all six training-noise sources in one environment
  make_env 3.11
  $P -m pip install -q torch==2.1.2 --index-url $TORCH/cu118
  $P -m pip install -q numpy==1.26.4 scipy==1.17.1 numba==0.61.2 omegaconf==2.3.1 pyyaml==6.0.3 \
    opencv-python-headless==4.11.0.86 scikit-image==0.26.0 einops==0.8.2 loguru==0.7.3 pillow==12.3.0 tqdm==4.70.1 ;;
newson)  # CUDA 11.8 nvcc and image headers for the Newson renderer build (python -m baselines.fetch newson)
  [ -x "$PREFIX/bin/nvcc" ] || conda create -q -y -p "$PREFIX" -c nvidia/label/cuda-11.8.0 -c conda-forge \
    cuda-nvcc cuda-cudart-dev libtiff libpng freeglut >/dev/null
  echo "then: python -m baselines.fetch newson --nvcc-prefix $PREFIX"; exit 0 ;;
*) echo "unknown environment $NAME" >&2; exit 2 ;;
esac
echo "$NAME ready: $P"
