#!/usr/bin/env bash
# Train a DINO-WM on the LGES episodes (wm/collect.py output) with this repo's own venv.
#
#   ./run_lges.sh                       # defaults: 5 Hz frames, frameskip 1, 3 history frames, batch 32
#   ./run_lges.sh frameskip=2 training.batch_size=16 training.epochs=50
#   DATASET_DIR=/somewhere ./run_lges.sh env.dataset.n_rollout=50      # a quick run on 50 episodes
#
# Checkpoints: ./outputs/<date>/<time>/checkpoints (ckpt_base_path below). wandb runs offline (./wandb); sync
# later with `wandb sync` if wanted. DINOv2 weights are cached under .torch_home (first run downloads them).
set -eu
cd "$(dirname "$0")"
export DATASET_DIR=${DATASET_DIR:-$HOME/Humanoid/LGES_Sim/data}
export TORCH_HOME=${TORCH_HOME:-$PWD/.torch_home}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$PWD}
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # less fragmentation on a 16 GB card
exec .venv/bin/python train.py --config-name train.yaml env=lges img_size=224 frameskip=1 num_hist=3 num_pred=1 \
  training.batch_size=8 training.epochs=200 env.num_workers=8 ckpt_base_path="$PWD" "$@"
