#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} torchrun --standalone --nproc_per_node=2 \
  -m burstgraphfm.pretrain \
  --config configs/pretrain_structural.yaml \
  --output runs/pretrain_structural
