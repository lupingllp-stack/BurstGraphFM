#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
torchrun --standalone --nproc_per_node=2 -m burstgraphfm.pretrain --config configs/pretrain.yaml --output runs/pretrain_graph
