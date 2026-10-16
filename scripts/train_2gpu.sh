#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
torchrun --standalone --nproc-per-node=2 -m burstgraphfm.train --config configs/baseline.yaml --model "${1:-graph}" --output "runs/${1:-graph}_ddp"
