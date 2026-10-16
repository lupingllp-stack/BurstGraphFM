# BurstGraphFM v0.3 — Structural Self-Supervised Pretraining

Research prototype for encrypted traffic analysis with Burst-level graphs, global Graph Transformer, learned non-local interaction edges and structural self-supervised learning.

**Status:** Source-code prototype; CPU smoke checks have been performed locally, but ISCX VPN-nonVPN results and 2×H20 runtime are not yet validated. Do not interpret synthetic-data accuracy as scientific evidence.

## 1. Clone on the training server

```bash
git clone https://github.com/lupingllp-stack/BurstGraphFM.git
cd BurstGraphFM
conda create -n burstgraphfm python=3.11 -y
conda activate burstgraphfm
# Choose CUDA-enabled PyTorch appropriate for your NVIDIA driver from pytorch.org.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

The cu126 install line is an **example**, not a required driver/runtime match. Check your `nvidia-smi` output and PyTorch installation selector first.

## 2. Local smoke test before using real data

```bash
python -m pytest -q
python scripts/make_smoke_pcap.py
python -m burstgraphfm.build --manifest data/smoke/manifest.json --output data/processed/smoke.pt
python -m burstgraphfm.pretrain --config configs/pretrain_structural_smoke.yaml --output runs/ssl_smoke
python -m burstgraphfm.train --config configs/finetune_structural_smoke.yaml --model structural --pretrained runs/ssl_smoke/best.pt --output runs/finetune_smoke
```

These synthetic samples only test the pipeline. They are **not** a benchmark.

## 3. Prepare ISCX VPN-nonVPN PCAP

Download PCAPs separately from the UNB/CIC dataset source (dataset files are not included in this repository). Unpack under `data/raw/iscx/`.

```bash
python scripts/create_manifest.py --raw-dir data/raw/iscx --output data/manifest.json --label-mode vpn_binary
```

**IMPORTANT:** Before using the manifest, manually verify every inferred `path`, `label` and `group` against official capture metadata. The `vpn_binary` mode only uses filename heuristics. Splitting is capture-group-disjoint and requires at least three independent capture groups per class. The PCAP parser supports classic Ethernet PCAP, not PCAPNG; convert unsupported captures beforehand.

```bash
python -m burstgraphfm.build --manifest data/manifest.json --output data/processed/graphs.pt --max-bursts 64 --min-packets 3 --max-flows-per-capture 3000
```

## 4. Self-supervised pretraining on two NVIDIA GPUs

```bash
nvidia-smi
bash scripts/pretrain_structural_2gpu.sh
```

Checkpoint: `runs/pretrain_structural/best.pt`

SSL tasks include masked Burst reconstruction, masked temporal-gap reconstruction, cross-view Flow contrastive learning and learned-relation consistency. The relation scorer's selected non-local edge weights contribute to the attention bias; the hard Top-k choice itself is discrete.

## 5. Fine-tune on two GPUs

```bash
torchrun --standalone --nproc_per_node=2 -m burstgraphfm.train \
  --config configs/finetune_structural.yaml \
  --model structural \
  --pretrained runs/pretrain_structural/best.pt \
  --output runs/finetune_structural
```

Compare against independent scratch/sequence/graph controls:

```bash
bash scripts/train_2gpu.sh sequence
bash scripts/train_2gpu.sh graph
```

## Project structure

- `burstgraphfm/pcap.py`: classic PCAP packet reader
- `burstgraphfm/build.py`: bidirectional Flow → direction-change Burst → 16-D node / 4-D temporal edge features
- `burstgraphfm/data.py`: batching, group-based splitting, feature normalization
- `burstgraphfm/model.py`: sequence and graph classifiers
- `burstgraphfm/interaction.py`: learned sparse non-local attention bias
- `burstgraphfm/pretrain.py`: masked-behavior/temporal objectives and DDP loop
- `burstgraphfm/structural.py`: relation-aware graph contrastive pretraining
- `burstgraphfm/train.py`: classifier training and encoder transfer
- `configs/`, `scripts/`, `tests/`: reproducibility and smoke tests

## Limitations and safeguards

- v0.3 is **not** a validated large-scale foundation model yet.
- Current parser is a research prototype and does not implement full TCP stream reassembly, robust PCAPNG, all IPv6 extension headers or large-PCAP streaming.
- A learned edge is not verified ground-truth interaction semantics.
- Group-level labels must be checked manually, and held-out captures should not enter pretraining.
- Only run `torch.load` on trusted graph files and checkpoints.
- Raw captures, checkpoints, secrets, and potentially identifying traffic metadata are excluded by `.gitignore`.

## Dataset reference

[ISCX VPN-nonVPN 2016 (UNB)](https://www.unb.ca/cic/datasets/vpn.html)
