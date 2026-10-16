# BurstGraphFM

**Research prototype for encrypted traffic analysis using Burst-level behavior graphs and self-supervised pretraining.**

> Repository initialization in progress. The complete v0.3 source code has **not yet been uploaded**; do not treat this repository as a runnable release until the source files and configs are committed.

## Planned v0.3 components

- PCAP → bidirectional Flow → Burst → Temporal Graph
- Burst Sequence Transformer and Graph Transformer baselines
- Masked Burst Modeling and temporal-gap reconstruction
- Learned non-local interaction edges
- Graph structural consistency pretraining
- Downstream classifier fine-tuning

## Availability

Full v0.3 source is available in the `BurstGraphFM_GitHub_v0.3_clean.zip` archive previously provided in the ChatGPT conversation. Upload that archive's **contents** (not the ZIP itself) to the repository root before running any scripts.

## Validation status

Only prototype CPU smoke tests have been reported; full ISCX VPN-nonVPN runs and dual H20 validation are pending. No research-performance claims are made.
