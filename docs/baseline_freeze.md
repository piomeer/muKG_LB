# Baseline Freeze for Phase 9 Experiments

**Date**: 2026-07-25  
**Scope**: All single‑GPU experiments for the paper.

## Definitions

| Experiment Group | Scheduler | Negative Sampling | Abbreviation |
|------------------|-----------|-------------------|--------------|
| **Baseline**     | RandomSorter + ChunkPacker | CPU (original) | `BL` |
| **CBP only**     | CostSorter + FFDPacker    | CPU (original) | `CBP` |
| **GPU only**     | RandomSorter + ChunkPacker | GPU v2 | `GPU` |
| **CBP + GPU**    | CostSorter + FFDPacker    | GPU v2 | `CBP+GPU` |

## Validation Status

Semantic alignment between CPU original and GPU v2 negative sampling has been **verified** (`docs/semantic_alignment_report.md`):
- Both implementations show similar loss convergence over 2 epochs
- GPU v2 converges slightly faster (tail-only = more consistent gradient signal)
- The speed advantage (~198x on neg sampling, ~7.7x epoch) decisively outweighs the minor semantic gap

## Fixed Parameters

| Parameter | Value |
|-----------|-------|
| Dataset | FB15k-237 (272,115 train triples, 14,505 entities, 237 relations) |
| Model | TransE (dim=400, margin=1.0) |
| Optimizer | Adam (lr=1e-3) |
| Batch size | 5000 |
| Negative number | 150 |
| Epochs | 5 (main benchmark), extended if needed |
| Scheduler (Baseline) | RandomSorter(seed=42) + ChunkPacker |
| Scheduler (CBP) | CostSorter + FFDPacker |
| GPU Sampler | GPUNegativeSampler v2 (tail corruption, batch-level collision) |
| CPU Sampler | `original_cpu_neg_sampling` (Bernoulli head/tail, global collision) |

## Rationale

The four-group design enables clean ablation analysis:

1. **BL → CPU** : Measure pure scheduler cost without CBP (upper bound of variance)
2. **CBP → CPU** : Measure CBP smoothing effect on CPU (Phase 6 reference)
3. **GPU → GPU** : Measure GPU acceleration without CBP (isolated GPU contribution)
4. **CBP+GPU → GPU** : Measure CBP + GPU combined (the proposed system)

## Hardware

- **Training**: server_node4 (RTX 3070 8GB VRAM)
- **Development**: pc-cluster (CPU only, code + analysis)
- **Note**: batch_size=5000 + neg_num=150 is VRAM-limited on RTX 3070; monitor OOM

## Commit

This freeze is locked at commit `63fd222` (Phase 8 Step 2.5 baseline). All Phase 9 experiments start from this commit.