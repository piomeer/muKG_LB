# GPU Runtime Architecture Design for MuKG

**Date**: 2026-07-21  
**Phase**: 7 Step 4  
**Status**: Draft  

## 1. Background
Phase 7 Step 3 established a GPU cost model demonstrating that pure GPU negative sampling (Route A) is already faster than CPU at our current batch size (N=750k). This document shifts the focus from *whether* to use GPU to *how* to architect the GPU runtime, especially in conjunction with the Cost-aware Batch Packing (CBP) framework developed in Phases 1–6.

## 2. Candidate Architectures

### Route A: Pure GPU Sampling
- **Flow**: CPU prepares positive triples → transfer to GPU → `torch.randint` + `torch.isin` on GPU → training forward/backward.
- **Pros**: Simple implementation; eliminates CPU negative sampling entirely.
- **Cons**: Ignores per‑sample cost; makes CBP scheduling irrelevant; no support for multi‑GPU load balancing.
- **Code sketch** (conceptual):
  ```python
  negs = torch.randint(0, E, (B*N,), device='cuda')
  mask = ~torch.isin(negs, pos_tails)
  valid_negs = negs[mask][:B*N]
  ```

### Route B: GPU Sampling + Negative Cache
- **Flow**: As Route A, but maintains a GPU‑resident cache of pre‑generated negatives for high‑degree entities.
- **Pros**: Potentially reduces isin overhead for extreme hubs.
- **Cons**: Additional VRAM; complex cache invalidation; microbenchmarks show isin cost is only ~1.16ms, so benefit is marginal.
- **Not recommended** given current data.

### Route C: CBP + GPU Runtime (Cost‑aware Unified Runtime)
- **Flow**:  
  1. CBP Scheduler (CPU): reorder/repack training triples based on cost model.  
  2. For each batch: positive triples are sent to GPU.  
  3. GPU sampling module generates negatives (Route A logic).  
  4. Training proceeds.  
  The cost model now serves as a *universal batch complexity estimator*, not just for CPU sampling.
- **Pros**:
  - Directly extends CBP investment; cost model becomes framework‑wide.
  - Enables load balancing for multi‑GPU (DDP): equal batch cost → reduced synchronization wait.
  - Extensible architecture: future modules (e.g., dynamic batching, memory prefetch) can plug into the runtime.
- **Cons**: Slightly more complex than Route A, but most components already exist.

## 3. Evaluation Matrix

| Criterion | Route A | Route B | Route C | Comments |
|-----------|---------|---------|---------|----------|
| Performance (current) | 4 | 5 | 5 | All beat CPU; C/B offer marginal extra gain now |
| Engineering effort | 5 | 2 | 3 | A trivial; B needs cache; C reuses existing code |
| Continuity with CBP | 1 | 2 | 5 | Only C preserves scheduling investment |
| Multi‑GPU readiness | 2 | 2 | 5 | Balanced batches crucial for DDP |
| Extensibility | 3 | 3 | 5 | C's framework supports new modules |
| Narrative coherence | 2 | 2 | 5 | C ties whole project together |

**Recommendation**: Proceed with Route C as the final runtime architecture. Route A will be implemented first as a validation milestone.

## 4. Unified Runtime Framework Blueprint

The framework consists of five core modules:

1. **Cost Model** – Predicts batch execution cost (currently negative sampling time; evolves to GPU computation + communication cost).
2. **Scheduler** – Sorts and packs batches using CostSorter + FFDPacker (or future variants).
3. **Runtime Policy** – Decides execution strategy (e.g., pure GPU vs. hybrid fallback); currently fixed to GPU.
4. **GPU Sampler** – Contains vectorized negative sampling kernel(s).
5. **Batch Provider** – Zero‑intrusion adapter that replaces PyTorch DataLoader.

Execution flow per epoch:
```
Feature Extraction (once) → Cost Table
        ↓
Scheduler.pack_batches() → reordered triples
        ↓
BatchProvider.iterate() → yields batches
        ↓
GPU Sampler (inside training loop) → negatives on GPU
        ↓
Forward/Backward/Optimizer
```

## 5. Next Steps

- **Phase 7 Step 5**: Refine the cost model for GPU context (e.g., use FLOPS or memory footprint).
- **Phase 7 Step 6**: Implementation plan for Phase 8 (GPU Sampler prototype, integration with CBP, evaluation).
- **Phase 8**: Code implementation and end‑to‑end evaluation.

This architecture is subject to review during Step 5 & 6.