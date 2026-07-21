# GPU Migration Feasibility for MuKG Runtime Pipeline

**Date**: 2026-07-21  
**Phase**: 7 Step 2  
**Status**: Draft for review  

## 1. Current Runtime Pipeline (Post-CBP)

Training step breakdown (per batch, average over 55 batches, Baseline config):

| Stage | CPU/GPU | Time (ms) | % of Step |
|-------|---------|-----------|------------|
| Negative Sampling (CPU) | CPU | ~80 ms | ~65% |
| Tensor Construction | CPU→GPU | ~13.5 ms | ~11% |
| GPU Forward/Backward | GPU | ~28 ms | ~23% |
| Optimizer | GPU | ~0.2 ms | <1% |

**Negative Sampling** internally consists of (Phase 2 data, relative to neg sampling time):
- B1 Random number generation: 42% → ~33.6 ms
- B2 Candidate build: 23% → ~18.4 ms
- B3 Collision check: 14% → ~11.2 ms
- B4 Retry: 0.4% → ~0.3 ms
- B5 Output build: 2.6% → ~2.1 ms

**Tensor Construction** breakdown (Phase 7 Step 1):
- T4 (neg tensor build): 11.5 ms (84.8%) — essentially `torch.randint` for 750k IDs
- T5 (CPU→GPU transfer): 1.15 ms
- T1-T3 (positive sample extraction & conversion): ~0.9 ms

## 2. GPU Migration Feasibility Matrix

| Sub-stage | Current Cost | GPU Feasibility | API Support | Key Challenge |
|-----------|--------------|-----------------|-------------|---------------|
| B1 Random generation | 33.6 ms | ✅ Highly feasible | `torch.randint` on GPU | None |
| B2 Candidate build | 18.4 ms | ✅ Feasible | Indexing / `gather` | May merge into single step |
| B3 Collision check | 11.2 ms | ⚠️ Partially feasible | `torch.isin`, `torch.unique` | Replace while-loop with vectorized mask + oversampling |
| B4 Retry | 0.3 ms | ❌ Not recommended | – | Serial retry logic not GPU-friendly; can be eliminated with oversampling |
| B5 Output build | 2.1 ms | ✅ Trivial | Tensor concatenation | – |
| T4 Neg tensor build | 11.5 ms | ✅ Eliminated | Already on GPU if negs generated there | – |
| T5 GPU transfer | 1.15 ms | ✅ Eliminated | No transfer needed | – |
| T1-T3 Pos tensor | 0.9 ms | ✅ Partially | Can move to GPU after batch composition | Minor overhead, may keep on CPU |

**Conclusion**: All major CPU bottlenecks (B1-B3, T4, T5) can be fully or partially migrated to GPU using standard PyTorch operations, without requiring custom CUDA kernels.

## 3. Proposed GPU Negative Sampling Algorithm (Vectorized)

### 3.1 Core Idea
For a batch of positive triples, generate a large pool of candidate negative entities on the GPU, filter out true entities using vectorized masking, and select the required number.

### 3.2 Pseudo-code
```python
def gpu_negative_sampling(pos_heads, pos_rels, pos_tails, neg_num, n_entities):
    batch_size = pos_heads.size(0)
    oversample = int(batch_size * neg_num * 1.2)   # 20% oversampling
    # Step 1: Generate random candidate entities (uniform)
    candidates = torch.randint(0, n_entities, (oversample,), device='cuda')
    
    # Step 2: Determine which true tail/head need to be excluded
    # For tail corruption: exclude pos_tails (if head corruption, exclude pos_heads)
    # Build a mask: candidates not in pos_tails
    mask = ~torch.isin(candidates, pos_tails)   # [oversample]
    valid = candidates[mask]
    
    # Step 3: If not enough, resample (or increase oversample factor)
    while valid.size(0) < batch_size * neg_num:
        extra = torch.randint(0, n_entities, (oversample,), device='cuda')
        extra_mask = ~torch.isin(extra, pos_tails)
        valid = torch.cat([valid, extra[extra_mask]])
    return valid[:batch_size * neg_num]
```

### 3.3 Performance Estimate
Based on micro-benchmarks: generating 750k random ints + masking on GPU takes < 1 ms (vs. ~11.5 ms CPU). Filtering with `torch.isin` scales linearly with candidate count, but with 20% oversampling it's still negligible. Expected total GPU sampling time < 2 ms, eliminating ~65 ms of CPU work.

## 4. Integration with CBP Framework

CBP provides the batch composition; GPU sampling receives the sorted/packed batch of positive triples. Cost table is not needed on GPU because the sampling cost is now constant (all operations GPU-accelerated). However, we may optionally use entity degree to bias sampling (e.g., avoid highly popular negatives) in future extensions.

The pipeline becomes:
```
CBP Scheduler (CPU, once per epoch)
    ↓
Batch of positive triples (already on CPU or moved to GPU)
    ↓
GPU Negative Sampling (torch ops)
    ↓
Forward/Backward (GPU)
```

## 5. Expected Benefits

- **Eliminate CPU negative sampling** (~65 ms) and tensor construction (~13.5 ms) → step time reduction of ~60%.
- **Reduce step time variance** further: GPU sampling has near-constant execution time.
- **Keep all data on GPU** after initial transfer of positive triples, minimizing PCIe communication.
- **Enhance scalability**: GPU sampling scales with larger batch sizes / neg_num much better than CPU.

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| GPU memory increase | Oversampling buffer is temporary; additional memory ~ (batch_size * neg_num * 1.2 * 4 bytes) ≈ 3.6 MB for 750k, negligible. |
| Collision rate high for Hub entities | CBP already redistributes Hubs; even if collision rate spikes, oversampling factor can be dynamically adjusted. |
| `torch.isin` not available in older PyTorch | MuKG uses PyTorch 1.11+; `torch.isin` introduced in 1.8. Safe. |
| Oversampling not sufficient for extreme hubs | Fallback to CPU retry for rare cases, or increase factor to 2.0 (still cheap). |

## 7. Next Steps

- **Step 3**: Literature Alignment & Existing GPU Sampling Survey (to ensure novelty).
- **Step 4**: Detailed GPU Sampling Algorithm Design (select among fully GPU, hybrid, etc.).
- **Step 5**: Prototype implementation and benchmark against current CPU baseline.
- **Step 6**: Full integration into Runtime Framework and Phase 8 execution.

**Prepared by**: MuKG Research Agent  
**Approval**: pending Step 3 design.