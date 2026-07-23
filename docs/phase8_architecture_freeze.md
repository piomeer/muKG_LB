# Phase 8 Architecture Freeze

**Date**: 2026-07-23  
**Purpose**: Finalize module interfaces and file modification plan before implementation.

## 1. Module Reuse & New Additions

| Module | Reuse? | Current File | Action |
|--------|--------|--------------|--------|
| CostModel | **Reuse** | `src/py/load/cost_model.py` | No changes needed. `build_cost_table()` returns dict of `triple -> cost`. |
| Scheduler | **Reuse** | `src/py/load/schedulers.py` | `Scheduler(sorter, packer, batch_size)` works as before. |
| BatchProvider | **Reuse** | `src/py/load/batch_provider.py` | `BatchProvider(scheduler, cost_table).iterate(triples)` produces batches of positive triples. |
| GPUNegativeSampler | **New** | `src/py/load/gpu_sampler.py` | Implements GPU-based negative sampling, replacing `generate_neg_triples_fast()`. |
| Training Loop | **Modify** | `src/py/experiments/run_cbp_evaluation.py` | Replace CPU neg‑sampling call with `GPUNegativeSampler.generate()`. |
| Runtime Policy | **Stub** | `src/py/load/runtime_policy.py` (new) | Initially only `GPUOnly` policy; interface defined for future expansion. |

## 2. GPUNegativeSampler Interface (New Module)

```python
class GPUNegativeSampler:
    def __init__(self, n_entities: int, neg_num: int, oversample_factor: float = 1.5, device: str = 'cuda'):
        """
        n_entities: total number of entities
        neg_num: number of negatives per positive triple
        oversample_factor: multiplier to generate extra candidates to handle collisions
        """
        ...

    def generate(self, batch_triples: List[Tuple[int,int,int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: batch of positive triples (list of (h,r,t))
        Output: neg_heads (B*neg_num,), neg_tails (B*neg_num,) on GPU
        This should replace the output of generate_neg_triples_fast().
        """
        ...
```

**Implementation Sketch**:
- Convert `batch_triples` to `pos_heads`, `pos_tails` tensors on GPU.
- Generate `candidates = torch.randint(0, n_entities, (total_needed * oversample_factor,), device=device)`.
- Filter with `~torch.isin(candidates, pos_tails)`.
- If not enough, repeat (but with 1.5x factor and CBP, should rarely need retry).
- Reshape to required output shape.

## 3. Integration Point in Training Loop

In `run_cbp_evaluation.py`, current training step (simplified):
```python
# ... inside train_epoch_with_provider
for batch_triples in provider.iterate(triples):
    # old CPU path:
    neg_heads, neg_tails = generate_neg_triples_fast(batch_triples, ...)
    # new GPU path:
    neg_heads, neg_tails = gpu_sampler.generate(batch_triples)
    # then forward/backward
```

**Required changes**:
- Add argument `--use_gpu_sampling` (bool) to evaluation script.
- Instantiate `GPUNegativeSampler` before training loop.
- Conditionally call GPU or CPU sampling.

## 4. File Modification List

| File | Action |
|------|--------|
| `src/py/load/gpu_sampler.py` | **Create** (new module) |
| `src/py/load/runtime_policy.py` | **Create** (stub for future) |
| `src/py/experiments/run_cbp_evaluation.py` | **Modify** (integrate sampler, add CLI flag) |
| `src/py/experiments/validate_gpu_sampler.py` | **Create** (validation script for Step 1) |
| `src/py/experiments/run_full_benchmark.py` | **Create** (later, for Step 3) |

## 5. Validation Criteria (Step 1)

Before full integration, we will run a small validation:
- Take one batch of 5000 triples.
- Generate negatives with both CPU and GPU samplers.
- Compare timing and output format (not exact values, as randomness differs).
- Confirm GPU sampler produces tensors of correct shape on GPU.
- Output: `output/results/gpu_sampler_validation.csv` with timing stats.

## 6. Risk Points
- `torch.isin` with large `candidates` and `pos_tails` may be memory‑heavy; oversample factor may need tuning.
- First‑time CUDA kernel launch overhead may skew Step 1 validation; we will run warmup.
- CBP cost model was trained for CPU sampling; GPU cost is flat, so scheduler still works but cost semantics change (acceptable per spec).

**Freeze approved** – proceed to Step 1 implementation.
