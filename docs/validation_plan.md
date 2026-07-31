# Validation Plan: Statistical Verification & Sensitivity Experiments

**Date**: 2026-07-31  
**Status**: Planning (no code modifications executed)  
**Purpose**: Define additional experiments needed to strengthen paper claims before submission.  
**Hardware Target**: server_node4 (RTX 3070 8GB VRAM, 32GB RAM, CUDA 11.3, PyTorch 1.10.2)

---

## A. Statistical Validation

### A.1 GPU Runtime Repeat Runs (P0 — Required)

| Item | Detail |
|------|--------|
| **Purpose** | Verify that GPU Runtime's 5.7× epoch acceleration and 142× neg_std reduction are statistically significant, not due to single-run noise |
| **Configurations** | `GPU` (Random+Chunk, GPU v2 sampler), `CBP+GPU` (Cost+FFD, GPU v2 sampler) |
| **Runs per config** | 5 independent runs (different random seeds: 42, 123, 456, 789, 1024) |
| **Epochs per run** | 5 epochs (matching Phase 9 Step 2 benchmark) |
| **Script name** | `src/py/experiments/phase10_step2_statistical_repeats.py` |
| **Output format** | CSV + mirrored MD |
| **Output files** | `output/results/phase10_step2/GPU/run_*/summary.csv` + `summary.md`; `output/results/phase10_step2/CBP+GPU/run_*/summary.csv` + `summary.md` |
| **Metrics to collect** | avg_epoch_time_s, neg_time_mean_ms, neg_time_std_ms, step_time_mean_ms, step_time_std_ms, final_loss, mrr, hits10 |
| **Aggregated stats** | Compute mean ± std across 5 runs for each metric; report 95% confidence interval (t-distribution, df=4) |
| **Estimated time** | ~50 min (2 configs × 5 runs × [5 epochs × ~4.5s/epoch + evaluation]) |
| **Priority** | **P0 — Must do before paper submission** |

**Aggregated Output Table (to be populated)**:

| Config | Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean ± Std | 95% CI |
|--------|--------|-------|-------|-------|-------|-------|-----------|--------|
| GPU | avg_epoch_time_s | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| GPU | neg_std_ms | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| GPU | mrr (final) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| CBP+GPU | avg_epoch_time_s | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| CBP+GPU | neg_std_ms | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| CBP+GPU | mrr (final) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

---

### A.2 CPU Baseline & CBP Repeat Runs (P1 — Recommended)

| Item | Detail |
|------|--------|
| **Purpose** | Verify that CBP's 78% neg_std reduction at batch_size=1000 (Phase 6) is reproducible; also provide statistical baseline for CPU path |
| **Configurations** | `BL` (Random+Chunk, CPU), `CBP` (Cost+FFD, CPU) at batch_size=1000, neg_num=150 |
| **Runs per config** | 3 independent runs (seeds: 42, 123, 456) |
| **Epochs per run** | 3 epochs (sufficient for variance analysis; block-wise neg_std measurement per batch) |
| **Script name** | `src/py/experiments/phase10_step2_cpu_statistical_repeats.py` |
| **Output format** | CSV + mirrored MD |
| **Output files** | `output/results/phase10_step2_cpu/BL/run_*/summary.csv` + `summary.md`; `output/results/phase10_step2_cpu/CBP/run_*/summary.csv` + `summary.md` |
| **Metrics to collect** | avg_neg_time_per_batch_ms, neg_std_per_batch_ms, step_time_mean_ms, epoch_time_s |
| **Aggregated stats** | Mean ± std across 3 runs; report 95% CI (t-distribution, df=2) |
| **Rationale** | batch_size=1000 was where CBP showed strongest effect (Phase 6). Re-running at batch_size=5000 would likely show no effect (as Phase 9 Step 4.5 already demonstrated). |
| **Estimated time** | ~8 min (2 configs × 3 runs × [3 epochs × ~25s/epoch]) |
| **Priority** | **P1 — Recommended for robust CBP claims** |

---

### A.3 Cost Model Bootstrap Validation (P1 — Recommended)

| Item | Detail |
|------|--------|
| **Purpose** | Compute bootstrap confidence interval for R²=0.9008 to quantify model stability |
| **Method** | Bootstrap resampling (n=1000 iterations) of the (candidate_size, measured_cost) pairs from Phase 5.5 data; fit linear regression per bootstrap sample; report 2.5th/97.5th percentiles |
| **Script name** | `scripts/bootstrap_cost_model.py` |
| **Output format** | Console printout + report file |
| **Output files** | `output/results/cost_model_bootstrap.md` |
| **Expected output** | R² 95% CI ≈ [0.87, 0.93] (estimated from typical linear regression with 455 data points) |
| **Runtime** | <1 min (no GPU needed, runs on CPU) |
| **Priority** | **P1 — Recommended for cost model credibility** |

---

## B. Sensitivity Experiments

### B.1 Batch Size Sensitivity (P0 — Required)

| Item | Detail |
|------|--------|
| **Purpose** | Verify that GPU Runtime's acceleration advantage holds across different batch sizes (including memory-bound scenarios) |
| **Independent variable** | batch_size: 1000, 3000, 5000, 8000 (8000 may OOM on 8GB VRAM; skip if OOM) |
| **Fixed parameters** | neg_num=150, GPU v2 sampler, Random+Chunk scheduler |
| **Epochs per config** | 3 epochs (sufficient for timing measurement) |
| **Script name** | `src/py/experiments/phase10_step2_sensitivity_batchsize.py` |
| **Output format** | CSV + mirrored MD |
| **Output files** | `output/results/phase10_step2_sensitivity/batch_size/bs_*/summary.csv` + `summary.md` |
| **Metrics to collect** | avg_epoch_time_s, neg_time_mean_ms, epoch_time_s, gpu_mem_mb |
| **Expected finding** | GPU acceleration should hold for all batch sizes; CBP's variance benefit should increase at smaller batch sizes (more batches per epoch) |
| **Estimated time** | ~5 min (4 configs × 3 epochs × ~5-20s/epoch, depending on batch size) |
| **Priority** | **P0 — Must do** |

**Output Table Template**:

| batch_size | config | epoch_time_s | neg_time_ms | gpu_mem_mb | batches_per_epoch |
|-----------|--------|-------------|------------|------------|-------------------|
| 1000 | GPU | [ ] | [ ] | [ ] | 273 |
| 1000 | CBP+GPU | [ ] | [ ] | [ ] | 273 |
| 3000 | GPU | [ ] | [ ] | [ ] | 91 |
| 3000 | CBP+GPU | [ ] | [ ] | [ ] | 91 |
| 5000 | GPU | [ ] | [ ] | [ ] | 55 |
| 5000 | CBP+GPU | [ ] | [ ] | [ ] | 55 |
| 8000 | GPU | [ ] | [ ] | [ ] | 35 (if no OOM) |
| 8000 | CBP+GPU | [ ] | [ ] | [ ] | 35 (if no OOM) |

---

### B.2 Negative Number Sensitivity (P0 — Required)

| Item | Detail |
|------|--------|
| **Purpose** | Verify that GPU Runtime's 142× neg_std compression is independent of neg_num (the dominant factor for variance) |
| **Independent variable** | neg_num: 10, 25, 50, 100, 150 |
| **Fixed parameters** | batch_size=5000, GPU v2 sampler, Random+Chunk scheduler |
| **Epochs per config** | 3 epochs |
| **Script name** | `src/py/experiments/phase10_step2_sensitivity_negnum.py` |
| **Output format** | CSV + mirrored MD |
| **Output files** | `output/results/phase10_step2_sensitivity/neg_num/nn_*/summary.csv` + `summary.md` |
| **Metrics to collect** | avg_epoch_time_s, neg_time_mean_ms, neg_time_std_ms, step_time_mean_ms |
| **Expected finding** | CPU neg_std should increase ∝ neg_num; GPU neg_std should remain flat (~0.2ms) regardless of neg_num |
| **Estimated time** | ~5 min (5 configs × 3 epochs × ~5-20s/epoch) |
| **Priority** | **P0 — Must do** |

**Output Table Template**:

| neg_num | config | epoch_time_s | neg_time_ms | neg_std_ms | step_time_ms |
|---------|--------|-------------|------------|-----------|-------------|
| 10 | GPU | [ ] | [ ] | [ ] | [ ] |
| 10 | CPU (BL) | [ ] | [ ] | [ ] | [ ] |
| 25 | GPU | [ ] | [ ] | [ ] | [ ] |
| 25 | CPU (BL) | [ ] | [ ] | [ ] | [ ] |
| 50 | GPU | [ ] | [ ] | [ ] | [ ] |
| 50 | CPU (BL) | [ ] | [ ] | [ ] | [ ] |
| 100 | GPU | [ ] | [ ] | [ ] | [ ] |
| 100 | CPU (BL) | [ ] | [ ] | [ ] | [ ] |
| 150 | GPU | [ ] | [ ] | [ ] | [ ] |
| 150 | CPU (BL) | [ ] | [ ] | [ ] | [ ] |

---

### B.3 Model Sensitivity (P1 — Recommended)

| Item | Detail |
|------|--------|
| **Purpose** | Verify that the Unified Runtime Framework generalizes to KGE models beyond TransE |
| **Models to test** | RotatE (complex-valued embedding, higher compute cost) and ConvE (2D convolution, different batch processing pattern) |
| **Fixed parameters** | batch_size=5000 (or reduced if OOM), neg_num=150, GPU v2 sampler |
| **Epochs per model** | 3 epochs (timing only; full convergence not needed) |
| **Script name** | `src/py/experiments/phase10_step2_sensitivity_model.py` |
| **Output format** | CSV + mirrored MD |
| **Output files** | `output/results/phase10_step2_sensitivity/model/rotate/summary.csv` + `summary.md`; `output/results/phase10_step2_sensitivity/model/conve/summary.csv` + `summary.md` |
| **Metrics to collect** | epoch_time_s, neg_time_ms, neg_std_ms, gpu_mem_mb |
| **Note** | RotatE and ConvE are already implemented in `src/torch/kge_models/`; need to load their configs from `src/py/args_kge/rotate_args.json` and `src/py/args_kge/conve_args.json` |
| **Estimated time** | ~5 min (2 models × 3 epochs × ~5-30s/epoch, depending on model complexity) |
| **Priority** | **P1 — Recommended for generalizability claims** |

---

### B.4 Dataset Sensitivity (P2 — Optional)

| Item | Detail |
|------|--------|
| **Purpose** | Verify that GPU Runtime's acceleration holds on a different dataset with different topological properties |
| **Dataset** | WN18RR (86,835 entities, 11 relations; different connectivity pattern from FB15k-237) |
| **Fixed parameters** | batch_size=5000, neg_num=150, TransE model, GPU v2 sampler |
| **Epochs** | 3 epochs (timing only) |
| **Script name** | `src/py/experiments/phase10_step2_sensitivity_dataset.py` |
| **Output format** | CSV + mirrored MD |
| **Output files** | `output/results/phase10_step2_sensitivity/dataset/wn18rr/summary.csv` + `summary.md` |
| **Metrics to collect** | epoch_time_s, neg_time_ms, neg_std_ms, dataset_stats (entities, relations, triples) |
| **Estimated time** | ~3 min (1 dataset × 3 epochs × ~30-60s/epoch, depending on entity count) |
| **Priority** | **P2 — Nice to have; low-hanging fruit if WN18RR data is already available** |

**Note**: WN18RR data needs to be downloaded first (requires internet). If unavailable on node4, skip this experiment.

---

## C. Summary Timeline

| Priority | Experiment | Estimated Time | Description |
|----------|-----------|---------------|-------------|
| **P0** | A.1 GPU Repeat Runs | ~50 min | 5× repeats for GPU and CBP+GPU |
| **P0** | B.1 Batch Size Sensitivity | ~5 min | 4 batch sizes × GPU configs |
| **P0** | B.2 Neg Num Sensitivity | ~5 min | 5 neg_nums × GPU+CPU |
| **P1** | A.2 CPU Repeat Runs | ~8 min | 3× repeats for BL and CBP at batch_size=1000 |
| **P1** | A.3 Cost Model Bootstrap | <1 min | R² bootstrap CI (CPU only) |
| **P1** | B.3 Model Sensitivity | ~5 min | RotatE + ConvE timing |
| **P2** | B.4 Dataset Sensitivity | ~3 min | WN18RR timing |
| **Total P0** | | ~60 min | Core statistical + sensitivity |
| **Total P0+P1** | | ~74 min | All recommended experiments |
| **Grand Total** | | ~77 min | All experiments |

---

## D. Execution Protocol

1. **Sync code to node4**: `rsync -avz --exclude='.git' /home/hma/muKG_LB/ user@node4:~/muKG_LB/`
2. **Run experiments in order**: P0 first (A.1 → B.1 → B.2); if time permits, P1 next
3. **Collect all output MD/CSV files**: Each run writes to `output/results/phase10_step2_*/` or `output/results/phase10_step2_sensitivity/`
4. **Copy results back to pc-cluster**: `rsync -avz user@node4:~/muKG_LB/output/results/ /home/hma/muKG_LB/output/results/`
5. **Update evidence matrix** (`docs/evidence_matrix.md`) with confirmed P-values/CIs
6. **Update paper outline** (`docs/paper_outline.md`) with sensitivity analysis results (Section 4.6)

---

## E. Expected Contributions After Validation

| Contribution | Before Validation | After Validation |
|-------------|------------------|-----------------|
| **C1 — GPU Runtime** | Point estimates (5.7× speedup, 142× var reduction) | Mean ± 95% CI across 5 runs; batch_size and neg_num sensitivity curves |
| **C2 — Unified Runtime** | Architecture validated for TransE | Architecture validated for RotatE + ConvE (cross-model) |
| **C3 — Cost Model** | R²=0.9008 (point estimate) | R² 95% CI via bootstrap |
| **C4 — CBP** | 78% neg_std reduction at batch_size=1000 (Phase 6, single run) | 78% ± CI across 3 runs at batch_size=1000; batch_size sensitivity curve showing benefit decay as batch_size increases |