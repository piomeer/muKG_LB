# Evidence Audit Part 1 — Paper Claim Inventory

**Date**: 2026-07-31  
**Audit Status**: Complete (read-only, based on frozen story line and existing experiment assets)  
**Based on**: `docs/paper_story_freeze.md`, `docs/paper_outline.md`, `docs/evidence_matrix.md`, `docs/baseline_freeze.md`, `paper/draft/method.md`, `paper_assets/experiment_summary.md`, and all verified files under `output/results/` and `paper_assets/`

---

## Overview

This document inventories all paper claims across the four contributions (C1–C4), mapping each claim to its supporting experiments, figures/tables, scripts, and CSV/summary data files. All paths are verified against the actual file system as of 2026-07-31. Claims are ordered by contribution strength per `paper_story_freeze.md` §Q3.

---

## C1: GPU Runtime — Fully Vectorized GPU Negative Sampling

**Contribution Summary** (from `paper_story_freeze.md` §Q3):
GPU 全向量化负采样器将负采样时间从 596ms 压缩到 3.0ms（198×），epoch 时间从 37s 降到 4.78s（7.7×），并将负采样标准差从 28.5ms 压到 0.2ms（142×）。

### Claim Inventory

| # | Claim | Experiment | Figure/Table | Script | CSV Data |
|---|-------|-----------|-------------|--------|----------|
| **C1.1** | GPU negative sampling achieves 198× speedup over CPU (596ms → 3.0ms per batch) | Phase 8 Step 2 (Unified Runtime Validation, 5 epochs); Phase 9 Step 2 (Main Benchmark, 5 epochs, 4 configs) | Fig.4 (`paper_assets/figures/fig4_gpu_runtime_trace.pdf`); Fig.5 (`paper_assets/figures/fig5_benchmark_bars.pdf`); Table 2 | `src/py/experiments/run_unified_runtime_validation.py`; `src/py/experiments/phase9_step2_benchmark.py`; `src/py/load/gpu_sampler.py` | `output/results/unified_runtime/epoch_summary_GPU.csv`; `output/results/phase9_step2/GPU/summary.csv`; `output/results/phase9_step2/summary.csv` |
| **C1.2** | GPU Runtime delivers 5.7× end-to-end epoch acceleration (CPU 25.1s → GPU 4.4s) at batch_size=5000, neg_num=150 | Phase 9 Step 2 (Main Benchmark, 5 epochs × 4 configs: BL/CBP/GPU/CBP+GPU) | Fig.5 (`paper_assets/figures/fig5_benchmark_bars.pdf`); Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/summary.csv`; `output/results/phase9_step2/GPU/summary.csv`; `output/results/phase9_step2/BL/summary.csv` |
| **C1.3** | GPU compresses neg-sampling standard deviation by 142× (28.5ms → 0.2ms), eliminating the dominant variance source in KGE training | Phase 9 Step 3 (10-epoch Ablation, 4 configs); Phase 9 Step 4.5 (CPU Neg-Sampling Variance Isolation) | Fig.6 (`paper_assets/figures/fig6_ablation_variance.pdf`, left panel: neg_std, right panel: step_std); Table 3; Table 4 | `src/py/experiments/phase9_step3_ablation.py`; `src/py/experiments/phase9_step4_5_cpu_variance.py` | `output/results/phase9_step3/GPU/summary.csv`; `output/results/phase9_step3/BL/summary.csv`; `output/results/phase9_step4_5/variance_summary.csv`; `output/results/phase9_step4_5/neg_sampling_variance.csv` |
| **C1.4** | GPU step time accelerates 8.5× (674ms → 79.7ms) | Phase 8 Step 2; Phase 9 Step 2 | Table 2; Fig.5 | Same as C1.1 | `output/results/phase9_step2/summary.csv` |
| **C1.5** | GPU tail-only negative sampling achieves comparable convergence to CPU Bernoulli sampling (semantic alignment verified) | Phase 9 Step 1 (Semantic Alignment: CPU original vs GPU v2, 2 epochs) | — (alignment report, not a paper figure) | `src/py/experiments/phase9_step1_alignment.py` | `docs/semantic_alignment_report.md`; `output/results/phase9_step1/results.md` |
| **C1.6** | GPU Sampler memory overhead is negligible (~2 MB additional VRAM on 8GB RTX 3070) | Phase 8 Step 1 (GPU Sampler prototype validation) | — (inline text in Method §3.4.3) | `src/py/experiments/validate_gpu_sampler.py`; `src/py/experiments/validate_gpu_sampler_full.py`; `src/py/load/gpu_sampler.py` | `output/results/gpu_sampler/validation.md` |
| **C1.7** | GPU acceleration persists across all 10 epochs (neg-sampling time stable at 2.8ms avg) | Phase 9 Step 3 (10-epoch ablation, all 4 configs) | Fig.6; Table 3 | `src/py/experiments/phase9_step3_ablation.py` | `output/results/phase9_step3/GPU/summary.csv`; `output/results/phase9_step3/CBP+GPU/summary.csv` |
| **C1.8** | GPU path holds comparable MRR/Hits@10 to CPU baseline (5-epoch relative: GPU MRR 0.0132 vs BL 0.0136) | Phase 9 Step 2 (5 epochs) | Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/GPU/summary.csv`; `output/results/phase9_step2/BL/summary.csv` |
| **C1.9** | GPU Runtime eliminates the bottleneck shift: post-GPU, neg-sampling drops from 65% to <5% of step time | Phase 6 (Profiling) + Phase 8 Step 2 | Fig.1 (`paper_assets/figures/fig1_profiling_breakdown.pdf`); Table 5 (⚠️ data exists but table not yet formatted) | `analyze_profiling.py` | `output/results/training_time_breakdown.md`; `output/results/unified_runtime/epoch_summary_GPU.md` |

**Coverage Notes**:
- C1.5 (semantic alignment): No figure in paper assets; documented in `docs/semantic_alignment_report.md`.
- C1.6 (memory overhead): Qualitative claim in method text; validation data in `output/results/gpu_sampler/validation.md`.
- C1.9 (bottleneck shift): Data exists in profiling outputs; Table 5 not yet formatted (`evidence_matrix.md` marks as ⚠️).

---

## C2: Unified Runtime Framework — Cost-aware Scheduling + GPU Execution Architecture

**Contribution Summary** (from `paper_story_freeze.md` §Q3):
定义了 FeatureExtractor → CostModel(纯函数) → Scheduler(Sort+Pack组合) → BatchProvider(Adapter) 四层解耦的运行时架构，并通过 GPUNegativeSampler 将 GPU 执行无缝嵌入。

### Claim Inventory

| # | Claim | Experiment | Figure/Table | Script | CSV Data |
|---|-------|-----------|-------------|--------|----------|
| **C2.1** | The Unified Runtime Framework provides a four-layer decoupled architecture (FeatureExtractor → CostModel → Scheduler → BatchProvider → GPUNegativeSampler) that separates concerns and enables transparent CPU/GPU migration | Phase 7 Step 4–5 (Architecture design + Route C recommendation); Phase 8 Step 0 (Architecture freeze) | — (architecture diagram in Method §3.5, to be drawn manually) | N/A (design documents) | `docs/gpu_runtime_architecture.md`; `docs/phase8_architecture_freeze.md`; `docs/runtime_framework_spec.md`; `paper/draft/method.md` §3.5 |
| **C2.2** | All four configurations (BL/CBP/GPU/CBP+GPU) are validated in a unified benchmark, confirming modular composability | Phase 9 Step 2 (Main Benchmark: 4 configs × 5 epochs) | Fig.5 (`paper_assets/figures/fig5_benchmark_bars.pdf`); Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/summary.csv` |
| **C2.3** | The Scheduler (Sort+Pack) and BatchProvider modules are reused identically across CPU and GPU execution paths, with only the Sampler backend swapped | Phase 8 Step 2 (Unified Runtime Validation); Phase 9 Step 2 | Table 1 (Experimental Configuration from `docs/baseline_freeze.md`) | `src/py/experiments/run_unified_runtime_validation.py`; `src/py/load/schedulers.py`; `src/py/load/batch_provider.py` | `output/results/unified_runtime/unified_runtime_validation.md`; `output/results/integration_validation/validation_summary.json` |
| **C2.4** | The CostModel is implemented as a pure function (pre-computed lookup table) with zero runtime overhead | Phase 5.5 / Phase 6 (Cost model fitting) | Fig.2 (`paper_assets/figures/fig2_cost_model_corr.pdf`) | `scripts/fit_cost_model.py`; `src/py/load/cost_model.py` | `output/results/cost_table.npy`; `output/results/cost_model_summary.md` |
| **C2.5** | The BatchProvider adapter wraps scheduled batches into the PyTorch DataLoader interface, enabling drop-in replacement of existing loaders | Phase 8 Step 0; Phase 9 Step 2 | — (inline in Method §3.5.2) | `src/py/load/batch_provider.py` | `output/results/integration_validation/batch_mapping.md`; `output/results/integration_validation/batch_composition.md` |
| **C2.6** | Framework overhead (CostSorter + FFDPacker) is ~0.5ms per epoch, negligible vs 25s epoch time | Phase 9 Step 2 (CBP vs BL epoch comparison on CPU: 25.3s vs 25.1s) | Table 5 (⚠️ data exists but table not yet formatted) | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/CBP/summary.csv`; `output/results/phase9_step2/BL/summary.csv` |

**Coverage Notes**:
- C2.1 (architecture diagram): No figure in paper assets yet (to be drawn manually per `paper_outline.md`).
- C2.6 (framework overhead): Quantitative data exists in Phase 9 Step 2; Table 5 not yet formatted.
- The framework reuse table in `docs/phase8_architecture_freeze.md` documents which modules are reused vs newly created.

---

## C3: Offline Runtime Cost Model — Topological Feature → Expected Cost Mapping

**Contribution Summary** (from `paper_story_freeze.md` §Q3):
将每个实体的 `candidate_size`（邻居池大小）映射为预期负采样成本 `E[retry] × B3_const`。该模型实现了 R²=0.9008 的预测精度（Phase 5.5/Phase 7 Step 3）。

### Claim Inventory

| # | Claim | Experiment | Figure/Table | Script | CSV Data |
|---|-------|-----------|-------------|--------|----------|
| **C3.1** | The offline cost model achieves R²=0.9008 prediction accuracy for per-entity neg-sampling cost based on candidate_size alone | Phase 5.5 / Phase 6 (Cost model formula validation via linear regression on 455 sampled entities) | Fig.2 (`paper_assets/figures/fig2_cost_model_corr.pdf`) | `scripts/fit_cost_model.py` | `docs/cost_model.md`; `output/results/cost_table.npy`; `output/results/cost_model_summary.md`; `output/results/cost_model_data.md` |
| **C3.2** | Runtime correlation between predicted cost and measured neg-sampling time is r=0.71 (CBP attribution experiment) | Phase 6 (Runtime Attribution: Weight vs Neg Sampling correlation) | — (inline in Method §3.2.3; attribution report) | `src/py/experiments/runtime_attribution.py` | `output/results/runtime_attribution/runtime_attribution.csv`; `output/results/runtime_attribution/attribution_interpretation.md` |
| **C3.3** | The cost model is a pure function of static KG statistics — no online profiling, no model-specific features, reusable across KGE models | Phase 5.5 (offline fitting); Phase 7 Step 3 (GPU cost microbench) | — (inline in Method §3.2.3) | `scripts/fit_cost_model.py`; `src/py/load/cost_model.py`; `src/py/load/features.py` | `output/results/cost_table.npy`; `output/results/entity_features.npz` |
| **C3.4** | GPU cost model break-even point: N* = 264k negative samples (below which CPU is faster due to CUDA launch overhead) | Phase 7 Step 3 (GPU cost microbench) | — (inline in Method; benchmark report) | `src/py/experiments/gpu_cost_microbench.py` (referenced but not visible in project tree — may need confirmation) | `output/results/gpu_cost_model/benchmark.csv`; `output/results/gpu_cost_model/benchmark.md` |
| **C3.5** | Entity degree correlates with measured sampling cost at R=0.816 (empirical validation of candidate_size intuition) | Phase 6 (Profiling + Attribution) | — (mentioned in Method §3.1.2) | `analyze_profiling.py`; `scripts/validate_b1_correlation.py` | `output/results/hub_analysis.md`; `output/results/negative_sampling_breakdown.md` |
| **C3.6** | Cost model lookup is O(1) per query and requires ~116 KB memory for 14,505 entities (64-bit precision) | Phase 5.5 (static analysis) | — (inline in Method §3.2.3) | `src/py/load/cost_model.py` | `output/results/cost_table.npy` |

**Coverage Notes**:
- C3.1 (R²=0.9008): Single-point linear regression; no 95% confidence interval or bootstrapping reported (`evidence_matrix.md` gap analysis).
- C3.2 (r=0.71): Moderate runtime correlation; gap between offline prediction and online measurement acknowledged in `evidence_matrix.md`.
- C3.4 (GPU cost microbench): Script `gpu_cost_microbench.py` referenced in `paper_story_freeze.md` but path not confirmed in project tree — noted as "需确认".
- C3.5 (R=0.816): This correlation value is mentioned in `paper/draft/method.md` §3.1.2; source data under `output/results/hub_analysis.md`.

---

## C4: Cost-aware Batch Packing (CBP) — Pluggable Sort+Pack Scheduling

**Contribution Summary** (from `paper_story_freeze.md` §Q3, §Q4):
在 CPU 路径上验证了代价感知调度的可行性（Phase 6 在 batch_size=1000 下将 neg_std 降低 78%），并为框架提供了可插拔的 Sort+Pack 策略接口。虽然在全训练循环（batch_size=5000）中边际收益被系统噪声稀释（Phase 9 Step 4.5: std 仅降低 8.4%），但 CBP 的实验过程是推动 GPU 迁移的关键动机。

**⚠️ Conditional Strength Note**: CBP is strong only at batch_size=1000 (Phase 6: 78% reduction). At standard training batch_size=5000, the effect is marginal (Phase 9 Step 4.5: 8.4% reduction). Per `paper_story_freeze.md`, CBP is positioned as "the critical intermediate step motivating GPU migration," not as a standalone highlight contribution.

### Claim Inventory

| # | Claim | Experiment | Figure/Table | Script | CSV Data |
|---|-------|-----------|-------------|--------|----------|
| **C4.1** | CBP reduces neg-sampling std by 78% at batch_size=1000 (15.5ms → 3.4ms, 275 batches/epoch) — demonstrates theoretical correctness of cost-aware scheduling | Phase 6 (Runtime Attribution, batch_size=1000, CBP vs BL) | Fig.3 (`paper_assets/figures/fig3_batch_cost_distribution.pdf`) | `src/py/experiments/runtime_attribution.py` | `output/results/runtime_attribution/runtime_attribution.csv`; `output/results/runtime_attribution/runtime_attribution.md` |
| **C4.2** | CBP provides a pluggable Sort+Pack strategy interface: {RandomSorter, CostSorter} × {ChunkPacker, FFDPacker} = 4 combinations | Phase 8 Step 0 (Architecture freeze); Phase 9 Step 2 (all 4 combinations benchmarked) | Table 1 (Experimental Configuration) | `src/py/load/schedulers.py`; `src/py/load/cost_model.py`; `src/py/load/features.py` | `output/results/phase9_step2/summary.csv` |
| **C4.3** | CBP's neg-sampling variance reduction is marginal (8.4%) at batch_size=5000 (55 batches/epoch) due to system noise dominance (Python GIL, OS jitter, tensor construction variance) | Phase 9 Step 4.5 (CPU Neg-Sampling Variance Isolation: BL 29.5ms → CBP 27.0ms) | Fig.6 (`paper_assets/figures/fig6_ablation_variance.pdf`, left panel: neg_std); Table 4 | `src/py/experiments/phase9_step4_5_cpu_variance.py` | `output/results/phase9_step4_5/variance_summary.csv`; `output/results/phase9_step4_5/neg_sampling_variance.csv` |
| **C4.4** | At batch_size=5000 full training loop, CBP vs BL neg_std is nearly identical (both ~28.5ms), confirming CBP's scheduling granularity is insufficient at large batch sizes | Phase 9 Step 3 (10-epoch ablation: CBP vs BL on CPU, neg_std ~28.5ms for both) | Fig.6; Table 3 | `src/py/experiments/phase9_step3_ablation.py` | `output/results/phase9_step3/CBP/summary.csv`; `output/results/phase9_step3/BL/summary.csv` |
| **C4.5** | CBP on CPU shows slightly higher MRR than BL at 5 epochs (0.0150 vs 0.0136), suggesting scheduling has a positive impact on convergence quality | Phase 9 Step 2 (5 epochs, 4 configs) | Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/CBP/summary.csv`; `output/results/phase9_step2/BL/summary.csv` |
| **C4.6** | CBP+GPU on GPU path yields MRR comparable to GPU-only (0.0113 vs 0.0132 at 5 epochs), indicating CBP is a lightweight optional component on GPU — preserved for extensibility but not critical | Phase 9 Step 2 (5 epochs) | Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/CBP+GPU/summary.csv`; `output/results/phase9_step2/GPU/summary.csv` |
| **C4.7** | CBP reduces CV (coefficient of variation) of batch costs from 0.055 to 0.012 at batch_size=1000 | Phase 6 (Runtime Attribution) | Fig.3 (`paper_assets/figures/fig3_batch_cost_distribution.pdf`) | `src/py/experiments/runtime_attribution.py` | `output/results/runtime_attribution/runtime_attribution.csv` |

**Coverage Notes**:
- C4.1 (78% reduction): Single-run data; no statistical repeats reported (`evidence_matrix.md` gap analysis recommends 3× repeats at batch_size=1000).
- C4.3 (8.4% marginal): Isolated CPU variance experiment; clearly documented limitation.
- C4.5 (MRR advantage): 5-epoch relative comparison only; not a full convergence claim (200-sample evaluation subset).
- All C4 claims acknowledge the batch_size dependency (strong at 1000, marginal at 5000) per `paper_story_freeze.md` narrative positioning.

---

## Cross-Reference: Experiment Phase → Contribution Mapping

| Phase / Step | Experiment Name | C1 (GPU) | C2 (Framework) | C3 (Cost Model) | C4 (CBP) | Primary Script(s) |
|-------------|----------------|:---:|:---:|:---:|:---:|-------------------|
| Phase 5.5 | Cost Model Fitting | — | — | ✅ | ✅ | `scripts/fit_cost_model.py` |
| Phase 6 | Runtime Attribution (batch_size=1000) | — | — | ✅ | ✅ | `src/py/experiments/runtime_attribution.py` |
| Phase 7 Step 3 | GPU Cost Microbench | ✅ | — | ✅ | — | `src/py/experiments/gpu_cost_microbench.py` ⚠️ |
| Phase 7 Step 4–5 | Route C Architecture Recommendation | — | ✅ | — | — | N/A (design doc) |
| Phase 8 Step 0 | Architecture Freeze | — | ✅ | — | — | N/A (design doc) |
| Phase 8 Step 1 | GPU Sampler Prototype | ✅ | — | — | — | `src/py/experiments/validate_gpu_sampler.py`; `src/py/experiments/validate_gpu_sampler_full.py` |
| Phase 8 Step 2 | Unified Runtime Validation | ✅ | ✅ | — | — | `src/py/experiments/run_unified_runtime_validation.py` |
| Phase 9 Step 1 | Semantic Alignment | ✅ | — | — | — | `src/py/experiments/phase9_step1_alignment.py` |
| Phase 9 Step 2 | Main Benchmark (5 epochs × 4 configs) | ✅ | ✅ | — | ✅ | `src/py/experiments/phase9_step2_benchmark.py` |
| Phase 9 Step 3 | Ablation Study (10 epochs × 4 configs) | ✅ | ✅ | — | ✅ | `src/py/experiments/phase9_step3_ablation.py` |
| Phase 9 Step 4.5 | CPU Neg-Sampling Variance Isolation | — | — | — | ✅ | `src/py/experiments/phase9_step4_5_cpu_variance.py` |

⚠️ `gpu_cost_microbench.py`: Referenced in `paper_story_freeze.md` §Q4 (C3) as `src/py/experiments/gpu_cost_microbench.py`, but path not confirmed in current project tree listing. Flagged for verification.

---

## Figure-to-Claim Traceability Matrix

| Fig # | File | Primary Contribution | Supports Claims | Status |
|-------|------|---------------------|----------------|--------|
| Fig.1 | `paper_assets/figures/fig1_profiling_breakdown.pdf` | Foundational (motivation) | C1.9, C3.5 (profiling context) | ✅ Generated |
| Fig.2 | `paper_assets/figures/fig2_cost_model_corr.pdf` | C3 (Cost Model) | C3.1, C2.4 | ✅ Generated |
| Fig.3 | `paper_assets/figures/fig3_batch_cost_distribution.pdf` | C3 + C4 | C4.1, C4.7, C3.2 | ✅ Generated |
| Fig.4 | `paper_assets/figures/fig4_gpu_runtime_trace.pdf` | C1 (GPU Runtime) | C1.1, C1.3 | ✅ Generated |
| Fig.5 | `paper_assets/figures/fig5_benchmark_bars.pdf` | C1 (GPU Runtime) | C1.1, C1.2, C1.4, C2.2 | ✅ Generated |
| Fig.6 | `paper_assets/figures/fig6_ablation_variance.pdf` | C1 + C4 | C1.3, C4.3, C4.4 | ✅ Generated |

---

## Table-to-Claim Traceability Matrix

| Table # | Data Source | Primary Contribution | Supports Claims | Status |
|---------|------------|---------------------|----------------|--------|
| Table 1 | `docs/baseline_freeze.md` | C2 (Framework) | C2.3, C4.2 | ✅ Data ready |
| Table 2 | `output/results/phase9_step2/summary.csv` | C1 (GPU Runtime) | C1.1, C1.2, C1.4, C1.8, C4.5, C4.6 | ✅ Data ready |
| Table 3 | `output/results/phase9_step3/*/summary.csv` | C1 + C4 | C1.3, C1.7, C4.4 | ✅ Data ready |
| Table 4 | `output/results/phase9_step4_5/variance_summary.csv` | C4 (CBP) | C1.3, C4.3 | ✅ Data ready |
| Table 5 | Phase 6 + Phase 8 profiling data | C1 + C2 | C1.9, C2.6 | ⚠️ Data exists but table not yet formatted |
| Table 6 | Future (Phase 10 Step 3) | C1 | — (placeholder) | ⚠️ Planned |

---

## Verified File Existence Checklist

### Input Documents (read-only)
- [x] `docs/paper_story_freeze.md`
- [x] `docs/paper_outline.md`
- [x] `docs/evidence_matrix.md`
- [x] `docs/baseline_freeze.md`
- [x] `paper/draft/method.md`
- [x] `paper_assets/experiment_summary.md`

### Experiment Data Directories
- [x] `output/results/phase9_step2/` (summary.csv + 4 config subdirs with summary.csv)
- [x] `output/results/phase9_step3/` (4 config subdirs with summary.csv)
- [x] `output/results/phase9_step4_5/` (variance_summary.csv, neg_sampling_variance.csv)
- [x] `output/results/unified_runtime/` (epoch_summary_*.csv, runtime_trace_*.csv)
- [x] `output/results/runtime_attribution/` (runtime_attribution.csv)
- [x] `output/results/gpu_cost_model/` (benchmark.csv)
- [x] `output/results/gpu_sampler/` (validation.md)
- [x] `output/results/cost_table.npy`
- [x] `output/results/entity_features.npz`

### Paper Assets
- [x] `paper_assets/figures/fig1_profiling_breakdown.pdf`
- [x] `paper_assets/figures/fig2_cost_model_corr.pdf`
- [x] `paper_assets/figures/fig3_batch_cost_distribution.pdf`
- [x] `paper_assets/figures/fig4_gpu_runtime_trace.pdf`
- [x] `paper_assets/figures/fig5_benchmark_bars.pdf`
- [x] `paper_assets/figures/fig6_ablation_variance.pdf`

### Experiment Scripts
- [x] `src/py/experiments/phase9_step2_benchmark.py`
- [x] `src/py/experiments/phase9_step3_ablation.py`
- [x] `src/py/experiments/phase9_step4_5_cpu_variance.py`
- [x] `src/py/experiments/phase9_step1_alignment.py`
- [x] `src/py/experiments/runtime_attribution.py`
- [x] `src/py/experiments/run_unified_runtime_validation.py`
- [x] `src/py/experiments/validate_gpu_sampler.py`
- [x] `src/py/experiments/validate_gpu_sampler_full.py`
- [x] `src/py/experiments/validate_cbp_integration.py`
- [x] `scripts/fit_cost_model.py`
- [x] `scripts/validate_b1_correlation.py`
- [x] `analyze_profiling.py`
- [x] `src/py/load/gpu_sampler.py`
- [x] `src/py/load/schedulers.py`
- [x] `src/py/load/cost_model.py`
- [x] `src/py/load/features.py`
- [x] `src/py/load/batch_provider.py`
- [x] `src/py/load/cost_estimator.py`
- [ ] `src/py/experiments/gpu_cost_microbench.py` — **Path not confirmed**: referenced in `paper_story_freeze.md` §Q4 (C3) but not visible in project tree listing. May exist under a different name or path.

---

## Identified Gaps and Weaknesses

### C1 (GPU Runtime) — Evidence Strength: ★★★★★
- **No critical gaps** for core speedup/variance claims.
- **Minor gaps**: No statistical repeats for end-to-end benchmark (single-run data); C1.9 (bottleneck shift table) not yet formatted.

### C2 (Unified Runtime Framework) — Evidence Strength: ★★★★☆
- **Gap**: Architecture diagram not yet generated (mentioned as "to be drawn manually" in `paper_outline.md`).
- **Gap**: Table 5 (Overhead and Bottleneck Shift) not yet formatted, although raw data exists.
- **Note**: Framework claims are primarily architectural/qualitative; validation through C1 and C4 experiments.

### C3 (Offline Cost Model) — Evidence Strength: ★★★★☆
- **Gap**: R²=0.9008 is single-point; no 95% CI or bootstrap validation.
- **Gap**: Runtime correlation r=0.71 is moderate; gap between offline and online acknowledged.
- **Gap**: GPU cost microbench script path not confirmed (`gpu_cost_microbench.py`).

### C4 (CBP) — Evidence Strength: ★★★☆☆ (Conditionally Strong)
- **Gap**: C4.1 (78% reduction at batch_size=1000) lacks statistical repeats; single-run data.
- **Gap**: At batch_size=5000, effect is marginal (8.4%) — honestly documented as limitation.
- **Gap**: No CBP evaluation on other datasets (WN18RR) or models (RotatE, ConvE).

### All Contributions — Cross-cutting Gaps
- **No statistical repeats** for end-to-end benchmark results (Phase 9 Step 2).
- **No confidence intervals** reported for any metrics.
- **Sensitivity analysis** (batch_size, neg_num, model, dataset) planned but not yet executed.
- **200-sample evaluation subset** limits direct comparability to literature SOTA.

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Claims Inventoried | 28 (C1: 9, C2: 6, C3: 6, C4: 7) |
| Claims with Full Evidence (Script + CSV + Figure/Table) | 19 |
| Claims with Partial Evidence (missing figure or table formatting) | 7 |
| Claims with Unconfirmed Script Path | 1 (C3.4: `gpu_cost_microbench.py`) |
| Verified Figures | 6 (all present in `paper_assets/figures/`) |
| Verified Data CSVs | 18+ (all verified in `output/results/`) |
| Verified Scripts | 17 (all verified in `src/py/` and `scripts/`) |
| Gaps Identified | 9 (across all contributions) |

---

*End of Evidence Audit Part 1. This document is read-only and based entirely on existing frozen assets. No files were modified, no code was executed, and no new experiments were created.*