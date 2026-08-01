# Evidence Audit Part 1 — Paper Claim Inventory

**Version**: 1.1
**Date**: 2026-08-01
**Inventory Status**: Complete and frozen for Parts 2–7
**Based on**: `docs/paper_story_freeze.md`, `docs/paper_outline.md`, `docs/evidence_matrix.md`, `docs/baseline_freeze.md`, `paper/draft/method.md`, `paper_assets/experiment_summary.md`, and all verified files under `output/results/` and `paper_assets/`

---

## Overview

This document inventories all candidate paper claims across the four contributions
(C1–C4), mapping each claim to its supporting experiments, figures/tables, scripts,
and CSV/summary data files. Claims are ordered by contribution strength per
`paper_story_freeze.md` §Q3.

This is an **inventory, not a validity verdict**. A file being present does not make
the associated claim correct. Parts 2–7 must trace each active claim to raw data,
recompute the metric, inspect the generating code, and only then assign A/B/C/D.

### Frozen Positioning Decision

The GPU path is positioned as a **redesigned GPU-native negative sampler**, not as a
semantically identical port of the original CPU sampler:

- CPU baseline: Bernoulli head/tail corruption with global triple collision checks.
- GPU path: tail-only corruption with a batch-level `pos_tails` filter.
- Runtime comparisons therefore compare two declared runtime paths with different
  sampling semantics.
- Claims of equivalent convergence or non-inferior link-prediction quality remain
  on hold until Part 2 audits or replaces the current quality evaluation.

### Inventory Status Vocabulary

- **ACTIVE**: candidate claim proceeds to Parts 2–5 for validity audit.
- **HOLD**: evidence exists, but the current wording is not paper-safe.
- **RETRACTED**: known contradiction or invalid variable; excluded from active paper
  claims but retained under the original ID for traceability.

---

## C1: GPU Runtime — Fully Vectorized GPU Negative Sampling

**Contribution Scope**: Runtime, variance, stability, quality, and memory observations
for the redesigned GPU-native sampling path. Reported values from different phases
remain separate until Part 2 reconciles protocol and aggregation differences.

### Claim Inventory

| # | Claim | Experiment | Figure/Table | Script | CSV Data |
|---|-------|-----------|-------------|--------|----------|
| **C1.1 — ACTIVE** | Under the Phase 8 protocol, the redesigned GPU sampling path reports approximately 596ms → 3.0ms per batch (~198×) relative to the original CPU path | Phase 8 Step 2 (CPU 2 epochs; GPU 5 epochs; warm-up handling requires audit) | Fig.4 (`paper_assets/figures/fig4_gpu_runtime_trace.pdf`) | `src/py/experiments/run_unified_runtime_validation.py`; `src/py/load/gpu_sampler.py` | `output/results/unified_runtime/runtime_trace_CPU.csv`; `output/results/unified_runtime/runtime_trace_GPU.csv` |
| **C1.2 — ACTIVE** | Under the Phase 9 Step 2 protocol at batch_size=5000 and neg_num=150, the GPU configuration reports 25.1s → 4.4s average epoch time (5.7×) relative to BL | Phase 9 Step 2 (5 epochs × 4 configs) | Fig.5 (`paper_assets/figures/fig5_benchmark_bars.pdf`); Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/summary.csv`; `output/results/phase9_step2/GPU/summary.csv`; `output/results/phase9_step2/BL/summary.csv` |
| **C1.3 — ACTIVE** | In the final epoch of the Phase 9 Step 3 runs, within-epoch per-batch neg-time std is 28.5ms for BL and 0.2ms for GPU (reported ratio 142×) | Phase 9 Step 3 (10-epoch ablation) | Fig.6 (`paper_assets/figures/fig6_ablation_variance.pdf`); Table 3 | `src/py/experiments/phase9_step3_ablation.py` | `output/results/phase9_step3/GPU/summary.csv`; `output/results/phase9_step3/BL/summary.csv` |
| **C1.4 — ACTIVE** | Under the Phase 8 protocol, mean step time is reported as approximately 674ms for CPU and 79.7ms for GPU (~8.5×) | Phase 8 Step 2 | Fig.4 | `src/py/experiments/run_unified_runtime_validation.py` | `output/results/unified_runtime/runtime_trace_CPU.csv`; `output/results/unified_runtime/runtime_trace_GPU.csv` |
| **C1.5 — HOLD** | Candidate non-inferiority claim: the redesigned tail-only GPU sampler preserves link-prediction quality relative to the original CPU Bernoulli sampler | Phase 9 Step 1 (2 epochs; known MRR/Hits@10 bug) | — | `src/py/experiments/phase9_step1_alignment.py` | `docs/semantic_alignment_report.md`; `output/results/phase9_step1/results.csv` |
| **C1.6 — HOLD** | Candidate memory claim: quantify additional peak/allocated VRAM attributable specifically to the GPU sampler | Phase 8 Step 1 | — | `src/py/experiments/validate_gpu_sampler.py`; `src/py/experiments/validate_gpu_sampler_full.py`; `src/py/load/gpu_sampler.py` | `output/results/gpu_sampler/validation.csv` (timing only; no direct memory measurement) |
| **C1.7 — ACTIVE** | Phase 9 Step 3 reports GPU neg-sampling means of 2.9–3.4ms across 10 epochs, with the post-warm-up epochs near 2.9–3.2ms | Phase 9 Step 3 | Fig.6; Table 3 | `src/py/experiments/phase9_step3_ablation.py` | `output/results/phase9_step3/GPU/summary.csv`; `output/results/phase9_step3/CBP+GPU/summary.csv` |
| **C1.8 — HOLD** | Descriptive five-epoch observation: sampled evaluation reports GPU MRR 0.0132 and BL MRR 0.0136; this is not yet an equivalence or full-convergence claim | Phase 9 Step 2 (200-sample evaluation subset) | Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/GPU/summary.csv`; `output/results/phase9_step2/BL/summary.csv` |
| **C1.9 — HOLD** | Candidate bottleneck-shift claim: quantify the negative-sampling share of step time before and after the redesigned GPU path using consistently defined timing components | Phase 6 profiling + Phase 8 Step 2 | Fig.1; Table 5 (not yet formatted) | `analyze_profiling.py`; `src/py/experiments/run_unified_runtime_validation.py` | `output/results/training_time_breakdown.md`; `output/results/unified_runtime/runtime_trace_GPU.csv` |

**Coverage Notes**:
- C1.1/C1.4 and C1.2 use different experiment phases and must not be merged
  into one estimand.
- C1.3 is a within-epoch dispersion metric, not between-run uncertainty.
- C1.5/C1.8 require a valid quality protocol before words such as
  “comparable”, “equivalent”, or “non-inferior” can be used.
- C1.6 has no direct sampler-only memory measurement in the cited asset.

---

## C2: Unified Runtime Framework — Cost-aware Scheduling + GPU Execution Architecture

**Contribution Summary** (from `paper_story_freeze.md` §Q3):
定义了 FeatureExtractor → CostModel(纯函数) → Scheduler(Sort+Pack组合) → BatchProvider(Adapter) 四层解耦的运行时架构，并通过 GPUNegativeSampler 将 GPU 执行无缝嵌入。

### Claim Inventory

| # | Claim | Experiment | Figure/Table | Script | CSV Data |
|---|-------|-----------|-------------|--------|----------|
| **C2.1 — ACTIVE** | The framework separates FeatureExtractor, CostModel, Scheduler, BatchProvider, and sampler backend, allowing configured CPU and GPU runtime paths without claiming sampler-semantic identity | Phase 7 Step 4–5; Phase 8 Step 0 | Architecture diagram (not yet drawn) | Design/source inspection | `docs/gpu_runtime_architecture.md`; `docs/phase8_architecture_freeze.md`; `docs/runtime_framework_spec.md`; `paper/draft/method.md` §3.5 |
| **C2.2 — ACTIVE** | A single benchmark driver executes BL, CBP, GPU, and CBP+GPU configurations | Phase 9 Step 2 (4 configs × 5 epochs) | Fig.5; Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/summary.csv` |
| **C2.3 — ACTIVE** | Scheduler and BatchProvider interfaces are shared across configured CPU and GPU paths, while the sampler backend is selectable | Phase 8 Step 2; Phase 9 Step 2 | Table 1 | `src/py/experiments/run_unified_runtime_validation.py`; `src/py/load/schedulers.py`; `src/py/load/batch_provider.py` | `output/results/unified_runtime/unified_runtime_validation.md`; `output/results/integration_validation/validation_summary.json` |
| **C2.4 — ACTIVE** | The cost model is implemented as a deterministic offline table-building function and runtime cost access is a precomputed array lookup; “zero overhead” is not asserted | Phase 5.5 / Phase 6 | Fig.2 | `scripts/fit_cost_model.py`; `src/py/load/cost_model.py` | `output/results/cost_table.npy`; `output/results/cost_model_summary.md` |
| **C2.5 — ACTIVE** | BatchProvider exposes an iterator of preassembled batches to the training loop and supports rank-strided batch assignment; it is not itself a PyTorch DataLoader | Phase 8 Step 0; source inspection | — | `src/py/load/batch_provider.py` | `output/results/integration_validation/batch_mapping.md`; `output/results/integration_validation/batch_composition.md` |
| **C2.6 — RETRACTED** | The previous “~0.5ms per epoch” scheduler-overhead claim is contradicted by recorded Phase 6 measurements: 64.757ms for baseline scheduling and 1165ms for CostSorter+FFDPacker | Phase 6 scheduler logging | Table 5 candidate | `src/py/load/batch_provider.py`; `src/py/load/schedulers.py` | `output/results/scheduler_overhead.md`; `output/results/exp_Baseline/training.md`; `output/results/exp_CBP/training.md` |

**Coverage Notes**:
- C2.1 (architecture diagram): No figure in paper assets yet (to be drawn manually per `paper_outline.md`).
- C2.6 is removed from active paper claims. The measured scheduler overhead and
  its end-to-end impact must be audited separately in Part 3.
- The framework reuse table in `docs/phase8_architecture_freeze.md` documents which modules are reused vs newly created.

---

## C3: Offline Runtime Cost Model — Topological Feature → Expected Cost Mapping

**Contribution Scope**: Static-feature cost construction, its relationship to
measured runtime, and lookup behavior. The reported R²=0.9008 and the later
R²=0.3751 use different targets and are not treated as mutual validation.

### Claim Inventory

| # | Claim | Experiment | Figure/Table | Script | CSV Data |
|---|-------|-----------|-------------|--------|----------|
| **C3.1 — HOLD** | Candidate measured-cost claim: candidate_size explains a reported R²=0.9008 on 455 sampled observations; target provenance, sampling unit, and leakage must be reconstructed before use | Phase 5.5 / Phase 6 | Fig.2 (`paper_assets/figures/fig2_cost_model_corr.pdf`) | `scripts/fit_cost_model.py` | `docs/cost_model.md`; `output/results/cost_model_summary.md`; `output/results/cost_model_data.md` |
| **C3.2 — ACTIVE** | Phase 6 reports an association of r=0.71 between predicted batch weight and measured negative-sampling time; no causal interpretation is implied | Phase 6 runtime attribution | Fig.3 context; attribution report | `src/py/experiments/runtime_attribution.py` | `output/results/runtime_attribution/runtime_attribution.csv`; `output/results/runtime_attribution/attribution_interpretation.md` |
| **C3.3 — ACTIVE** | The implemented cost table is a deterministic function of static KG features and configured constants, requiring no online profiler during training | Phase 5.5; source inspection | — | `scripts/fit_cost_model.py`; `src/py/load/cost_model.py`; `src/py/load/features.py` | `output/results/cost_table.npy`; `output/results/entity_features.npz` |
| **C3.4 — ACTIVE** | The Phase 7 GPU microbenchmark reports a CPU/GPU timing crossover between N=150k and N=300k samples; the exact interpolated break-even estimate requires recomputation | Phase 7 Step 3 | — | `src/py/experiments/gpu_cost_microbench.py` | `output/results/gpu_cost_model/benchmark.csv`; `output/results/gpu_cost_model/benchmark.md` |
| **C3.5 — RETRACTED** | The previous `hub_entity_count` Pearson R≈0.816 claim is invalid because the variable has only two distinct values and mostly encodes the short final batch; it must not appear as evidence for degree/candidate_size | Phase 2/3 historical analysis | Removed from active figure mapping | `analyze_neg_sampling.py`; `scripts/validate_b1_correlation.py`; `scripts/plot_corrected_B_correlation.py` | `output/results/hub_analysis.md`; `output/results/negative_sampling_breakdown.md` |
| **C3.6 — ACTIVE** | Runtime access to a precomputed entity cost is O(1); the current stored table contains 14,505 float32 values and occupies 58,020 data bytes (~56.7KiB), excluding `.npy` header overhead | Source/static inspection | — | `src/py/load/cost_model.py` | `output/results/cost_table.npy` |

**Coverage Notes**:
- C3.1 (R²=0.9008): The Phase 10 bootstrap used a generated cost table
  rather than the original measured-cost target, so it does not close this gap.
- C3.2 (r=0.71): Moderate runtime correlation; gap between offline prediction and online measurement acknowledged in `evidence_matrix.md`.
- C3.4 script and benchmark assets are confirmed present.
- C3.5 is retained only as a retracted historical claim. A corrected
  `unique_entities` analysis, if used, must enter Part 4 as a new candidate claim.
- C3.6 corrects the paper draft's hypothetical float64 estimate to the actual
  float32 artifact.

---

## C4: Cost-aware Batch Packing (CBP) — Pluggable Sort+Pack Scheduling

**Contribution Summary** (from `paper_story_freeze.md` §Q3, §Q4):
在 CPU 路径上验证了代价感知调度的可行性（Phase 6 在 batch_size=1000 下将 neg_std 降低 78%），并为框架提供了可插拔的 Sort+Pack 策略接口。虽然在全训练循环（batch_size=5000）中边际收益被系统噪声稀释（Phase 9 Step 4.5: std 仅降低 8.4%），但 CBP 的实验过程是推动 GPU 迁移的关键动机。

**⚠️ Conditional Strength Note**: CBP is strong only at batch_size=1000 (Phase 6: 78% reduction). At standard training batch_size=5000, the effect is marginal (Phase 9 Step 4.5: 8.4% reduction). Per `paper_story_freeze.md`, CBP is positioned as "the critical intermediate step motivating GPU migration," not as a standalone highlight contribution.

### Claim Inventory

| # | Claim | Experiment | Figure/Table | Script | CSV Data |
|---|-------|-----------|-------------|--------|----------|
| **C4.1 — HOLD** | A Phase 6 single-run experiment at batch_size=1000 reports neg-time std reduction from 15.5ms to 3.4ms (78%); this is empirical, not proof of theoretical correctness | Phase 6 runtime attribution | Fig.3 | `src/py/experiments/runtime_attribution.py` | `output/results/runtime_attribution/runtime_attribution.csv`; `output/results/runtime_attribution/runtime_attribution.md` |
| **C4.2 — ACTIVE** | The implementation exposes two sorter classes and two packer classes that can be composed; Phase 9 benchmarks two paired scheduling strategies (Random+Chunk and Cost+FFD), not all four sorter×packer combinations | Phase 8 Step 0; Phase 9 Step 2 | Table 1 | `src/py/load/schedulers.py`; `src/py/load/cost_model.py`; `src/py/load/features.py` | `output/results/phase9_step2/summary.csv` |
| **C4.3 — ACTIVE** | Phase 9 Step 4.5 reports BL 29.5ms versus CBP 27.0ms neg-time std at batch_size=5000 (8.4% reduction); proposed noise explanations remain hypotheses | Phase 9 Step 4.5 | Fig.6; Table 4 | `src/py/experiments/phase9_step4_5_cpu_variance.py` | `output/results/phase9_step4_5/variance_summary.csv`; `output/results/phase9_step4_5/neg_sampling_variance.csv` |
| **C4.4 — ACTIVE** | In the Phase 9 Step 3 ten-epoch runs at batch_size=5000, BL and CBP have similar observed neg-time dispersion; generalization beyond this protocol is not asserted | Phase 9 Step 3 | Fig.6; Table 3 | `src/py/experiments/phase9_step3_ablation.py` | `output/results/phase9_step3/CBP/summary.csv`; `output/results/phase9_step3/BL/summary.csv` |
| **C4.5 — HOLD** | Descriptive five-epoch observation: CBP sampled MRR is 0.0150 and BL sampled MRR is 0.0136; no positive convergence effect is claimed | Phase 9 Step 2 (200-sample evaluation subset) | Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/CBP/summary.csv`; `output/results/phase9_step2/BL/summary.csv` |
| **C4.6 — HOLD** | Descriptive five-epoch observation: CBP+GPU sampled MRR is 0.0113 and GPU sampled MRR is 0.0132; no equivalence claim is made | Phase 9 Step 2 (200-sample evaluation subset) | Table 2 | `src/py/experiments/phase9_step2_benchmark.py` | `output/results/phase9_step2/CBP+GPU/summary.csv`; `output/results/phase9_step2/GPU/summary.csv` |
| **C4.7 — ACTIVE** | Phase 6 reports batch-cost CV of 0.055 for the baseline schedule and 0.012 for CBP at batch_size=1000 | Phase 6 runtime attribution | Fig.3 | `src/py/experiments/runtime_attribution.py` | `output/results/runtime_attribution/runtime_attribution.csv` |

**Coverage Notes**:
- C4.1 (78% reduction): Single-run data; no independent-run uncertainty at
  batch_size=1000 is available.
- C4.3 (8.4% marginal): Isolated CPU variance experiment; clearly documented limitation.
- C4.5/C4.6 are descriptive observations only, based on a five-epoch,
  200-sample evaluation subset.
- All C4 claims acknowledge the batch_size dependency (strong at 1000, marginal at 5000) per `paper_story_freeze.md` narrative positioning.

---

## Cross-Reference: Experiment Phase → Contribution Mapping

| Phase / Step | Experiment Name | C1 (GPU) | C2 (Framework) | C3 (Cost Model) | C4 (CBP) | Primary Script(s) |
|-------------|----------------|:---:|:---:|:---:|:---:|-------------------|
| Phase 5.5 | Cost Model Fitting | — | — | ✅ | ✅ | `scripts/fit_cost_model.py` |
| Phase 6 | Runtime Attribution (batch_size=1000) | — | — | ✅ | ✅ | `src/py/experiments/runtime_attribution.py` |
| Phase 7 Step 3 | GPU Cost Microbench | ✅ | — | ✅ | — | `src/py/experiments/gpu_cost_microbench.py` |
| Phase 7 Step 4–5 | Route C Architecture Recommendation | — | ✅ | — | — | N/A (design doc) |
| Phase 8 Step 0 | Architecture Freeze | — | ✅ | — | — | N/A (design doc) |
| Phase 8 Step 1 | GPU Sampler Prototype | ✅ | — | — | — | `src/py/experiments/validate_gpu_sampler.py`; `src/py/experiments/validate_gpu_sampler_full.py` |
| Phase 8 Step 2 | Unified Runtime Validation | ✅ | ✅ | — | — | `src/py/experiments/run_unified_runtime_validation.py` |
| Phase 9 Step 1 | Semantic Alignment | ✅ | — | — | — | `src/py/experiments/phase9_step1_alignment.py` |
| Phase 9 Step 2 | Main Benchmark (5 epochs × 4 configs) | ✅ | ✅ | — | ✅ | `src/py/experiments/phase9_step2_benchmark.py` |
| Phase 9 Step 3 | Ablation Study (10 epochs × 4 configs) | ✅ | ✅ | — | ✅ | `src/py/experiments/phase9_step3_ablation.py` |
| Phase 9 Step 4.5 | CPU Neg-Sampling Variance Isolation | — | — | — | ✅ | `src/py/experiments/phase9_step4_5_cpu_variance.py` |
| Phase 10 Step 2.5 | Repeats + Bootstrap + Sensitivity | ✅ | — | ✅ | ✅ | `src/py/experiments/phase10_step2_5_validation.py`; `src/py/experiments/phase10_step2_5_sensitivity_only.py` |

---

## Figure-to-Claim Traceability Matrix

| Fig # | File | Primary Contribution | Supports Claims | Status |
|-------|------|---------------------|----------------|--------|
| Fig.1 | `paper_assets/figures/fig1_profiling_breakdown.pdf` | Foundational (motivation) | C1.9; C3.5 removed | ✅ Generated; lineage audit pending |
| Fig.2 | `paper_assets/figures/fig2_cost_model_corr.pdf` | C3 (Cost Model) | C3.1, C2.4 | ✅ Generated |
| Fig.3 | `paper_assets/figures/fig3_batch_cost_distribution.pdf` | C3 + C4 | C4.1, C4.7, C3.2 | ✅ Generated |
| Fig.4 | `paper_assets/figures/fig4_gpu_runtime_trace.pdf` | C1 (GPU Runtime) | C1.1, C1.4 | ✅ Generated |
| Fig.5 | `paper_assets/figures/fig5_benchmark_bars.pdf` | C1 (GPU Runtime) | C1.2, C2.2 | ✅ Generated |
| Fig.6 | `paper_assets/figures/fig6_ablation_variance.pdf` | C1 + C4 | C1.3, C4.3, C4.4 | ✅ Generated |

---

## Table-to-Claim Traceability Matrix

| Table # | Data Source | Primary Contribution | Supports Claims | Status |
|---------|------------|---------------------|----------------|--------|
| Table 1 | `docs/baseline_freeze.md` | C2 (Framework) | C2.3, C4.2 | ✅ Data ready |
| Table 2 | `output/results/phase9_step2/summary.csv` | C1 (GPU Runtime) | C1.2, C1.8, C2.2, C4.5, C4.6 | ✅ Data ready |
| Table 3 | `output/results/phase9_step3/*/summary.csv` | C1 + C4 | C1.3, C1.7, C4.4 | ✅ Data ready |
| Table 4 | `output/results/phase9_step4_5/variance_summary.csv` | C4 (CBP) | C4.3 | ✅ Data ready |
| Table 5 | Phase 6 + Phase 8 profiling/overhead data | C1 + C2 | C1.9; retracted C2.6 context | ⚠️ Not formatted; audit pending |
| Table 6 | `output/results/phase10_step2_5/` | C1 + C3 + C4 | Repeat/bootstrap/sensitivity candidates | ✅ Data exists; protocol audit pending |

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
- [x] `output/results/gpu_sampler/` (validation.csv + validation.md)
- [x] `output/results/phase10_step2_5/` (repeat/bootstrap/sensitivity CSVs)
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
- [x] `src/py/experiments/phase10_step2_5_validation.py`
- [x] `src/py/experiments/phase10_step2_5_sensitivity_only.py`
- [x] `scripts/fit_cost_model.py`
- [x] `scripts/validate_b1_correlation.py`
- [x] `scripts/plot_corrected_B_correlation.py`
- [x] `analyze_neg_sampling.py`
- [x] `analyze_profiling.py`
- [x] `src/py/load/gpu_sampler.py`
- [x] `src/py/load/schedulers.py`
- [x] `src/py/load/cost_model.py`
- [x] `src/py/load/features.py`
- [x] `src/py/load/batch_provider.py`
- [x] `src/py/load/cost_estimator.py`
- [x] `src/py/experiments/gpu_cost_microbench.py`

---

## Identified Gaps and Weaknesses

### C1 (GPU Runtime)
- The GPU path is a redesigned sampler with different corruption and collision
  semantics; this must be disclosed in every performance comparison.
- Phase 8, Phase 9, and Phase 10 use different epoch counts and aggregation
  rules. Their 198×, 142×, 43×, 5.7×, and 5.8× values are separate estimands.
- C1.5/C1.8 quality wording is on hold because the semantic-alignment report
  acknowledges a broken MRR/Hits@10 result and later evaluation uses a small
  sample rather than a full convergence protocol.
- C1.6 has no direct sampler-only memory measurement.

### C2 (Unified Runtime Framework)
- **Gap**: Architecture diagram not yet generated (mentioned as "to be drawn manually" in `paper_outline.md`).
- **Contradiction**: C2.6's prior ~0.5ms overhead claim conflicts with logged
  scheduler measurements (64.757ms baseline; 1165ms Cost+FFD) and is retracted.
- **Note**: Framework claims are primarily architectural/qualitative; validation through C1 and C4 experiments.

### C3 (Offline Cost Model)
- **Gap**: R²=0.9008 has not received a like-for-like bootstrap on its original
  measured target. The Phase 10 R²=0.3751 analysis uses the generated cost table.
- **Gap**: Runtime correlation r=0.71 is moderate; gap between offline and online acknowledged.
- **Correction**: C3.5 `hub_entity_count` correlation is retracted.
- **Correction**: The actual cost table is float32 and 58,020 data bytes, not
  the previously stated 116KB float64 artifact.

### C4 (CBP)
- **Gap**: C4.1 (78% reduction at batch_size=1000) lacks statistical repeats; single-run data.
- **Gap**: At batch_size=5000, effect is marginal (8.4%) — honestly documented as limitation.
- **Correction**: Phase 9 benchmarks two paired scheduler strategies, not all
  four sorter×packer combinations.
- **Hold**: Short sampled MRR differences do not establish convergence effects
  or equivalence.

### All Contributions — Cross-cutting Gaps
- Phase 10 repeats and sensitivity data exist, but their raw precision,
  warm-up handling, aggregation unit, and CPU/GPU protocol symmetry require
  audit before they can close earlier gaps.
- Confidence intervals generated after one-decimal CSV rounding include
  `[nan, nan]` and cannot be accepted as final.
- **200-sample evaluation subset** limits direct comparability to literature SOTA.

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Claims Inventoried | 28 (C1: 9, C2: 6, C3: 6, C4: 7) |
| ACTIVE candidates proceeding to audit | 18 |
| HOLD candidates requiring safer wording or stronger evidence | 8 |
| RETRACTED historical claims retained for traceability | 2 (C2.6, C3.5) |
| Verified Figures | 6 (all present in `paper_assets/figures/`) |
| Verified Data CSVs | 18+ (all verified in `output/results/`) |
| Listed Scripts Confirmed Present | 23 |

---

*End of Evidence Audit Part 1. Version 1.1 freezes the candidate-claim registry
and known contradictions for Parts 2–7. The revision used existing artifacts and
read-only recomputation only; no experiment was run.*
