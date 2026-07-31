# Evidence Matrix: Contribution → Experiment → Figure/Table Mapping

**Date**: 2026-07-31  
**Status**: Final (mapping verified against all completed experiments)  
**Based on**: `docs/paper_story_freeze.md` §Q3, §Q4

---

## Contribution–Evidence Mapping Table

| # | Contribution | Supporting Experiments | Key Metrics | Figure(s) | Table(s) | Evidence Strength | Notes |
|---|-------------|----------------------|-------------|-----------|----------|-------------------|-------|
| **C1** | **GPU Runtime** — Fully vectorized GPU negative sampling, eliminating CPU bottleneck | Phase 8 Step 1 (GPU Sampler prototype), Phase 8 Step 2 (Unified Runtime validation), Phase 9 Step 1 (Semantic alignment), Phase 9 Step 2 (5-epoch benchmark), Phase 9 Step 3 (10-epoch ablation) | neg_time: 596ms→3.0ms (198×); epoch_time: 25.1s→4.4s (5.7×); neg_std: 28.5ms→0.2ms (142×); step_time: 674ms→79.7ms (8.5×) | Fig.4 (GPU runtime trace), Fig.5 (benchmark bars), Fig.6 (ablation variance, right panel step_std) | Table 2 (main benchmark), Table 3 (10-epoch ablation) | **★★★★★ Strongest** | Core experimental contribution; all metrics verified across multiple experiments |
| **C2** | **Unified Runtime Framework** — FeatureExtractor→CostModel→Scheduler→BatchProvider→GPUNegativeSampler four-layer architecture | Phase 7 Step 4–5 (Architecture design + Route C recommendation), Phase 8 Step 0 (Architecture freeze), Phase 8 Step 2 (Unified runtime validation), Phase 9 Step 2 (Four-config full combination) | Architecture validated with 4 configs (BL/CBP/GPU/CBP+GPU); transparent CPU/GPU switching verified | — (architecture diagram to be drawn manually) | Table 1 (Experimental Configuration) | **★★★★☆ Architecture** | Modular design verified; framework reuse table in `docs/phase8_architecture_freeze.md` |
| **C3** | **Offline Runtime Cost Model** — candidate_size → expected neg-sampling cost mapping (R²=0.9008) | Phase 5.5 / Phase 6 (Cost model formula validation), Phase 6 (Runtime Attribution: Weight vs Neg Sampling r=0.71), Phase 7 Step 3 (GPU cost microbench, break-even N*=264k) | R²=0.9008 (offline fit); r=0.71 (runtime correlation); break-even N*=264k (GPU cost model) | Fig.2 (cost model correlation scatter), Fig.3 (batch cost distribution) | — (inline with text) | **★★★★☆ Theory** | Strong offline fit; runtime correlation weaker due to system noise; break-even analysis provides GPU migration justification |
| **C4** | **Cost-aware Batch Packing (CBP)** — Pluggable Sort+Pack strategy leveraging cost model | Phase 6 (neg_std 78% reduction at batch_size=1000), Phase 9 Step 3 (10-epoch CPU comparison: CBP vs BL neg_std), Phase 9 Step 4.5 (Isolated CPU neg-sampling variance: marginal 8.4% reduction) | batch_size=1000: neg_std 15.5→3.4ms (78% reduction); batch_size=5000: neg_std 28.5ms (CBP≈BL, no effect); isolated CPU: std 29.5→27.0ms (8.4% reduction) | Fig.3 (batch cost distribution), Fig.6 (ablation variance, left panel neg_std) | Table 4 (variance analysis) | **★★★☆☆ Conditionally Strong** | ⚠️ Strong only at batch_size=1000; marginal at batch_size=5000; narrative position: "critical intermediate step motivating GPU migration" |

---

## Cross-Reference: Experiment Phases → Contribution Mapping

| Phase / Step | Experiment Name | Supports C1 (GPU) | Supports C2 (Framework) | Supports C3 (Cost Model) | Supports C4 (CBP) | Key Output File |
|-------------|----------------|:---:|:---:|:---:|:---:|-----------------|
| Phase 5.5 | Cost Model Fitting | — | — | ✅ | ✅ | `scripts/fit_cost_model.py`; `output/results/cost_table.npy` |
| Phase 6 | Runtime Attribution (batch_size=1000) | — | — | ✅ | ✅ | `output/results/runtime_attribution/runtime_attribution.md` |
| Phase 7 Step 3 | GPU Cost Microbench | ✅ | — | ✅ | — | `output/results/gpu_cost_model/benchmark.md` |
| Phase 7 Step 4–5 | Route C Architecture Recommendation | — | ✅ | — | — | `docs/gpu_runtime_architecture.md` |
| Phase 8 Step 0 | Architecture Freeze | — | ✅ | — | — | `docs/phase8_architecture_freeze.md` |
| Phase 8 Step 1 | GPU Sampler Prototype | ✅ | — | — | — | `output/results/gpu_sampler/validation.md` |
| Phase 8 Step 2 | Unified Runtime Validation (5 epochs) | ✅ | ✅ | — | — | `output/results/unified_runtime/unified_runtime_validation.md` |
| Phase 9 Step 1 | Semantic Alignment (CPU original ↔ GPU v2) | ✅ | — | — | — | `docs/semantic_alignment_report.md` |
| Phase 9 Step 2 | Main Benchmark (4 configs × 5 epochs) | ✅ | ✅ | — | ✅ | `output/results/phase9_step2/summary.md` |
| Phase 9 Step 3 | Ablation Study (4 configs × 10 epochs) | ✅ | ✅ | — | ✅ | `output/results/phase9_step3/*/summary.csv` |
| Phase 9 Step 4.5 | CPU Neg-Sampling Variance Isolation | — | — | — | ✅ | `output/results/phase9_step4_5/variance_summary.csv` |

---

## Figure → Contribution → Experiment Mapping

| Fig # | Title | Primary Contribution | Data Source Phase | Status |
|-------|-------|---------------------|-------------------|--------|
| Fig.1 | Training Step Time Profiling Breakdown | Foundational (motivation) | Phase 6 | ✅ Generated (`paper_assets/figures/fig1_profiling_breakdown.pdf`) |
| Fig.2 | Cost Model: Predicted vs Measured Neg-Sampling Cost | C3 (Cost Model) | Phase 5.5 / Phase 6 | ✅ Generated (`paper_assets/figures/fig2_cost_model_corr.pdf`) |
| Fig.3 | Batch Cost Distribution (Baseline vs CBP) | C3 (Cost Model) + C4 (CBP) | Phase 6 / Phase 9 Step 3 | ✅ Generated (`paper_assets/figures/fig3_batch_cost_distribution.pdf`) |
| Fig.4 | GPU Runtime Trace (275 steps) | C1 (GPU Runtime) | Phase 8 Step 2 | ✅ Generated (`paper_assets/figures/fig4_gpu_runtime_trace.pdf`) |
| Fig.5 | Epoch Time Benchmark (4-config bars) | C1 (GPU Runtime) | Phase 9 Step 2 | ✅ Generated (`paper_assets/figures/fig5_benchmark_bars.pdf`) |
| Fig.6 | Ablation Variance (neg_std + step_std) | C1 (GPU Runtime) + C4 (CBP) | Phase 9 Step 3 | ✅ Generated (`paper_assets/figures/fig6_ablation_variance.pdf`) |
| Fig.7 | Sensitivity Analysis [Placeholder] | C1 (GPU Runtime) | Future (Phase 10 Step 3) | ⚠️ Planned |
| Fig.8 | Bottleneck Shift (pre- vs post-GPU) | C1 (GPU Runtime) | Phase 6 + Phase 8 | ⚠️ Not yet generated |

---

## Table → Contribution → Experiment Mapping

| Table # | Title | Primary Contribution | Data Source Phase | Status |
|---------|-------|---------------------|-------------------|--------|
| Table 1 | Experimental Configuration | C2 (Framework) | Baseline Freeze (`docs/baseline_freeze.md`) | ✅ Data ready |
| Table 2 | Main Benchmark Results (5 epochs, 4 configs) | C1 (GPU Runtime) | Phase 9 Step 2 (`output/results/phase9_step2/summary.md`) | ✅ Data ready |
| Table 3 | Ablation Study (10 epochs, per-epoch breakdown) | C1 (GPU Runtime) + C4 (CBP) | Phase 9 Step 3 (`output/results/phase9_step3/*/summary.csv`) | ✅ Data ready |
| Table 4 | Runtime Variance Analysis | C4 (CBP) | Phase 9 Step 4.5 (`output/results/phase9_step4_5/variance_summary.csv`) | ✅ Data ready |
| Table 5 | Overhead and Bottleneck Shift | C1 (GPU Runtime) + C2 (Framework) | Phase 6 + Phase 8 profiling data | ⚠️ Data exists but table not yet formatted |
| Table 6 | Sensitivity Analysis Results [Placeholder] | C1 (GPU Runtime) | Future (Phase 10 Step 3) | ⚠️ Planned |

---

## Gap Analysis: Contributions with Weak Evidence

### C4 (CBP) — ⚠️ 需补充

| Gap | Description | Mitigation Strategy |
|-----|-------------|-------------------|
| batch_size=5000 marginal effect | CBP's neg_std reduction is only 8.4% at standard training batch size | Acknowledge honestly in paper: CBP's benefit manifests at smaller batch sizes; at large batch sizes, GPU Runtime dominates. Frame CBP as the *exploration step* that motivated GPU migration, not as a standalone contribution |
| Lack of statistical repeats | CBP results are from single runs (Phase 9 Step 2: 5 epochs; Step 3: 10 epochs) | Add statistical validation (see `docs/validation_plan.md` §A): repeat BL and CBP 3× at batch_size=1000 to verify the 78% neg_std reduction is significant |
| Single dataset | All CBP experiments on FB15k-237 only | Optional: test on WN18RR (P2 priority per `validation_plan.md`) |

### C3 (Cost Model) — ⚠️ 需补充

| Gap | Description | Mitigation Strategy |
|-----|-------------|-------------------|
| R²=0.9008 validation is single-point | Only one linear regression fit | Report 95% confidence interval for R²; consider bootstrapping for robustness |
| Runtime correlation r=0.71 is moderate | CBP attribution experiment shows only moderate runtime correlation | Acknowledge gap between offline prediction (R²=0.9008) and online measurement (r=0.71); explain via system noise factors |

### All Contributions — ⚠️ 需补充

| Gap | Description | Mitigation Strategy |
|-----|-------------|-------------------|
| No statistical repeats for end-to-end results | All main benchmark results (Phase 9 Step 2) are single-run | Add 5× repeated runs for GPU and CBP+GPU configurations (see `docs/validation_plan.md` §A) |
| No confidence intervals reported | All metrics reported as point estimates | Add mean±std + 95% CI after statistical repeats |
| Sensitivity analysis not executed | No batch_size / neg_num / model / dataset sensitivity data | Execute planned experiments (see `docs/validation_plan.md` §B) |

---

## Narrative Flow Validation

| Paper Section | Contributions Covered | Figures Used | Tables Used | Coherence Check |
|--------------|----------------------|-------------|-------------|-----------------|
| **Introduction** | All 4 (summary) | Fig.1, Fig.5 | Table 2 | ✅ Motivated by profiling → preview key results |
| **Related Work** | Positioning all 4 | — | — | ✅ Gap statement clear: no unified cost-aware GPU runtime |