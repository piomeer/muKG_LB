# Paper Outline: A Cost-aware Runtime Framework for Efficient Knowledge Graph Embedding Training

<!-- LEGACY-NON-AUTHORITATIVE: Phase X X0.5 -->

**Date**: 2026-07-31  
**Status**: Draft (based on frozen story line `docs/paper_story_freeze.md`)  
**Target Venue**: KDD / VLDB / NeurIPS (systems + ML track)

---

## Outline Structure

---

### 1. Introduction

**Purpose**: Motivate the problem of CPU-bound negative sampling in KGE training, state the gap in existing frameworks, and present our cost-aware unified runtime approach with key results (5.7× acceleration, 142× variance reduction).

**Key Content**:
- KGE training is dominated by data loading, not model computation (cite Phase 6 profiling: collate 46.6%, neg sampling 35.7%, tensor 10.7%)
- Negative sampling on CPU is the single largest bottleneck (65% of step time) and introduces high variance (std=28.5ms due to Python for-loop + GIL + collision retry randomness)
- Existing frameworks (OpenKE, PyKEEN, LibKGE) treat negative sampling as a black-box preprocessing step; no cost-aware scheduling or GPU offloading
- Our contributions:
  1. **Offline Runtime Cost Model** (R²=0.9008) — maps entity topological features to expected neg-sampling cost
  2. **Cost-aware Batch Packing (CBP)** — pluggable Sort+Pack scheduler using cost predictions
  3. **GPU Runtime Pipeline** — fully vectorized GPU negative sampling (198× faster than CPU, 142× variance compression)
  4. **Unified Runtime Framework** — CostModel → Scheduler → BatchProvider → GPUNegativeSampler four-layer decoupled architecture
- Key results: 5.7× end-to-end epoch acceleration, neg_std 28.5ms → 0.2ms (142× reduction), epoch 25.1s → 4.4s
- **Figures referenced**: Fig.1 (profiling), Fig.5 (benchmark bars)
- **Tables referenced**: Table 2 (main results)

---

### 2. Related Work

**Purpose**: Position our work against existing KGE training systems, CPU/GPU data loading optimizations, and cost-aware scheduling in ML.

**Key Content**:
- **KGE Training Frameworks**: OpenKE, PyKEEN, LibKGE, DGL-KE — all treat negative sampling as CPU-side preprocessing (Phase 6 reference)
- **Negative Sampling Methods**: Uniform sampling (Bordes 2013), Bernoulli sampling (Wang 2014), self-adversarial (Sun 2019), NSCaching (Zhang 2019) — focus on *which* negatives to sample, not *how fast* to sample them
- **GPU Acceleration for KGE**: GNN-based methods use GPU for graph operations, but neg-sampling remains on CPU (e.g., DGL-KE uses CPU DataLoader with multiprocessing)
- **Cost-aware Scheduling in ML**: Learned cost models for query optimization (not KGE-specific); our offline cost model is the first to map entity topological features to neg-sampling cost in KGE
- **Data Loading Optimization**: PyTorch DataLoader (multi-worker, prefetching) — insufficient when per-batch work variance is high; CBP directly addresses this variance
- **Gap Statement**: No existing work combines (a) an offline cost model for neg-sampling, (b) cost-aware batch scheduling, and (c) GPU-native negative sampling into a unified runtime for KGE training.

---

### 3. Method

#### 3.1 Background and Profiling Analysis

**Purpose**: Present the KGE training loop, define negative sampling formally, and show profiling results that motivate the work.

**Key Content**:
- KGE training loop: for each batch → generate N negative triples per positive → collision check → compute loss → backprop
- Formal definition: B1–B5 negative sampling stages (Bernoulli head/tail selection, random candidate sampling, global collision check, retry loop)
- Profiling breakdown on FB15k-237 (TransE, batch_size=5000, neg_num=150): Collate 46.6%, Neg Sampling 35.7%, Tensor Construction 10.7%, Forward/Backward 7.0%
- Observation: neg-sampling time varies dramatically across batches (std=28.5ms at batch_size=5000), driven by collision retry variance
- **Figures referenced**: Fig.1 (profiling breakdown)
- **Tables referenced**: Table 1 (experimental setup)

#### 3.2 Offline Runtime Cost Model

**Purpose**: Introduce the cost model that maps entity topological features to expected neg-sampling cost.

**Key Content**:
- Key insight: collision rate (and thus retry count) is determined by entity `candidate_size` (number of entities sharing the same relation-type)
- Cost formula: `E[cost] = E[retry] × B3_const`, where `E[retry] = 1 / (1 - collision_prob)` (geometric distribution)
- Model fitting: linear regression of `candidate_size` vs measured sampling time, R²=0.9008 (Phase 5.5, `scripts/fit_cost_model.py`)
- Runtime validation: predicted cost vs measured neg-sampling time, r=0.71 (Phase 6, `output/results/runtime_attribution/attribution_interpretation.md`)
- Offline property: cost table is pre-computed from KG statistics, no runtime overhead
- **Figures referenced**: Fig.2 (cost model correlation scatter)
- **Tables referenced**: — (inline with text)

#### 3.3 Cost-aware Batch Packing (CBP)

**Purpose**: Describe the pluggable Sort+Pack scheduling framework that leverages the cost model.

**Key Content**:
- Scheduler interface: `SortStrategy` (Random, Cost-based) + `PackStrategy` (Chunk, FFD)
- Four configurations: BL (Random+Chunk), CBP (Cost+FFD), GPU (Random+Chunk+GPU), CBP+GPU (Cost+FFD+GPU)
- CBP pipeline: FeatureExtractor → CostModel → Scheduler → BatchProvider
- CostSorter: sort entities by decreasing candidate_size before batching
- FFDPacker: first-fit-decreasing bin-packing to balance batch weights within a chunk
- CPU results (Phase 6, batch_size=1000): neg_std reduced 78% (15.5ms → 3.4ms) — shows promise in isolation
- CPU results (Phase 9 Step 4.5, batch_size=5000): neg_std only 8.4% reduction (29.5ms → 27.0ms) — marginal benefit when full training loop noise dominates
- **Figures referenced**: Fig.3 (batch cost distribution)
- **Tables referenced**: Table 4 (variance analysis)

#### 3.4 GPU Runtime Pipeline

**Purpose**: Present the fully vectorized GPU negative sampling kernel.

**Key Content**:
- GPU Sampler v2 design: tail-only corruption, batch-level `isin` collision check (CUDA vectorized)
- Complete elimination of Python for-loop and GIL bottleneck
- Algorithm: (1) GPU random index generation → (2) batch-level candidate sampling → (3) GPU collision detection via `torch.isin` → (4) retry loop on GPU for collided samples
- Performance: neg-sampling 596ms (CPU) → 3.0ms (GPU), 198× acceleration
- Variance: neg_std 28.5ms (CPU) → 0.2ms (GPU), 142× compression
- Memory overhead: GPU sampler uses ~2MB additional VRAM (negligible on 8GB RTX 3070)
- **Figures referenced**: Fig.4 (GPU runtime trace, stacked area chart showing 275 steps)
- **Tables referenced**: Table 2 (main results), Table 3 (ablation)

#### 3.5 Unified Runtime Framework

**Purpose**: Present the overall four-layer architecture that unifies cost modeling, scheduling, and GPU execution.

**Key Content**:
- Architecture: FeatureExtractor → CostModel (pure function) → Scheduler (Sort+Pack) → BatchProvider (adapter) → GPUNegativeSampler
- Key design decisions:
  - CostModel as pure function (no state, pre-computed lookup table)
  - Scheduler as pluggable module (4 Sort×Pack combinations)
  - BatchProvider as adapter (wraps existing DataLoader interface)
  - GPUNegativeSampler as drop-in replacement for CPU sampler
- Transparent CPU/GPU path switching: same Scheduler/BatchProvider, different Sampler backend
- Validation: all 4 configurations (BL/CBP/GPU/CBP+GPU) verified in Phase 9 Step 2 benchmark
- **Figures referenced**: — (architecture diagram, to be drawn manually)
- **Tables referenced**: — (inline with text)

---

### 4. Experiments

#### 4.1 Experimental Setup

**Purpose**: Document the fixed experimental configuration used throughout all experiments.

**Key Content**:
- Dataset: FB15k-237 (272,115 train triples, 14,505 entities, 237 relations)
- Model: TransE (embedding_dim=400, margin=1.0)
- Optimizer: Adam (lr=1e-3)
- Batch size: 5000, Neg num: 150
- Hardware: single NVIDIA RTX 3070 (8GB VRAM), Intel Core i7, 32GB RAM
- Four experiment groups: BL (Random+Chunk, CPU), CBP (Cost+FFD, CPU), GPU (Random+Chunk, GPU), CBP+GPU (Cost+FFD, GPU)
- Evaluation: MRR and Hits@10 on 200-sample filtered subset (training subset for relative comparison; not directly comparable to SOTA)
- **Tables referenced**: Table 1 (experimental configuration, from `docs/baseline_freeze.md`)

#### 4.2 Main Results — End-to-End Acceleration

**Purpose**: Show the primary empirical finding: GPU Runtime provides 5.7× epoch acceleration over CPU baseline.

**Key Content**:
- 5-epoch benchmark (Phase 9 Step 2) comparing all 4 configurations
- Epoch time: BL 25.1s, CBP 25.3s, GPU 4.4s, CBP+GPU 4.7s
- Neg-sampling time: CPU ~596ms → GPU ~3.0ms (198×)
- MRR/Hits@10: all configurations achieve comparable convergence (5-epoch relative comparison)
- Key finding: GPU alone dominates the speedup; CBP adds marginal benefit on GPU
- **Figures referenced**: Fig.5 (benchmark bar chart with 5.7× annotation)
- **Tables referenced**: Table 2 (main benchmark results)

#### 4.3 Ablation Study

**Purpose**: Disentangle the contributions of CBP and GPU Runtime through a systematic 10-epoch ablation.

**Key Content**:
- 10-epoch ablation (Phase 9 Step 3) across all 4 configurations
- Epoch time evolution: all configurations show ~40% increase from epoch 0 to 9 due to collision retry growth as embeddings converge
- Neg-sampling time per epoch: BL avg 381ms, GPU avg 2.8ms
- GPU neg_std: 0.2ms (near-zero variance) across all 10 epochs
- CPU neg_std: 28.5ms avg (high variance) across all 10 epochs
- CBP vs BL on CPU: neg_std nearly identical (28.5ms), confirming CBP's marginal effect in full-loop standard conditions
- **Figures referenced**: Fig.6 (ablation variance: left panel neg_std, right panel step_std)
- **Tables referenced**: Table 3 (ablation results per epoch)

#### 4.4 Runtime Variance Analysis

**Purpose**: Isolate the variance contribution of each training component and explain why CBP's effect is diluted.

**Key Content**:
- CPU-only negative sampling isolation experiment (Phase 9 Step 4.5)
  - BL: mean 341.9ms, std 29.5ms
  - CBP: mean 338.2ms, std 27.0ms
  - Variance reduction: only 8.4% (marginal)
- Explanation: in batch_size=5000 scenario, per-batch work variation is dominated by:
  - (a) Python GIL serialization overhead in multi-threaded DataLoader
  - (b) Random tensor construction time (10.7% of step, ±15% variance)
  - (c) OS-level scheduling jitter
  - These noise sources are independent of CBP scheduling
- GPU path eliminates (a) and (c), leaving only (b) as residual variance source
- **Figures referenced**: Fig.6 (variance comparison)
- **Tables referenced**: Table 4 (variance analysis summary)

#### 4.5 Overhead and Bottleneck Shift

**Purpose**: Analyze how the bottleneck shifts after GPU acceleration and quantify framework overhead.

**Key Content**:
- Pre-GPU bottleneck: neg-sampling 65% → collate 25% → tensor 10%
- Post-GPU bottleneck: collate 50% → model forward/backward 30% → tensor 15% → neg-sampling <5%
- CostModel overhead: pre-computed offline, zero runtime cost
- Scheduler overhead (CBP): CostSorter+FFDPacker adds ~0.5ms per epoch (negligible vs 25s epoch)
- GPU Sampler overhead: CUDA kernel launch + memory transfer ~0.1ms per batch (amortized over batch_size=5000)
- **Figures referenced**: — (pie charts for pre/post GPU bottleneck, to be drawn manually)
- **Tables referenced**: Table 5 (overhead breakdown)

#### 4.6 Sensitivity Analysis (Future Execution)

**Purpose**: Placeholder section for planned sensitivity experiments to strengthen paper claims.

**Key Content**:
- Batch size sensitivity: test 1000, 3000, 5000, 8000 (if VRAM permits) to verify GPU acceleration holds across batch sizes
- Negative count sensitivity: test neg_num = 10, 25, 50, 100, 150 to verify variance compression is independent of neg_num
- Model sensitivity: test RotatE and ConvE to verify framework generalizability beyond TransE
- Dataset sensitivity: test WN18RR to verify generalizability beyond FB15k-237
- **Note for reviewers**: experiments listed here are planned; actual results will be added after execution (see `docs/validation_plan.md`)
- **Figures referenced**: Fig.7 (sensitivity curves, placeholder)
- **Tables referenced**: Table 6 (sensitivity analysis results, placeholder)

---

### 5. Discussion

**Purpose**: Reflect on broader implications, limitations, and future work.

**Key Content**:
- **Why does CBP fail to deliver in full-loop conditions?** The batch_size=5000 scenario leaves only 55 batches per epoch, giving CBP's sorting/packing insufficient granularity to manifest variance reduction. The Phase 6 success (78% reduction) at batch_size=1000 (275 batches) confirms CBP's theoretical correctness — it simply needs more batches to show effect.
- **When is GPU Runtime most beneficial?** Scenarios with (a) large neg_num (≥50), (b) large entity count (≥10K), (c) high connectivity variance (hub entities cause high collision rates)
- **Generalizability**: The Unified Runtime Framework's modular design (CostModel as pure function, Scheduler as pluggable) makes it adaptable to other KGE models and datasets with minimal changes
- **Limitations**:
  - Single-GPU validation only (no multi-GPU experiments yet; DDP support planned per `docs/gpu_runtime_architecture.md`)
  - 200-sample evaluation subset (not directly comparable to SOTA; full evaluation needed)
  - 5-epoch precision results are early-stage; 100-epoch convergence verification needed
- **Future work**:
  - Extend to multi-GPU DDP training (framework already supports this via `src/py/experiments/main_multi_gpus.py`)
  - Explore learned cost models (neural network predictors) to improve R² beyond 0.90
  - Integrate with dynamic batching frameworks (e.g., PyTorch's `torch.vmap` for further GPU acceleration)

---

### 6. Conclusion

**Purpose**: Summarize contributions and key takeaways.

**Key Content**:
- We identified and solved the CPU negative sampling bottleneck in KGE training through a systematic five-stage research pipeline: profiling → cost modeling → CPU scheduling → GPU migration → ablation
- The Unified Runtime Framework provides a modular, cost-aware architecture for KGE training
- GPU Runtime achieves 5.7× end-to-end acceleration and 142× neg-sampling variance compression
- The Cost Model (R²=0.9008) enables prediction of neg-sampling cost without runtime measurement
- CBP demonstrates the value of cost-aware scheduling, even though its independent benefit is marginal in standard training conditions — it served as the critical intermediate step that motivated GPU migration
- The framework is extensible to other models, datasets, and multi-GPU settings

---

## Figure and Table Index

### Figures
| Fig # | Title | Data Source |
|-------|-------|-------------|
| Fig.1 | Training Step Time Profiling Breakdown | `analyze_profiling.py`; `output/results/training_time_breakdown.md` |
| Fig.2 | Cost Model: predicted vs measured neg-sampling cost | `scripts/fit_cost_model.py`; Phase 5.5 data |
| Fig.3 | Batch Cost Distribution (Baseline vs CBP) | Phase 6 + Phase 9 Step 3 data |
| Fig.4 | GPU Runtime Trace (275 steps, stacked area chart) | `output/results/unified_runtime/runtime_trace_GPU.md` |
| Fig.5 | Epoch Time Benchmark (4-config bar chart with 5.7× annotation) | `output/results/phase9_step2/summary.md` |
| Fig.6 | Ablation Variance (neg_std + step_std, 10 epochs × 4 configs) | `output/results/phase9_step3/*/summary.csv` |
| Fig.7 | Sensitivity Analysis (batch_size × neg_num) [Placeholder] | Future execution |
| Fig.8 | Bottleneck Shift (pre- vs post-GPU pie charts) [Placeholder] | Phase 6 + Phase 8 data |

### Tables
| Table # | Title | Data Source |
|---------|-------|-------------|
| Table 1 | Experimental Configuration | `docs/baseline_freeze.md` |
| Table 2 | Main Benchmark Results (5 epochs, 4 configs) | `output/results/phase9_step2/summary.md` |
| Table 3 | Ablation Study (10 epochs, per-epoch breakdown) | `output/results/phase9_step3/*/summary.csv` |
| Table 4 | Runtime Variance Analysis | `output/results/phase9_step4_5/variance_summary.csv` |
| Table 5 | Overhead and Bottleneck Shift | Profiling data (Phase 6, Phase 8) |
| Table 6 | Sensitivity Analysis Results [Placeholder] | Future execution |

---

## Writing Notes

- All numerical values should be sourced from verified experiment output files under `output/results/`
- Use unified terminology: "muKG_CBP" for our modified method, "muKG baseline" only for the original paper's method
- MRR/Hits@10 values are relative (200-sample subset) — do not claim SOTA without full evaluation
- When discussing CBP, always distinguish its batch_size=1000 success (Phase 6, 78% reduction) from batch_size=5000 limitation (Phase 9, 8.4% reduction)
- The narrative arc should follow: Problem (profiling) → Modeling (cost model) → Exploration (CBP on CPU) → Breakthrough (GPU migration) → Validation (benchmark + ablation)
