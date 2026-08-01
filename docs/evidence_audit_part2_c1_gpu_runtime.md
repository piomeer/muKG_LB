# Evidence Audit Part 2 — C1 GPU Runtime

**Version**: 1.1
**Date**: 2026-08-01
**Status**: Complete; C1-R1 replacement experiment incorporated
**Scope**: C1.1–C1.9 from `docs/evidence_audit_part1_claim_inventory.md`
**Execution Boundary**: v1.0 audited existing artifacts only. v1.1 adds the
authorized C1-R1 v1.1 replacement experiment on node4 / RTX 3070.

---

## 1. Executive Verdict

| Grade | Count | Claims |
|-------|------:|--------|
| A (Verified) | 3 | C1.2-R1, C1.3-R1, C1.7-R1 |
| B (Re-analysis) | 2 | C1.1, C1.4 |
| C (Re-experiment) | 4 | C1.5, C1.6, C1.8, C1.9 |
| D (Invalid) | 0 | — |

C1.2-R1, C1.3-R1, and C1.7-R1 satisfy the frozen publication-level A
standard: integer-nanosecond raw observations, symmetric frozen estimands, six
independent paired seeds, repeat-level uncertainty, complete generating code,
and protocol-limited paper wording.

The main conclusions are:

1. **Phase 8's 198×/8.5× results are removed from paper evidence.** The CPU
   comparator is a synthetic validation function that simultaneously replaces
   head and tail and does not perform the original global collision check.
2. **C1-R1 replaces the historical 5.7× headline with 6.013×.** The geometric
   mean of six paired end-to-end speedups is 6.013×, with a 95% log-scale
   t interval of [5.944, 6.084].
3. **The historical 142× remains invalid, but C1.3-R1 supplies a valid
   replacement.** The old result is
   28.5/0.2=142.5 from two rounded, final-epoch, within-run population standard
   deviations. C1.3-R1 instead estimates full-batch standard-deviation
   compression at 87.88× [72.92, 105.91] across six paired runs.
4. **No quality-equivalence claim is allowed.** Phase 9 Step 1 has a broken
   evaluator; the later 200-sample values are not a full official-test or
   convergence protocol.
5. **Sampler-only VRAM and bottleneck-shift claims are unsupported.** Existing
   memory is whole-training peak allocation, while Phase 6 and Phase 8 use
   incompatible timing denominators.

### Reproducible Audit Package

- Script: `scripts/audit_c1_gpu_runtime.py`
- Source manifest: `output/results/evidence_audit_part2/source_manifest.json`
- Recomputed metrics: `output/results/evidence_audit_part2/recomputed_metrics.csv`
- Machine-readable checks and grades:
  `output/results/evidence_audit_part2/audit_checks.json`
- C1-R1 runner: `src/py/experiments/c1_r1_combined_rerun.py`
- C1-R1 analyzer: `scripts/analyze_c1_r1.py`
- C1-R1 frozen protocol, raw traces, telemetry, hashes, and results:
  `output/results/c1_r1_combined_rerun/`

---

## 2. Frozen Audit Rules and Protocol Findings

### 2.1 Evidence and Grade Rules

- Raw observation traces outrank summaries, figures, narrative reports, and
  story documents.
- A stored rounded number can reproduce a report but cannot establish
  publication-level uncertainty.
- Within-epoch batch dispersion, variation between epochs in one run, and
  variation between independent runs are separate estimands.
- CPU/GPU runtime comparisons must disclose the frozen semantic difference:
  original CPU Bernoulli/global-collision sampling versus GPU tail-only,
  batch-`pos_tails` filtering.
- B is used only when existing observations can be salvaged by re-analysis or
  corrected wording. C is used when a valid paper claim requires new
  measurements.

### 2.2 Protocol Matrix

| Evidence | Observational Unit | Precision | Key Audit Finding |
|----------|--------------------|-----------|-------------------|
| Phase 8 runtime trace | Per step; CPU 2 epochs, GPU 5 epochs | Full float | CPU comparator is synthetic; final batch is short; first GPU step contains CUDA warm-up |
| Phase 9 Step 2 | Five per-epoch rows per configuration | Timing 0.1s; MRR 0.0001; memory 1MiB | Single execution; process-dependent `hash(label)` and no frozen Torch seed |
| Phase 9 Step 3 | Ten per-epoch summaries per configuration | Timing 0.1ms/0.1s | No per-step trace; `numpy.std(..., ddof=0)` includes the final short batch |
| Phase 10 repeats | Three CPU and five GPU summary rows | Timing 0.1ms/0.1s | Values rounded before CI/std; neg/step metrics aggregate all epochs while epoch time is the final epoch |
| Phase 6 profile | One aggregate component table | Mixed rounded | Includes Collate and Tensor Construction; denominator differs from Phase 8 |
| C1-R1 v1.1 | Six paired seeds; five epochs/job; throughput and trace passes in independent processes | Integer ns | Preflight passed; 24 primary jobs completed; one thermal-marked throughput attempt was retained and its full pair rerun once under the frozen infrastructure-failure rule |

### 2.3 Figure Lineage

| Figure | Audit Result |
|--------|--------------|
| Fig.4 | Reads only `runtime_trace_GPU.md`; it has no CPU series and cannot support C1.1/C1.4 speedup ratios |
| Fig.5 | Bar heights 25.1, 25.3, 4.4, and 4.7 are hardcoded in `generate_paper_assets.py`; the figure is not generated from the cited summary CSV |
| Fig.6 | Reads only the final rounded row of each Phase 9 Step 3 summary; absent per-step traces prevent raw verification |

---

## 3. Claim-Level Audit

### C1.1 — Phase 8 Negative-Sampling Component Speedup

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.1 |
| Inventory Status | ACTIVE |
| Claim Statement | Under the Phase 8 protocol, the redesigned GPU path reports approximately 596ms → 3.0ms (~198×) relative to the original CPU path |
| Claim Type | Component performance |
| Frozen Protocol | FB15k-237; TransE dim=400; batch_size=5000; neg_num=150; CostSorter+FFDPacker; CPU 2 epochs; GPU 5 epochs; RTX 3070 |
| Supporting Figure/Table | Fig.4 exists but plots GPU only and does not support the ratio |
| Primary Raw Evidence | `output/results/unified_runtime/runtime_trace_CPU.csv` (110 steps) and `output/results/unified_runtime/runtime_trace_GPU.csv` (275 steps) |
| Derived Evidence | `recomputed_metrics.csv`: all batches 590.5968/3.0934=190.9236×; full-size batches 597.1678/3.1249=191.0989×; full-size plus first-GPU-step exclusion 597.1678/2.9711=200.9947× |
| Supporting Script | `src/py/experiments/run_unified_runtime_validation.py`; `src/py/load/gpu_sampler.py` |
| Key Variables | `neg_time_ms`, `epoch`, `step`, sampler identity |
| Variable Trustworthiness | Timing is full precision and synchronized; batch length is not stored but the last step of each epoch is known to be partial from the protocol |
| Metric / Estimand Definition | Arithmetic mean of per-step negative-sampling regions; sensitivity views separately exclude partial batches and the first GPU observation |
| Statistical Method Audit | No independent repeats; asymmetric epoch counts; the exact 198× aggregation rule is not recoverable from the claim |
| Code Audit | Phase 8 CPU generates both a replacement head and tail for every negative and does not use `all_triples_set`; GPU is tail-only |
| Semantic / Fairness Audit | Comparator is neither the original CPU sampler nor semantically identical to the GPU sampler |
| Conclusion | **B — Re-analysis** |
| Paper-safe Wording | Audit only: Phase 8 recorded different component timings for two non-equivalent validation samplers |
| Fix Recommendation | If a component result is needed, run the frozen original CPU and GPU samplers on matched batches with unrounded traces, explicit warm-up, and ≥3 independent runs |
| Responsible / Status | Codex; audit closed; claim excluded from paper |

### C1.2-R1 — Replacement End-to-End Epoch Speedup

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.2-R1; replacement child of inventory Claim C1.2 |
| Inventory Status | ACTIVE |
| Claim Statement | Under C1-R1 v1.1, the redesigned GPU runtime path accelerates end-to-end training epochs relative to BL |
| Claim Type | End-to-end performance |
| Frozen Protocol | FB15k-237; Phase-9 loader lineage; Random(42) held-out 5,000; training_set_size=267,115; TransE dim=400; RandomSorter+ChunkPacker; batch_size=5,000; neg_num=150; seeds 42–47; five epochs; three disposable warm-up steps; independent process per job |
| Supporting Figure/Table | `analysis/paired_metrics.csv`; the old hardcoded Fig.5 must be regenerated |
| Primary Raw Evidence | 60 unrounded `epoch_time_ns` observations under `jobs/throughput_*`; per-job telemetry, loss, manifest, and SHA-256 records |
| Derived Evidence | Run means: BL=26.2785±0.2744s and GPU=4.36981±0.00489s across six runs (sample SD). Paired speedups=5.982, 6.047, 5.969, 6.103, 5.922, 6.059 |
| Supporting Script | `src/py/experiments/c1_r1_combined_rerun.py`; `scripts/rerun_c1_r1_pair.py`; `scripts/analyze_c1_r1.py` |
| Key Variables | Integer `epoch_time_ns`, configuration, paired seed, attempt, sampler semantics |
| Variable Trustworthiness | Throughput epochs synchronize only at epoch boundaries; timer includes scheduling and the final partial batch; loss is read once per epoch, outside the timed boundary |
| Metric / Estimand Definition | For each seed, mean of five end-to-end epoch times for BL divided by the corresponding GPU mean; geometric mean across six paired ratios |
| Statistical Method Audit | Two-sided 95% t interval on log paired ratios, df=5: **6.013× [5.944, 6.084]**; lower bound exceeds 1 |
| Code Audit | Python, NumPy, Torch, and CUDA seeds are frozen; each job is a separate process; raw timing is not rounded before analysis |
| Semantic / Fairness Audit | BL uses original CPU Bernoulli/global-collision sampling; GPU is tail-only with batch-level tail filtering. The result compares declared runtime paths and does not establish sampling-quality equivalence |
| Conclusion | **A — Verified** |
| Paper-safe Wording | On FB15k-237 with TransE, batch_size=5,000 and 150 negatives, the declared GPU runtime path achieved a 6.01× paired geometric-mean end-to-end epoch speedup over BL (95% CI 5.94×–6.08×; six seeds) |
| Fix Recommendation | None for this frozen estimand; regenerate the main-results figure/table directly from `analysis/paired_metrics.csv` |
| Responsible / Status | Codex; replacement experiment and audit closed |

### C1.3-R1 — Full-Batch Negative-Time Dispersion Compression

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.3-R1; replacement child of inventory Claim C1.3 |
| Inventory Status | ACTIVE |
| Claim Statement | The redesigned GPU runtime path compresses full-batch within-epoch negative-time standard deviation relative to BL |
| Claim Type | Runtime dispersion |
| Frozen Protocol | C1-R1 trace pass; same data/model/config pairs as C1.2-R1; synchronization at every component boundary |
| Supporting Figure/Table | `analysis/run_level_metrics.csv` and `analysis/paired_metrics.csv`; old Fig.6 is not evidence for this replacement |
| Primary Raw Evidence | 3,240 unrounded step rows (270/job × 12 trace jobs), including actual batch size, partial flag, component timings, total, and residual |
| Derived Evidence | Mean run-level epoch population SD: BL=3.1672±0.4358ms; GPU=0.03629±0.00702ms. Paired compression ratios range from 64.40× to 103.80× |
| Supporting Script | C1-R1 runner and analyzer |
| Key Variables | `neg_time_ns`, `batch_size_actual`, `is_partial`, epoch, paired seed |
| Variable Trustworthiness | Primary filter is hard-coded as `is_partial == False AND batch_size_actual == 5000`; every epoch asserts 53 full batches and one 2,115-example partial batch |
| Metric / Estimand Definition | Within each epoch, population SD (`ddof=0`) across 53 full batches; each run is the mean of its five epoch SDs; paired BL/GPU run ratio |
| Statistical Method Audit | Geometric mean and two-sided 95% log-scale t interval across six paired ratios: **87.88× [72.92, 105.91]** |
| Code Audit | Analyzer verifies component sums, timing residuals, row counts, and the full/partial filter before calculating the estimand |
| Semantic / Fairness Audit | This is standard-deviation/dispersion compression, not variance compression and not repeat-level uncertainty; sampler semantics remain explicitly different |
| Conclusion | **A — Verified** |
| Paper-safe Wording | Across six paired seeds, the declared GPU path reduced full-batch within-epoch negative-sampling time dispersion by 87.9× in standard-deviation terms (95% CI 72.9×–105.9×) |
| Fix Recommendation | Replace the old 142× figure and never label this ratio as variance compression |
| Responsible / Status | Codex; replacement experiment and audit closed |

### C1.4 — Phase 8 Recorded Step-Time Speedup

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.4 |
| Inventory Status | ACTIVE |
| Claim Statement | Under Phase 8, mean step time is approximately 674ms for CPU and 79.7ms for GPU (~8.5×) |
| Claim Type | Recorded step performance |
| Frozen Protocol | Same Phase 8 protocol and asymmetric CPU/GPU epoch counts as C1.1 |
| Supporting Figure/Table | Fig.4 plots GPU only |
| Primary Raw Evidence | Full-precision Phase 8 CPU/GPU runtime traces |
| Derived Evidence | All batches 670.6492/79.4216=8.4442×; full-size 678.0374/80.2314=8.4510×; full-size plus first-GPU-step exclusion 678.0374/79.8330=8.4932× |
| Supporting Script | `src/py/experiments/run_unified_runtime_validation.py` |
| Key Variables | `total_step_ms`, component boundaries, sampler identity |
| Variable Trustworthiness | `total_step_ms` is the sum of neg/fwd/bwd/opt only and is full precision; it is not a complete DataLoader-to-step denominator |
| Metric / Estimand Definition | Mean of recorded synchronized component sums under three batch/warm-up views |
| Statistical Method Audit | No independent repeats; exact 674/79.7 provenance is not reproduced by the stored CSV views |
| Code Audit | Uses the synthetic Phase 8 CPU sampler identified in C1.1 |
| Semantic / Fairness Audit | Not an original-CPU end-to-end comparison |
| Conclusion | **B — Re-analysis** |
| Paper-safe Wording | Audit only: Phase 8 recorded step-component sums for two non-equivalent validation paths |
| Fix Recommendation | Replace with the matched original-CPU/GPU repeat protocol for C1.2 and retain unrounded step-component traces |
| Responsible / Status | Codex; audit closed; claim excluded from paper |

### C1.5 — GPU Sampler Quality Non-Inferiority

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.5 |
| Inventory Status | HOLD |
| Claim Statement | Candidate claim that the redesigned GPU sampler preserves link-prediction quality relative to the original CPU sampler |
| Claim Type | Quality / non-inferiority |
| Frozen Protocol | Phase 9 Step 1; CostSorter+FFDPacker; two epochs; CPU original vs GPU v2 |
| Supporting Figure/Table | None |
| Primary Raw Evidence | `output/results/phase9_step1/results.csv` |
| Derived Evidence | Both configurations store MRR=6.8903619e-05 and Hits@10=0.0; these values are invalid |
| Supporting Script | `src/py/experiments/phase9_step1_alignment.py` |
| Key Variables | true score, filtered candidate mask, rank, MRR, Hits@10, non-inferiority margin |
| Variable Trustworthiness | Rank is corrupted by the evaluator; no margin was pre-specified |
| Metric / Estimand Definition | Intended filtered ranking after two epochs, limited to at most 500 tail queries |
| Statistical Method Audit | No convergence protocol, independent seeds, uncertainty, or non-inferiority test |
| Code Audit | The evaluator saves `true_score` as a tensor view and then sets every known tail, including the target, to `inf`; the target score becomes `inf` and ranking is meaningless |
| Semantic / Fairness Audit | Samplers deliberately differ in corruption and collision semantics; runtime evidence cannot establish quality equivalence |
| Conclusion | **C — Re-experiment** |
| Paper-safe Wording | No quality-equivalence or non-inferiority statement is allowed |
| Fix Recommendation | Only if the paper later needs this claim: correct the evaluator, use official splits, pre-freeze a non-inferiority margin, train to convergence, and run sufficient independent seeds |
| Responsible / Status | Codex; audit closed; claim excluded from paper |

### C1.6 — Sampler-Only VRAM Overhead

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.6 |
| Inventory Status | HOLD |
| Claim Statement | Quantify additional peak/allocated VRAM attributable specifically to the GPU sampler |
| Claim Type | Memory |
| Frozen Protocol | Phase 8 sampler validation plus Phase 9 Step 2 whole-training peaks |
| Supporting Figure/Table | None |
| Primary Raw Evidence | `output/results/gpu_sampler/validation.csv` contains timing only; Phase 9 summary contains rounded configuration peaks |
| Derived Evidence | Phase 9 stores BL=5818MiB and GPU=5819MiB, a rounded 1MiB difference |
| Supporting Script | `src/py/experiments/validate_gpu_sampler.py`; `src/py/experiments/phase9_step2_benchmark.py`; `src/py/load/gpu_sampler.py` |
| Key Variables | allocated VRAM, reserved VRAM, peak reset point, sampler delta |
| Variable Trustworthiness | Existing `gpu_mem_mb` includes model, optimizer and training tensors and is rounded to whole MiB |
| Metric / Estimand Definition | Required estimand is sampler-attributable peak/delta; it is not present |
| Statistical Method Audit | No isolated observations or repeats |
| Code Audit | Phase 9 calls `max_memory_allocated()` for the whole configuration; the sampler validator never records memory |
| Semantic / Fairness Audit | CPU tensor transfer and GPU-native allocation must use the same measurement boundary in a future comparison |
| Conclusion | **C — Re-experiment** |
| Paper-safe Wording | No sampler-only VRAM overhead claim is allowed |
| Fix Recommendation | Reset peaks after warm-up, measure allocated and reserved memory immediately around sampler generation on matched batches, and repeat ≥3 times |
| Responsible / Status | Codex; audit closed; measurement open |

### C1.7-R1 — GPU Full-Batch Negative-Sampling Mean

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.7-R1; replacement child of inventory Claim C1.7 |
| Inventory Status | ACTIVE |
| Claim Statement | Quantify the redesigned GPU path's full-batch negative-sampling mean and repeat-level stability |
| Claim Type | Component performance / stability |
| Frozen Protocol | C1-R1 trace GPU jobs; six seeds; five measured epochs; 53 full batches per epoch; batch_size=5,000; neg_num=150; disposable warm-up |
| Supporting Figure/Table | `analysis/run_level_metrics.csv` and `analysis/summary.json` |
| Primary Raw Evidence | 1,590 GPU full-batch `neg_time_ns` observations after the frozen filter |
| Derived Evidence | Six run means: 2.9995, 2.9990, 3.0325, 2.9630, 3.0115, and 3.0103ms |
| Supporting Script | C1-R1 runner and analyzer |
| Key Variables | `neg_time_ns`, seed, epoch, batch size, partial flag |
| Variable Trustworthiness | GPU readiness is enforced by synchronization at the end of the timing region; partial batches are excluded by two explicit predicates |
| Metric / Estimand Definition | Mean full-batch negative time within each GPU run, followed by arithmetic mean and repeat-level sample SD across six independent runs |
| Statistical Method Audit | Six-run mean **3.0026ms**, sample SD **0.0229ms**, two-sided 95% t interval **[2.9786, 3.0266]ms** |
| Code Audit | All raw step rows, batch identities, loss diagnostics, telemetry, and hashes passed machine checks |
| Semantic / Fairness Audit | Valid only for the redesigned tail-only, batch-tail-filtered GPU path under C1-R1 |
| Conclusion | **A — Verified** |
| Paper-safe Wording | The redesigned GPU sampler averaged 3.003ms per full 5,000-example batch (six-run sample SD 0.023ms; 95% CI 2.979–3.027ms) |
| Fix Recommendation | None for this frozen component estimand |
| Responsible / Status | Codex; replacement experiment and audit closed |

### C1.8 — Five-Epoch Sampled Quality Observation

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.8 |
| Inventory Status | HOLD |
| Claim Statement | Five-epoch sampled evaluation reports GPU MRR=0.0132 and BL MRR=0.0136 without asserting equivalence |
| Claim Type | Descriptive quality |
| Frozen Protocol | Phase 9 Step 2; five epochs; 200 sampled triples from a shuffled 5k subset removed from training |
| Supporting Figure/Table | Table 2 |
| Primary Raw Evidence | Per-configuration Phase 9 Step 2 summaries only |
| Derived Evidence | Final stored MRR: BL=0.0136, GPU=0.0132; Hits@10: BL=0.0225, GPU=0.0300 |
| Supporting Script | `src/py/experiments/phase9_step2_benchmark.py` |
| Key Variables | sampled triples, MRR, Hits@10, official split, convergence |
| Variable Trustworthiness | Evaluator code is improved relative to Step 1, but the sample is not the official full test protocol and values are rounded |
| Metric / Estimand Definition | Head+tail filtered ranking on 200 fixed sampled held-out training triples after five epochs |
| Statistical Method Audit | No independent seeds, full convergence, full-test evaluation, or uncertainty |
| Code Audit | Model initialization lacks a frozen Torch seed; `hash(label)` is process-dependent |
| Semantic / Fairness Audit | Different samplers may change optimization behavior; a small endpoint difference cannot establish equivalence |
| Conclusion | **C — Re-experiment** |
| Paper-safe Wording | Values are retained only as audit history and do not enter the manuscript |
| Fix Recommendation | Revisit only through the corrected full-convergence protocol defined for C1.5 |
| Responsible / Status | Codex; audit closed; claim excluded from paper |

### C1.9 — Bottleneck Shift

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.9 |
| Inventory Status | HOLD |
| Claim Statement | Quantify the negative-sampling share before and after the GPU path using consistently defined timing components |
| Claim Type | Profiling / bottleneck attribution |
| Frozen Protocol | Phase 6 aggregate profile versus Phase 8 synchronized per-step component trace |
| Supporting Figure/Table | Fig.1 and unformatted Table 5 candidate |
| Primary Raw Evidence | `output/results/training_time_breakdown.md`; Phase 8 CPU/GPU runtime traces |
| Derived Evidence | Phase 6: Collate=46.63%, Negative Sampling=35.70%; Phase 8 recorded-component denominator: synthetic CPU neg share=88.0730%, GPU neg share=3.7216% |
| Supporting Script | `analyze_profiling.py`; `src/py/experiments/run_unified_runtime_validation.py` |
| Key Variables | component boundary, component time, denominator, Collate inclusion |
| Variable Trustworthiness | Values are internally computable, but component names and denominators differ across phases |
| Metric / Estimand Definition | Phase 6 percentage of its aggregate profiled total versus Phase 8 ratio of summed `neg_time_ms` to summed recorded `total_step_ms` |
| Statistical Method Audit | Cross-phase ratio comparison violates denominator consistency and lacks repeats |
| Code Audit | Phase 8 `total_step_ms` sums neg/fwd/bwd/opt and has no comparable Collate field |
| Semantic / Fairness Audit | Phase 8 CPU is also the synthetic comparator, preventing a clean original-CPU→GPU causal narrative |
| Conclusion | **C — Re-experiment** |
| Paper-safe Wording | Phase 6 and Phase 8 shares may only be reported separately under their own definitions, not as a pre/post shift |
| Fix Recommendation | Instrument original CPU and GPU paths in one driver with identical exhaustive timing regions that sum to the same end-to-end denominator |
| Responsible / Status | Codex; audit closed; replacement protocol open |

---

## 4. Variable Lineage

| Variable | Source | Claims | Observed Range / Values | Definition Status | Audit Result |
|----------|--------|--------|-------------------------|-------------------|--------------|
| `neg_time_ms` | Phase 8 synchronized trace | C1.1, C1.9 | CPU/GPU per-step full precision | Clear within Phase 8 only | Comparator invalid for original-CPU claim |
| `total_step_ms` | Sum of Phase 8 neg/fwd/bwd/opt | C1.4, C1.9 | Per-step full precision | Incomplete end-to-end denominator | Cannot represent Collate-inclusive step |
| `epoch_time_ns` | C1-R1 throughput pass | C1.2-R1 | BL run mean 26.2785s; GPU 4.36981s | Integer ns; includes scheduler and partial batch | A-level paired repeat evidence |
| `neg_time_ns` full-batch epoch SD | C1-R1 trace pass, `ddof=0` | C1.3-R1 | BL run-level mean 3.1672ms; GPU 0.03629ms | Explicit full-batch filter | A-level dispersion evidence |
| `neg_time_ns` full-batch run mean | C1-R1 trace pass | C1.7-R1 | Six-run mean 3.0026ms | Integer ns; GPU readiness synchronized | A-level component evidence |
| `mrr` / `hits10` | Phase 9 evaluators | C1.5, C1.8 | Step 1 invalid; Step 2 sampled | Protocol-dependent | No paper quality claim |
| `gpu_mem_mb` | Whole-training `max_memory_allocated()` | C1.6 | 5818–5820MiB | Not sampler-specific | Required estimand missing |
| `pct` | Phase 6 aggregate profile | C1.9 | Negative Sampling=35.70% | Clear only in Phase 6 denominator | Not comparable to Phase 8 |

---

## 5. Cross-Claim Risk and Required Follow-Up

| Claim | Grade | Paper Risk | Required Action |
|-------|-------|------------|-----------------|
| C1.1 | B | High | Remove 198× from paper; rerun only if component result is required |
| C1.2-R1 | A | Low under frozen runtime-path wording | Regenerate result assets from C1-R1 derived CSV |
| C1.3-R1 | A | Medium if mislabeled as variance | Use standard-deviation/dispersion wording and the full-batch estimand |
| C1.4 | B | High | Remove Phase 8 8.5× from paper |
| C1.5 | C | High | No quality claim; corrected convergence study only if later required |
| C1.6 | C | Medium | Isolated sampler memory measurement |
| C1.7-R1 | A | Low under frozen GPU-path wording | Use repeat-level mean, sample SD, and CI |
| C1.8 | C | High | Exclude sampled values from manuscript |
| C1.9 | C | High | One-driver exhaustive component profiling |

### Paper-Section Impact

| Paper Section | Impact |
|---------------|--------|
| Abstract / Introduction headline | Historical 5.7× is superseded; 6.01× [5.94, 6.08] is eligible with the declared runtime-path wording |
| Method §3.4 GPU Runtime | Design and semantic description may remain; no quality-equivalence wording |
| Experiments §4.2 Main Results | 198× and 8.5× remain removed; regenerate Fig.5 from C1-R1 paired data |
| Experiments §4.3 Ablation | Replace 142× with 87.9× [72.9, 105.9] standard-deviation compression |
| Experiments §4.5 Bottleneck Shift | Pre/post percentage chart blocked until one-denominator profiling exists |

---

## 6. Remediation Order

1. **C1.2-R1/C1.3-R1/C1.7-R1**: closed at A under C1-R1 v1.1.
2. **C1.6 memory isolation**: add allocated/reserved/peak measurements around
   sampler generation within the same matched protocol.
3. **C1.9 unified profiling**: use identical exhaustive timing regions for both
   runtime paths.
4. **C1.5 quality**: remains out of scope unless the paper later chooses to
   pursue a pre-registered non-inferiority claim.

C1-R1 was subsequently authorized and executed. Preflight and all selected
jobs passed. The first throughput GPU attempt for seed 45 recorded
`sw_thermal_slowdown=Active` before warm-up; no measured-epoch snapshot was
active. The attempt was nevertheless excluded under the frozen rule, its
artifacts were retained, and the complete seed-45 throughput GPU→BL pair was
rerun once. Attempt 2 passed every check and is the selected evidence.

---

*End of Evidence Audit Part 2. The Part 1 Claim IDs and statuses remain
unchanged; this report supplies the independent A/B/C/D credibility verdicts.*
