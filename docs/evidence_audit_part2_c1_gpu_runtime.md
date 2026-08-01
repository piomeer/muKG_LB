# Evidence Audit Part 2 — C1 GPU Runtime

**Version**: 1.0
**Date**: 2026-08-01
**Status**: Complete
**Scope**: C1.1–C1.9 from `docs/evidence_audit_part1_claim_inventory.md`
**Execution Boundary**: Existing artifacts and static source inspection only; no
training or GPU experiment was run

---

## 1. Executive Verdict

| Grade | Count | Claims |
|-------|------:|--------|
| A (Verified) | 0 | — |
| B (Re-analysis) | 2 | C1.1, C1.4 |
| C (Re-experiment) | 7 | C1.2, C1.3, C1.5, C1.6, C1.7, C1.8, C1.9 |
| D (Invalid) | 0 | — |

No current C1 claim satisfies the frozen publication-level A standard:
unrounded raw observations, a symmetric and frozen estimand, at least three
independent repeats, repeat-level uncertainty, valid generating code, and
paper wording limited to the audited protocol.

The main conclusions are:

1. **Phase 8's 198×/8.5× results are removed from paper evidence.** The CPU
   comparator is a synthetic validation function that simultaneously replaces
   head and tail and does not perform the original global collision check.
2. **The Phase 9 summaries do contain 25.1s and 4.4s, yielding 5.7045×.**
   This remains a C-level headline because the Phase 9 result is a single run
   and the Phase 10 repeats discarded full precision before statistical
   analysis.
3. **The reported 142× is not a repeat-level variance result.** It is
   28.5/0.2=142.5 from two rounded, final-epoch, within-run population standard
   deviations. Rounding alone permits a ratio from 113.8× to 190.3×.
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

### C1.2 — Phase 9 End-to-End Epoch Speedup

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.2 |
| Inventory Status | ACTIVE |
| Claim Statement | Under Phase 9 Step 2, GPU reports 25.1s → 4.4s average epoch time (5.7×) relative to BL |
| Claim Type | End-to-end performance |
| Frozen Protocol | FB15k-237; held-out 5k shuffled training triples; TransE dim=400; RandomSorter+ChunkPacker; batch_size=5000; neg_num=150; five epochs per configuration |
| Supporting Figure/Table | Fig.5 and Table 2; Fig.5 values are hardcoded |
| Primary Raw Evidence | Per-configuration Phase 9 Step 2 epoch summaries; no unrounded per-step/epoch timing trace |
| Derived Evidence | Mean of stored epochs: BL=25.1s, GPU=4.4s, ratio=5.704545×; possible ratio from 0.1s rounding alone is 5.6292×–5.7816× |
| Supporting Script | `src/py/experiments/phase9_step2_benchmark.py`; Phase 10 is context only |
| Key Variables | `epoch_time_s`, configuration, run/seed, sampler semantics |
| Variable Trustworthiness | Direct epoch observations exist but were rounded before storage; Phase 10 repeat timings were also rounded before statistics |
| Metric / Estimand Definition | Mean of five rounded epoch times within one Phase 9 execution; not a mean across independent runs |
| Statistical Method Audit | Does not meet the ≥3 independent matched-repeat rule. Phase 10 stores 3 BL and 5 GPU summaries and yields 5.8258×, but lost precision prevents valid CI/zero-variance claims |
| Code Audit | `hash(label)` is process-dependent; the Phase 9 driver does not set `torch.manual_seed()` before model initialization |
| Semantic / Fairness Audit | Phase 9 correctly uses the original CPU implementation for BL and the frozen tail-only GPU design, but their semantic difference must be disclosed |
| Conclusion | **C — Re-experiment** |
| Paper-safe Wording | Audit only: the stored Phase 9 summaries report BL=25.1s and GPU=4.4s |
| Fix Recommendation | Run matched BL/GPU at ≥3 seeds, retain full-precision per-epoch timing, freeze warm-up, compute run-level speedups, and report repeat-level uncertainty |
| Responsible / Status | Codex; audit closed; headline held pending re-experiment |

### C1.3 — Phase 9 Negative-Sampling Dispersion Ratio

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.3 |
| Inventory Status | ACTIVE |
| Claim Statement | Final-epoch within-epoch neg-time std is 28.5ms for BL and 0.2ms for GPU (reported 142×) |
| Claim Type | Runtime dispersion |
| Frozen Protocol | Phase 9 Step 3; ten epochs; batch_size=5000; neg_num=150; BL vs GPU; final epoch selected after observing all epochs |
| Supporting Figure/Table | Fig.6 and Table 3; Fig.6 reads rounded final summary rows |
| Primary Raw Evidence | No per-step Phase 9 trace exists |
| Derived Evidence | Final summary ratio 28.5/0.2=142.5×; rounding interval 113.8×–190.3333×; mean of the ten stored epoch stds is 30.31/0.37=81.9189× |
| Supporting Script | `src/py/experiments/phase9_step3_ablation.py` |
| Key Variables | `neg_time_std_ms`, `neg_times`, final epoch, batch length |
| Variable Trustworthiness | Stored at 0.1ms; denominator is only 0.2ms; batch length is absent |
| Metric / Estimand Definition | `numpy.std(neg_times)` with `ddof=0` across batches inside one epoch, including the short final batch |
| Statistical Method Audit | Batch dispersion is not repeat uncertainty; final-epoch selection and rounding make the ratio unstable; no raw data exist to remove the partial batch |
| Code Audit | Computation is `np.std` over every batch in the epoch and writes only a one-decimal summary |
| Semantic / Fairness Audit | Different CPU/GPU samplers are acceptable only as declared runtime paths; “eliminates variance” would overgeneralize |
| Conclusion | **C — Re-experiment** |
| Paper-safe Wording | Audit only: rounded final-epoch summaries contain BL=28.5ms and GPU=0.2ms |
| Fix Recommendation | Save unrounded per-step time and batch size for matched repeats; compute full-size-batch within-run dispersion per run and between-run uncertainty separately |
| Responsible / Status | Codex; audit closed; 142× paper claim held |

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

### C1.7 — GPU Negative-Sampling Mean Range

| Field | Audit Finding |
|-------|---------------|
| Claim ID | C1.7 |
| Inventory Status | ACTIVE |
| Claim Statement | Phase 9 reports GPU neg-sampling means of 2.9–3.4ms across ten epochs, with post-warm-up epochs near 2.9–3.2ms |
| Claim Type | Component performance / stability |
| Frozen Protocol | Phase 9 Step 3 GPU configuration; ten epochs; batch_size=5000; neg_num=150 |
| Supporting Figure/Table | Fig.6 and Table 3 |
| Primary Raw Evidence | No per-step trace |
| Derived Evidence | Stored epoch-summary minimum/maximum=2.9/3.4ms; after excluding epoch 0=2.9/3.2ms |
| Supporting Script | `src/py/experiments/phase9_step3_ablation.py` |
| Key Variables | `neg_time_mean_ms`, epoch, warm-up, batch length |
| Variable Trustworthiness | Values are rounded to 0.1ms and include the final partial batch |
| Metric / Estimand Definition | Range of ten within-run epoch means; “post-warm-up” is a sensitivity label, not a pre-frozen rule |
| Statistical Method Audit | Ten epochs are not ten independent runs; no unrounded uncertainty |
| Code Audit | Per-step data are discarded after each rounded epoch row is written |
| Semantic / Fairness Audit | Valid only for the redesigned GPU tail-only path under this protocol |
| Conclusion | **C — Re-experiment** |
| Paper-safe Wording | Audit only: rounded Phase 9 epoch summaries range from 2.9ms to 3.4ms |
| Fix Recommendation | Retain unrounded per-step traces in the matched repeat protocol for C1.2/C1.3 and summarize steady-state full-size batches |
| Responsible / Status | Codex; audit closed; claim held |

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
| `epoch_time_s` | Phase 9/10 runner | C1.2 | 25.1s and 4.4s in Phase 9 | Clear, but stored at 0.1s | Repeat uncertainty lost |
| `neg_time_std_ms` | `np.std(neg_times)`, `ddof=0` | C1.3 | 28.5ms and 0.2ms final epoch | Clear formula; mixed batch sizes | Not repeat-level uncertainty |
| `neg_time_mean_ms` | Phase 9 epoch summary | C1.7 | 2.9–3.4ms | Rounded; raw steps absent | Requires re-experiment |
| `mrr` / `hits10` | Phase 9 evaluators | C1.5, C1.8 | Step 1 invalid; Step 2 sampled | Protocol-dependent | No paper quality claim |
| `gpu_mem_mb` | Whole-training `max_memory_allocated()` | C1.6 | 5818–5820MiB | Not sampler-specific | Required estimand missing |
| `pct` | Phase 6 aggregate profile | C1.9 | Negative Sampling=35.70% | Clear only in Phase 6 denominator | Not comparable to Phase 8 |

---

## 5. Cross-Claim Risk and Required Follow-Up

| Claim | Grade | Paper Risk | Required Action |
|-------|-------|------------|-----------------|
| C1.1 | B | High | Remove 198× from paper; rerun only if component result is required |
| C1.2 | C | High | Matched unrounded CPU/GPU repeat benchmark |
| C1.3 | C | High | Per-step, per-batch-size traces plus independent repeats |
| C1.4 | B | High | Remove Phase 8 8.5× from paper |
| C1.5 | C | High | No quality claim; corrected convergence study only if later required |
| C1.6 | C | Medium | Isolated sampler memory measurement |
| C1.7 | C | Medium | Fold into the matched C1.2/C1.3 repeat protocol |
| C1.8 | C | High | Exclude sampled values from manuscript |
| C1.9 | C | High | One-driver exhaustive component profiling |

### Paper-Section Impact

| Paper Section | Impact |
|---------------|--------|
| Abstract / Introduction headline | 5.7× must remain on hold until the C1.2 replacement protocol reaches A |
| Method §3.4 GPU Runtime | Design and semantic description may remain; no quality-equivalence wording |
| Experiments §4.2 Main Results | 198× and 8.5× removed; Fig.5 must be regenerated from audited data after rerun |
| Experiments §4.3 Ablation | 142× cannot be presented as repeat-level variance compression |
| Experiments §4.5 Bottleneck Shift | Pre/post percentage chart blocked until one-denominator profiling exists |

---

## 6. Remediation Order

1. **C1.2/C1.3/C1.7 combined protocol**: matched BL/GPU runs, ≥3 seeds,
   unrounded per-step and per-epoch data, explicit batch length and warm-up,
   run-level speedup and dispersion uncertainty.
2. **C1.6 memory isolation**: add allocated/reserved/peak measurements around
   sampler generation within the same matched protocol.
3. **C1.9 unified profiling**: use identical exhaustive timing regions for both
   runtime paths.
4. **C1.5 quality**: remains out of scope unless the paper later chooses to
   pursue a pre-registered non-inferiority claim.

No remediation experiment was authorized or executed in Part 2.

---

*End of Evidence Audit Part 2. The Part 1 Claim IDs and statuses remain
unchanged; this report supplies the independent A/B/C/D credibility verdicts.*
