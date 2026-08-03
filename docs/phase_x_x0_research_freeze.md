# Phase X X0 — Research Question, Scope, Contribution, and Estimand Freeze

**Freeze date:** 2026-08-03

**Authority:** approved X0 design spec and C1/C2 evidence audits
**Status:** canonical; no training or GPU execution is required

## 1. Freeze Authority and Chronology

This document is the canonical X0 truth source. The supporting design record is
[`docs/superpowers/specs/2026-08-03-phase-x-x0-rq-scope-contribution-estimand-freeze-design.md`](superpowers/specs/2026-08-03-phase-x-x0-rq-scope-contribution-estimand-freeze-design.md).

The paper adopts strategy A: a focused GPU runtime redesign. C1 is the sole
primary empirical contribution; C2 is supporting implementation architecture;
C3 and C4 remain conditional or exploratory until Parts 4 and 5 pass.

The chronology is explicit:

1. Phase 8 and Phase 9 supplied discovery-stage observations.
2. The C1-R1-v1.1 replacement protocol and analysis rules were frozen before
   the replacement run.
3. X0 is a post-result formalization of the paper-level questions after C1-R1
   results were available.

Therefore RQ1/RQ2 are primary research questions with frozen primary
estimands, not prospectively preregistered hypotheses. The manuscript must not
imply that X0 predates all observations.

## 2. Research Questions

### RQ1 — End-to-End Performance

> Under the frozen single-GPU muKG protocol, how much does the declared
> GPU-native negative-sampling runtime path change end-to-end epoch time
> relative to the original CPU baseline path?

### RQ2 — Runtime Dispersion

> Under the same protocol, how much does the declared GPU runtime path change
> within-epoch full-batch negative-sampling time dispersion relative to the CPU
> baseline?

RQ1 and RQ2 are separate because epoch time and within-epoch sampling-time
dispersion are different estimands.

### RQ3 — Implemented Framework Boundary

> Which implemented roles, interfaces, and execution boundaries constitute the
> audited two-stage runtime framework used to select and execute the declared
> CPU and GPU paths?

RQ3 is an implementation question. It does not claim framework novelty,
general superiority, DDP readiness, or validation across models, datasets, or
devices.

### Exploratory Questions

- **EQ1 — Cost-model validity:** Can the static cost model predict measured CPU
  negative-sampling cost under a provenance-complete, leakage-free protocol?
  This is pending Part 4 and must not inherit the historical R²=0.9008.
- **EQ2 — Scheduling effect:** Do cost-aware sorting and a genuinely distinct
  packing strategy independently change runtime dispersion? This is pending
  Part 5; the current fixture shows `FFDPacker == ChunkPacker`.
- **EQ3 — Performance–quality trade-off:** What training-quality trade-off
  accompanies the declared sampler-semantic change? This requires a new full-
  test, multi-run protocol.

### FINER Assessment

| RQ | Feasible | Interesting | Novel | Ethical | Relevant | Average |
|---|---:|---:|---:|---:|---:|---:|
| RQ1 | 5 | 5 | 3 | 5 | 5 | 4.6 |
| RQ2 | 5 | 4 | 3 | 5 | 4 | 4.2 |
| RQ3 | 5 | 3 | 2 | 5 | 4 | 3.8 |

Novelty is capped pending X1.5 literature and novelty audit. RQ3 is not a
standalone novelty claim.

## 3. Scope

### In Scope

- muKG implementation under study;
- FB15k-237 with the C1-R1-v1.1 split: 5,000 held out and 267,115 training
  triples;
- SimpleTransE with embedding dimension 400;
- one RTX 3070 execution environment;
- batch size 5,000 and 150 negatives per positive;
- original CPU Bernoulli/global-triple-collision runtime path;
- redesigned tail-only GPU sampler with batch-level positive-tail filtering;
- end-to-end epoch time;
- full-batch within-epoch negative-sampling-time dispersion; and
- the implemented two-stage, five-role runtime organization.

### Out of Scope

- CPU/GPU sampler semantic equivalence;
- quality equivalence or non-inferiority;
- general performance claims for all KGE models, datasets, GPUs, or frameworks;
- SOTA superiority over external systems;
- sampler-only VRAM, energy, power, or cost efficiency;
- multi-GPU or DDP readiness;
- automatic runtime-policy or backend selection;
- cross-model or cross-dataset empirical generalization; and
- a causal claim that hardware migration alone produced the speedup.

The implementation may be described as extensible where an actual extension
point exists. Empirical performance conclusions remain restricted to the
frozen protocol. The abstract must name muKG, SimpleTransE/TransE,
FB15k-237, and RTX 3070; it must not use unverified class-level phrases such
as “lightweight KGE models” or “mid-range GPUs.”

### Triple-Single Boundary

Current evidence has one model, one dataset, and one GPU model. This does not
invalidate E1/E2, but the Discussion must state that effects may change with
model compute intensity, graph scale and topology, negative-sampling pressure,
host-to-device balance, and GPU architecture.

No current artifact supplies publication-grade batch-size or negative-count
sensitivity. C1-R1 fixes batch size at 5,000 and negative count at 150; rounded
Phase 10 summaries cannot substitute for a sensitivity analysis.

## 4. Contribution Hierarchy

### Primary Contribution — C1

**Auditable GPU-native runtime-path redesign.** The work redesigns the muKG
negative-sampling runtime path for GPU-native execution under explicitly
different sampler semantics, and measures end-to-end epoch time and full-batch
sampling-time dispersion with paired, independent-seed, unrounded evidence.

The comparison object is the whole declared runtime path, not a claim that the
same sampler algorithm was moved unchanged between devices. C1 does not
include semantic equivalence, quality non-inferiority, sampler-only VRAM,
universal KGE generalization, or SOTA superiority.

### Supporting Contribution — C2

The canonical implementation contains the offline control plane
FeatureExtractor → CostModel → Cost Table and the online path
Scheduler → BatchProvider. The minimum viable paper contribution is narrower:
the shared online Scheduler/BatchProvider integration boundary with explicit
CPU/GPU backend selection in the training loop.

Offline cost-model roles enter the main Method only if Part 4 or Part 5
establishes that they are necessary to a retained paper claim. Otherwise they
remain implementation context or move to an appendix. RuntimePolicy and
GPUExecution are future concepts; rank-strided slicing is not DDP readiness.

### Conditional C3

C3 enters the contribution list only if Part 4 recovers the measured-cost
target and sampling unit, excludes target leakage, defines an out-of-sample
prediction estimand, reports uncertainty, and reaches paper-safe A/B evidence.
Otherwise the cost table is an implementation detail and R²=0.9008 or “90%
explained variance” is removed.

### Exploratory or Conditional C4

C4 enters as a secondary contribution only if Part 5 demonstrates genuinely
different sorter/packer behavior, separates sorter, packer, and interaction
effects, uses independent repeats, and establishes a protocol-limited
incremental effect. Otherwise the main path is Random+Chunk; CBP moves to an
exploratory ablation, appendix, research-history note, or is removed.

Unsupported C3/C4 results cannot be repackaged as “future-proof,” “key
enabler,” “visionary,” or equivalent contribution claims.

### Worst-Case Manuscript Backbone

The minimum viable manuscript assumes EQ1 and EQ2 remain exploratory. Its
Method backbone is:

1. profiled problem and frozen comparison boundary;
2. CPU/GPU sampler-semantics disclosure;
3. GPU-native sampling-path design;
4. integration through the shared Scheduler/BatchProvider online path; and
5. frozen measurement and statistical protocol.

The paper remains coherent as: identify bottleneck → define semantic redesign
boundary → implement GPU-native path → integrate explicitly → verify runtime
effects. It does not depend on CostModel or CBP success.

## 5. Primary and Supporting Estimands

### E1 — RQ1 End-to-End Epoch Speedup

| Field | Frozen definition |
|---|---|
| Target condition | C1-R1-v1.1; FB15k-237; 267,115 training triples; SimpleTransE dim=400; batch size 5,000; 150 negatives; RTX 3070 |
| Treatment | RandomSorter(42)+ChunkPacker+tail-only GPU sampler |
| Comparator | Shared Scheduler/BatchProvider path+original CPU Bernoulli/global-collision sampler |
| Independent unit | Paired seed-level job; seeds 42–47 |
| Nested observations | Five measured epochs per job |
| Per-run statistic | BL mean epoch time / GPU mean epoch time for the same seed |
| Cross-run summary | Geometric mean of six paired ratios |
| Uncertainty | Two-sided 95% t interval on paired log-ratios; df=5 |
| Partial batch | Included in the complete epoch |
| Observed estimate | 6.013×; 95% CI [5.944, 6.084] |
| Manuscript precision | 6.01×; 95% CI [5.94, 6.08] |

Epochs are nested observations, not independent replicates. No post-hoc
outlier removal is permitted.

### E2 — RQ2 Full-Batch Sampling-Time Dispersion Compression

| Field | Frozen definition |
|---|---|
| Target condition | C1-R1-v1.1 trace pass |
| Independent unit | Paired seed-level trace job; seeds 42–47 |
| Nested observations | Five epochs; 53 full batches per epoch |
| Primary filter | `is_partial == False AND batch_size_actual == 5000` |
| Per-epoch statistic | Population SD of `neg_time_ns` over 53 full batches; `ddof=0` |
| Per-run statistic | Arithmetic mean of five epoch SDs |
| Paired effect | BL run-level SD / GPU run-level SD |
| Cross-run summary | Geometric mean of six paired ratios |
| Uncertainty | Two-sided 95% t interval on paired log-ratios; df=5 |
| Observed estimate | 87.88×; 95% CI [72.92, 105.91] |
| Manuscript precision | 87.9×; 95% CI [72.9, 105.9] |

This is standard-deviation/dispersion compression. It is not variance
compression, between-run variance reduction, training stability, or quality
stability.

### E3 — Supporting GPU Full-Batch Component Time

The object is the GPU path full-batch negative-sampling mean. Each GPU
seed-level run contributes its mean full-batch `neg_time_ns`; the six run means
are summarized arithmetically with sample SD and a two-sided 95% t interval.
The observed estimate is **3.0026 ms**, sample SD **0.0229 ms**, 95% CI
**[2.9786, 3.0266] ms**. The manuscript uses 3.003 ms, SD 0.023 ms, CI
[2.979, 3.027] ms. E3 is secondary descriptive evidence.

### E4 — RQ3 Implementation Evidence Object

E4 is conjunctive implementation evidence, not a statistical estimand. It
requires that the five implemented roles exist with frozen interfaces,
Scheduler/BatchProvider are shared across configured CPU/GPU paths, backend
selection occurs explicitly in the training loop, scheduling is triggered on
each epoch iteration, rank-strided slicing passes coverage/disjointness
fixtures, RuntimePolicy/GPUExecution implementations do not exist, and DDP
readiness or automatic backend injection is not claimed.

## 6. Quality and External-Validity Boundaries

Link-prediction quality is diagnostic-only. Existing Phase 9 quality values do
not enter the manuscript. Loss is a finite-training sanity diagnostic only. A
future corrected study may report full filtered MRR and Hits@1/3/10 as a
performance–quality trade-off. “Equivalent,” “comparable,” “preserved,” and
“non-inferior” require a new protocol with a pre-specified margin and suitable
uncertainty.

Optional generalization is a separate gap-closing branch and creates new Claim
IDs and estimands:

1. cross-model, fixed dataset/hardware (for example RotatE or ConvE);
2. cross-dataset, fixed model/hardware;
3. cross-GPU-model single-GPU replication; and
4. multi-GPU scaling, which requires a real distributed path and scaling
   efficiency estimands.

The first three test generalization; the fourth tests parallel scalability.
Multi-GPU conditions must not be used to retroactively validate single-GPU E1
or E2.

## 7. C3/C4 Promotion Gates and Worst-Case Manuscript

Part 4 must establish C3 target provenance, sampling unit, leakage control,
out-of-sample prediction, and uncertainty before any predictive cost claim is
promoted. Part 5 must resolve `FFDPacker == ChunkPacker` and provide a true
sorter/packer factorial evidence path before CBP is promoted.

Until those gates pass, the paper follows the worst-case backbone in Section 4:
C1 is primary, C2 is an integration boundary, and CostModel/CBP are context or
exploration.

## 8. Optional Generalization Branches

Any optional cross-model, cross-dataset, cross-GPU, or multi-GPU study must
have its own frozen protocol, source manifest, raw observations, independent
unit, estimand, and uncertainty. It cannot alter E1/E2 or be described as a
retroactive correction to the C1-R1 evidence.

## 9. Manuscript Reporting Rules

- RQ wording contains no observed result.
- Abstract and Results report rounded, informative effect estimates and CIs;
  they do not hide results behind thresholds such as “greater than 5×.”
- Abstract identifies muKG, SimpleTransE/TransE, FB15k-237, and RTX 3070.
- The Introduction discloses protocol scope and non-equivalent sampler
  semantics when summarizing the headline result.
- Machine-readable audit artifacts retain full stored precision.
- Method is finalized after Part 5 decides whether C4 belongs in the main text,
  an exploratory ablation, an appendix, or nowhere.
- Historical 198×, 8.5×, 5.7×, and 142× values do not re-enter paper evidence.

## 10. Downstream Phase Gates

1. X1.5: systematic mapping and novelty audit for KGE runtime systems, GPU
   negative sampling, and modular reproducibility.
2. Part 4: C3 cost-model audit.
3. Part 5: C4 scheduler/packer audit.
4. Contribution triage: retain, demote, or remove C3/C4.
5. Targeted gap-closing experiments only after the preceding decisions.
6. Part 7: regenerate manuscript assets from canonical derived artifacts.
7. Artifact package and clean-room reproduction.
8. Final claim-reference and reviewer-integrity gate.
