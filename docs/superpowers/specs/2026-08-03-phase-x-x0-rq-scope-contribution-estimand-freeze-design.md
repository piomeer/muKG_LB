# Phase X X0 — RQ, Scope, Contribution, and Estimand Freeze Design

**Date:** 2026-08-03

**Status:** Revised after written-spec review; pending final review

**Mode:** Read-only research design and evidence governance

**Experiment boundary:** No training or GPU execution

**Code boundary:** No runtime or training-code changes

## 1. Purpose

X0 freezes the scientific identity of the paper before the remaining evidence
audits and manuscript rewrite. It separates:

1. what the paper asks;
2. what the current evidence can answer;
3. what remains conditional on Part 4 or Part 5;
4. how every primary effect is calculated; and
5. which interpretations are prohibited.

The selected paper strategy is **A: a focused GPU runtime redesign paper**.
C1 is the sole primary empirical contribution. C2 is supporting implementation
architecture. C3 and C4 remain conditional or exploratory until their
respective audits pass.

The minimum viable manuscript is designed under the conservative assumption
that C3 and C4 do not survive their audits. The paper must remain coherent and
complete with C1 as the sole empirical contribution.

## 2. Research Chronology and Epistemic Status

The paper must disclose the research chronology accurately:

1. Phase 8 and Phase 9 supplied discovery-stage observations.
2. The C1-R1-v1.1 replacement protocol and analysis rules were frozen before
   the replacement experiment was executed.
3. X0 formally states the paper-level research questions after the C1-R1
   results were available.

Consequently, RQ1 and RQ2 are **primary research questions with frozen primary
estimands**, but they are not described as prospectively preregistered
hypotheses. The manuscript must not imply that X0 predates all observations.

Research questions, estimands, and observed estimates are stored as separate
objects. Research-question wording is outcome-neutral.

## 3. Research Question Brief

### 3.1 Primary Research Questions

**RQ1 — End-to-end performance**

> Under the frozen single-GPU muKG protocol, how much does the declared
> GPU-native negative-sampling runtime path change end-to-end epoch time
> relative to the original CPU baseline path?

**RQ2 — Runtime dispersion**

> Under the same protocol, how much does the declared GPU runtime path change
> within-epoch full-batch negative-sampling time dispersion relative to the CPU
> baseline?

RQ1 and RQ2 are deliberately separate because epoch time and within-epoch
sampling-time dispersion are different estimands.

### 3.2 Supporting Research Question

**RQ3 — Implemented framework boundary**

> Which implemented roles, interfaces, and execution boundaries constitute the
> audited two-stage runtime framework used to select and execute the declared
> CPU and GPU paths?

RQ3 is an implementation and architecture question. It does not ask whether the
framework is novel, generally superior, DDP-ready, or validated across models,
datasets, or devices.

### 3.3 Exploratory Questions

**EQ1 — Cost-model validity**

> Can the static cost model predict measured CPU negative-sampling cost under a
> provenance-complete, leakage-free protocol?

EQ1 remains unanswered until Part 4. It must not inherit the historical
R²=0.9008 as an assumed result.

**EQ2 — Scheduling effect**

> Do cost-aware sorting and a genuinely distinct packing strategy independently
> change runtime dispersion?

EQ2 remains unanswered until Part 5. It cannot become a paper RQ while the
current frozen fixtures show `FFDPacker == ChunkPacker`.

**EQ3 — Performance–quality trade-off**

> What training-quality trade-off accompanies the declared sampler-semantic
> change?

EQ3 requires a corrected full-test, multi-run quality protocol. Existing
Phase 9 sampled quality values do not answer it.

### 3.4 FINER Assessment

| RQ | Feasible | Interesting | Novel | Ethical | Relevant | Average |
|---|---:|---:|---:|---:|---:|---:|
| RQ1 | 5 | 5 | 3 | 5 | 5 | 4.6 |
| RQ2 | 5 | 4 | 3 | 5 | 4 | 4.2 |
| RQ3 | 5 | 3 | 2 | 5 | 4 | 3.8 |

Novelty is capped pending X1.5. RQ3 is supporting evidence and is not presented
as a standalone novelty claim.

## 4. Scope Freeze

### 4.1 In Scope

- muKG as the implementation under study.
- FB15k-237 under the exact C1-R1-v1.1 data split.
- SimpleTransE with embedding dimension 400.
- One RTX 3070 execution environment.
- Batch size 5,000 and 150 negatives per positive.
- Original CPU Bernoulli/global-triple-collision runtime path.
- Redesigned tail-only GPU sampler with batch-level positive-tail filtering.
- End-to-end epoch time.
- Full-batch within-epoch negative-sampling-time dispersion.
- The implemented two-stage, five-role runtime organization.

### 4.2 Out of Scope

- Semantic equivalence between the CPU and GPU samplers.
- Quality equivalence or non-inferiority.
- General performance claims for all KGE models, datasets, GPUs, or frameworks.
- State-of-the-art superiority over DGL-KE, GraphVite, PyTorch-BigGraph,
  Marius, LibKGE, PyKEEN, or other external systems.
- Sampler-only VRAM, energy, power, or cost efficiency.
- Multi-GPU or DDP readiness.
- Automatic runtime-policy or backend selection.
- Cross-model or cross-dataset empirical generalization.
- A causal claim that hardware migration alone produced the observed speedup.

### 4.3 Generalization Rule

The implementation may be described as extensible where extension points
actually exist. Empirical performance conclusions remain restricted to the
frozen protocol. A title or abstract may mention KGE training only if it clearly
identifies the work as a muKG case study and does not universalize the measured
effects.

The abstract must name the tested model, dataset, and hardware rather than use
an unverified class-level phrase such as “lightweight KGE models” or
“mid-range GPUs.” The current evidence supports TransE on FB15k-237 on one
RTX 3070, not the surrounding model or hardware classes.

### 4.4 Triple-Single External-Validity Boundary

The current evidence has three coupled external-validity limits:

1. one model: SimpleTransE;
2. one dataset: FB15k-237; and
3. one GPU model: RTX 3070.

These limits do not invalidate E1 or E2, but they constrain publication scope
and venue fit. The Discussion and Limitations must state that the observed
effect may change with model compute intensity, graph scale and topology,
negative-sampling pressure, host-to-device balance, and GPU architecture.

No current artifact establishes batch-size or negative-count sensitivity under
the C1-R1 standard. C1-R1 fixes batch size at 5,000 and negative count at 150.
The rounded Phase 10 sensitivity summaries cannot be substituted for a
publication-grade sensitivity analysis.

## 5. Contribution Freeze

### 5.1 Primary Contribution — C1

**Auditable GPU-native runtime-path redesign**

- Redesigns the muKG negative-sampling runtime path for GPU-native execution
  under explicitly different sampler semantics.
- Uses paired, independent-seed, unrounded observations to estimate
  end-to-end epoch speedup and full-batch sampling-time dispersion compression.
- Treats the whole declared runtime path as the comparison object.

It does not include semantic equivalence, quality non-inferiority,
sampler-only VRAM, universal KGE generalization, or SOTA superiority.

### 5.2 Supporting Contribution — C2

**Implemented and audited runtime organization**

- The canonical implementation contains the audited offline control plane
  FeatureExtractor → CostModel → Cost Table and online path
  Scheduler → BatchProvider.
- The minimum viable paper contribution is narrower: a shared online
  Scheduler/BatchProvider integration boundary with explicit CPU/GPU backend
  selection in the training loop.
- Offline cost-model roles enter the main Method only if Part 4 or Part 5
  establishes that they are necessary to a retained paper claim. Otherwise,
  they remain implementation context or move to an appendix.

The contribution is limited to implemented interfaces, boundaries, and
auditable composition. RuntimePolicy and GPUExecution remain future concepts.
Rank-strided slicing is not described as DDP readiness.

### 5.3 Conditional Contribution — C3

C3 may enter the paper's contribution list only if Part 4:

1. recovers the measured-cost target and sampling unit;
2. excludes target leakage;
3. defines an out-of-sample prediction estimand;
4. reports suitable uncertainty; and
5. reaches paper-safe A- or B-level evidence.

If these conditions fail, the cost table remains an implementation detail.
The paper removes R²=0.9008, “90% explained variance,” and predictive
contribution wording.

### 5.4 Exploratory or Conditional Contribution — C4

C4 may become a secondary contribution only if Part 5:

1. demonstrates that the compared sorter and packer implementations perform
   genuinely different operations or layouts;
2. separates sorter, packer, and interaction effects;
3. uses independent repeats; and
4. establishes a protocol-limited incremental effect.

If these conditions fail, the main path uses Random+Chunk. The Scheduler
interface remains in the implementation description, while CBP moves to an
exploratory ablation, research-history note, appendix, or is removed.

Failed or unsupported C3/C4 results must not be repackaged as “future-proof,”
“key enabler,” “visionary,” or equivalent contribution claims.

### 5.5 Worst-Case Manuscript Backbone

Until Part 4 and Part 5 pass their promotion gates, manuscript planning assumes
that EQ1 and EQ2 remain exploratory. The minimum Method backbone is:

1. profiled problem and frozen comparison boundary;
2. CPU and GPU sampler-semantics disclosure;
3. GPU-native sampling-path design;
4. integration through the shared Scheduler/BatchProvider online path; and
5. frozen measurement and statistical protocol.

The primary narrative is therefore:

> identify the runtime bottleneck → define the semantic redesign boundary →
> implement the GPU-native path → integrate it explicitly → verify its
> end-to-end and dispersion effects.

The narrative does not depend on CostModel or CBP success. If Part 4 or Part 5
later passes, the supported material is added as a secondary branch rather than
used to repair the primary story.

## 6. Estimand Freeze

### 6.1 E1 — RQ1 End-to-End Epoch Speedup

| Field | Frozen Definition |
|---|---|
| Target condition | C1-R1-v1.1; FB15k-237; 267,115 training triples; SimpleTransE dim=400; batch size 5,000; 150 negatives; RTX 3070 |
| Treatment | RandomSorter(42) + ChunkPacker + tail-only GPU sampler |
| Comparator | Shared Scheduler/BatchProvider path + original CPU Bernoulli/global-collision sampler |
| Independent unit | Paired seed-level job; seeds 42–47 |
| Nested observations | Five measured epochs per job |
| Per-run statistic | BL mean epoch time divided by GPU mean epoch time for the same seed |
| Cross-run summary | Geometric mean of six paired ratios |
| Uncertainty | Two-sided 95% t interval on paired log-ratios; df=5 |
| Partial batch | Included as part of the complete epoch |
| Primary observed estimate | 6.013×, 95% CI [5.944, 6.084] |
| Manuscript precision | 6.01×, 95% CI [5.94, 6.08] |

Epochs are nested observations, not independent replicates. No post-hoc outlier
removal is permitted.

### 6.2 E2 — RQ2 Full-Batch Sampling-Time Dispersion Compression

| Field | Frozen Definition |
|---|---|
| Target condition | C1-R1-v1.1 trace pass |
| Independent unit | Paired seed-level trace job; seeds 42–47 |
| Nested observations | Five epochs per job; 53 full batches per epoch |
| Primary filter | `is_partial == False AND batch_size_actual == 5000` |
| Per-epoch statistic | Population SD of `neg_time_ns` over 53 full batches; `ddof=0` |
| Per-run statistic | Arithmetic mean of the five epoch SDs |
| Paired effect | BL run-level SD divided by GPU run-level SD |
| Cross-run summary | Geometric mean of six paired ratios |
| Uncertainty | Two-sided 95% t interval on paired log-ratios; df=5 |
| Primary observed estimate | 87.88×, 95% CI [72.92, 105.91] |
| Manuscript precision | 87.9×, 95% CI [72.9, 105.9] |

This is standard-deviation or dispersion compression. It is not variance
compression, between-run variance reduction, training stability, or quality
stability.

### 6.3 E3 — Supporting GPU Full-Batch Component Time

| Field | Frozen Definition |
|---|---|
| Object | GPU-path full-batch negative-sampling mean |
| Independent unit | GPU seed-level run |
| Per-run statistic | Mean full-batch `neg_time_ns` within each run |
| Cross-run summary | Arithmetic mean of six run means |
| Uncertainty | Sample SD and two-sided 95% t interval |
| Observed estimate | 3.0026 ms; sample SD 0.0229 ms; 95% CI [2.9786, 3.0266] ms |
| Manuscript precision | 3.003 ms; sample SD 0.023 ms; 95% CI [2.979, 3.027] ms |
| Status | Secondary descriptive estimand |

### 6.4 E4 — RQ3 Implementation Evidence Object

E4 is a conjunctive implementation-evidence object, not a statistical
estimand. It requires evidence that:

1. the five implemented roles exist with the frozen interfaces;
2. Scheduler and BatchProvider are shared across configured CPU/GPU paths;
3. backend selection occurs explicitly in the training loop;
4. scheduling is triggered on each epoch iteration;
5. rank-strided slicing passes coverage and disjointness fixtures;
6. RuntimePolicy and GPUExecution implementations do not exist; and
7. DDP readiness, automatic backend injection, and empirical validation of all
   four sorter×packer combinations are not claimed.

## 7. Quality-Outcome Freeze

Link-prediction quality is diagnostic-only in X0:

- Existing Phase 9 quality values do not enter the manuscript.
- Loss must be finite and is used only as a training sanity diagnostic.
- A later corrected study may report full filtered MRR and Hits@1/3/10 as a
  performance–quality trade-off.
- “Equivalent,” “comparable,” “preserved,” and “non-inferior” are prohibited
  without a pre-specified margin and suitable design.

## 8. Statistical Reporting Rules

- RQ1 and RQ2 are separate primary estimands.
- Each receives its effect size and two-sided 95% CI.
- No unregistered combined p-value or global hypothesis is constructed.
- The manuscript does not substitute “statistically significant” for the
  effect estimate and uncertainty.
- Batches and epochs are not promoted to independent replicates.
- Sensitivity analyses are labeled and cannot replace the primary definition.
- No post-hoc outlier exclusion is allowed.
- Quality, VRAM, energy, cross-model, and cross-dataset outcomes have no frozen
  primary estimand and cannot silently enter the conclusion.

## 9. Manuscript Reporting Rules

- RQ wording contains no observed result.
- The abstract and Results report rounded, informative effect estimates and
  confidence intervals rather than hiding them behind thresholds such as
  “greater than 5×.”
- The abstract identifies the empirical setting as muKG with TransE on
  FB15k-237 and an RTX 3070. It does not imply validation for a model class,
  dataset class, hardware class, or general KGE framework.
- Machine-readable audit artifacts retain full stored precision.
- The Introduction may summarize the headline result but must disclose the
  protocol scope and non-equivalent sampler semantics.
- The Method is finalized only after Part 5 determines whether C4 belongs in
  the main text, an exploratory ablation, an appendix, or nowhere.
- Old 198×, 8.5×, 5.7×, and 142× values do not re-enter paper evidence.

## 10. Optional External-Validity Expansion

External-validity work is a separate gap-closing branch and does not modify the
definitions of E1 or E2. Every added cell requires a new Claim ID, protocol,
raw observations, and estimand.

Priority order, if time and laboratory access permit:

1. **Cross-model, fixed dataset and hardware:** repeat the declared runtime-path
   comparison with at least one materially different KGE model. RotatE probes a
   different scoring cost; ConvE probes a more compute-intensive architecture.
2. **Cross-dataset, fixed model and hardware:** repeat with a graph that differs
   in scale or topology. The dataset must be chosen after a memory and runtime
   preflight.
3. **Cross-hardware, fixed model and dataset:** execute the same single-GPU
   protocol on a second GPU model. This tests hardware sensitivity more directly
   than multi-GPU scaling.
4. **Multi-GPU scaling:** treat as a separate future research question. It
   requires a real distributed execution path, communication boundaries, and
   scaling-efficiency estimands. Rank-strided slicing alone is insufficient.

The first three branches test generalization. The fourth tests parallel
scalability. They must not be conflated.

If no expansion is completed before submission, the paper proceeds with the
minimum viable scope and carries the triple-single boundary as an explicit
limitation rather than implying unmeasured stability.

## 11. X0 Acceptance Criteria

X0 is complete when:

1. the canonical freeze reproduces this design without contradictions;
2. RQ, scope, contribution, and estimand fields are explicitly separated;
3. every primary estimand identifies its independent unit and nested
   observations;
4. discovery, replacement verification, and post-result formalization are
   distinguished;
5. C3/C4 promotion gates and failure paths are explicit;
6. prohibited interpretations are listed;
7. the worst-case manuscript remains coherent if C3 and C4 fail;
8. triple-single limitations and optional generalization branches are distinct;
9. PROGRESS and project memory point to the freeze; and
10. no training code, runtime code, GPU experiment, paper Method, story freeze,
   or historical audit register is modified.
