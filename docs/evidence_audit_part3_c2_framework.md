# Evidence Audit Part 3 — C2 Unified Runtime Framework

**Version**: 1.0
**Date**: 2026-08-03
**Scope**: C2.1–C2.6
**Method**: Read-only source/artifact inspection, AST checks, SHA-256 lineage,
deterministic CPU fixtures, and independent recomputation
**Experiment policy**: No GPU or training run

## 1. Outcome

The canonical architecture is frozen as a **two-stage architecture with five
implemented roles**:

- offline control plane: FeatureExtractor → CostModel → Cost Table;
- online per-epoch path: Scheduler → BatchProvider;
- external training loop: explicitly selects the CPU or GPU sampling backend.

`RuntimePolicy` and `GPUExecution` are Future Extensions, not current
implementation roles. DDP wording is limited to rank-strided partitioning.

| Claim | Inventory relationship | Grade | Disposition |
|---|---|:---:|---|
| C2.1-R1 | Replaces held C2.1 without editing Part 1 | **A** | Canonical implementation claim |
| C2.2 | Original ACTIVE ID retained | **B** | Four configs verified; artifact lineage requires re-analysis |
| C2.3 | Original ACTIVE ID retained | **A** | Shared scheduling path and explicit backend selection verified |
| C2.4 | Original ACTIVE ID retained | **A** | Deterministic construction/lookup implementation claim only |
| C2.5 | Original ACTIVE ID retained | **A** | Iterator and rank-strided partition API verified |
| C2.6 | Original RETRACTED ID retained | **D** | Historical overhead claim remains invalid |

Part 1 remains the frozen inventory. The `-R1` suffix records that C2.1 was
reframed after resolving the inconsistent architecture descriptions.

## 2. Audit Scope and Evidence Precedence

The audit gives source and raw artifacts precedence over narrative documents.
The authoritative machine-readable outputs are:

- `output/results/evidence_audit_part3/source_manifest.json`
- `output/results/evidence_audit_part3/architecture_mapping.csv`
- `output/results/evidence_audit_part3/recomputed_metrics.csv`
- `output/results/evidence_audit_part3/audit_checks.json`

The source manifest records the SHA-256, size, type, schema/symbols, and where
applicable row count of every inspected input. It contains no dynamic timestamp.

## 3. Claim-Level Audit

### C2.1-R1 — Canonical implementation architecture

| Field | Audit result |
|---|---|
| Claim ID | C2.1-R1 (replacement for held C2.1) |
| Inventory status | C2.1 was HOLD; Part 1 is unchanged |
| Claim type | Architecture / implementation |
| Claim statement | The implementation comprises an offline FeatureExtractor–CostModel–Cost Table control plane and an online per-epoch Scheduler–BatchProvider path; the external training loop selects the sampling backend. |
| Frozen protocol | Source state hashed in `source_manifest.json`; CPU-only fixtures; no runtime experiment |
| Primary evidence | `src/py/load/features.py`; `cost_model.py`; `schedulers.py`; `batch_provider.py`; Phase 9 Step 2 driver |
| Derived evidence | `architecture_mapping.csv`; `audit_checks.json` facts and fixtures |
| Key variables/interfaces | `candidate_size`, `degree`, `hub_flag`, `cost_table`, `pack_batches()`, `iterate()` |
| Variable trustworthiness | Interface names, fields, and return shapes are directly visible in source and checked by AST/fixture |
| Metric/estimand | Presence and behavior of implemented roles; not a performance estimand |
| Statistical audit | Not applicable to an implementation-existence claim |
| Code audit | Role symbols present; scheduling occurs within each `iterate()` call; backend branch is external |
| Semantic/fairness audit | “Role” replaces inconsistent “layer” terminology; Future Extensions are excluded |
| Conclusion | **A** |
| Paper-safe wording | “The implementation comprises an offline FeatureExtractor–CostModel–cost-table control plane and an online Scheduler–BatchProvider path; the training loop selects the negative-sampling backend.” |
| Minimum remedy | None for this claim; Part 7 must apply the frozen wording and figure specification |
| Responsible/status | Phase X / closed for Part 3 |

### C2.2 — One driver enumerates four configurations

| Field | Audit result |
|---|---|
| Claim ID | C2.2 |
| Inventory status | ACTIVE |
| Claim type | Experiment-driver implementation / artifact lineage |
| Claim statement | The Phase 9 Step 2 driver defines BL, CBP, GPU, and CBP+GPU in one configuration loop, but its checked-in aggregation lineage is not self-consistent. |
| Frozen protocol | Read-only AST inspection of `phase9_step2_benchmark.py`; existing artifacts only |
| Primary evidence | Driver `configs` assignment and `run_config()`/final aggregation paths |
| Derived evidence | `audit_checks.json`: exact sorted labels `BL`, `CBP`, `CBP+GPU`, `GPU`; write suffix `.md`; read suffix `.csv` |
| Key variables/interfaces | `configs`, `label`, `use_gpu`, `sorter`, `packer`, `summary_path`, `sp` |
| Variable trustworthiness | Config tuples are unambiguous; artifact conversion from `.md` to `.csv` is not represented in the driver |
| Metric/estimand | Exact configuration enumeration and source-to-aggregate lineage |
| Statistical audit | Performance values are outside this C2 implementation claim |
| Code audit | Four configurations pass; per-config writer opens `summary.md`, while final aggregation attempts each `summary.csv` |
| Semantic/fairness audit | The four labels differ in both scheduler pair and sampling backend as declared |
| Conclusion | **B** |
| Paper-safe wording | “The Phase 9 Step 2 driver defines BL, CBP, GPU, and CBP+GPU in one configuration loop; its checked-in per-configuration artifact lineage requires reconciliation.” |
| Minimum remedy | Repair or document the `.md`→`.csv` conversion lineage, hash the actual per-config inputs, and regenerate the aggregate from those inputs |
| Responsible/status | Part 7 or artifact repair / open |

The B grade is not a denial that the driver contains four configurations. It
prevents the driver alone from proving a self-contained artifact lineage.

### C2.3 — Shared scheduling path, explicit sampler selection

| Field | Audit result |
|---|---|
| Claim ID | C2.3 |
| Inventory status | ACTIVE |
| Claim type | Architecture / interface composition |
| Claim statement | CPU and GPU configurations share Scheduler and BatchProvider; the training loop explicitly selects the sampling backend. |
| Frozen protocol | Source AST plus deterministic CPU composition fixture |
| Primary evidence | `schedulers.py`, `batch_provider.py`, Phase 9 Step 2 `use_gpu` branch |
| Derived evidence | `audit_checks.json` backend-selection and scheduling-interface facts |
| Key variables/interfaces | `Scheduler(sorter, packer)`, `BatchProvider(...)`, `provider.iterate()`, `sampler_obj.generate()`, CPU sampling call |
| Variable trustworthiness | The branch and calls are syntactically explicit |
| Metric/estimand | Shared interface path and location of backend selection |
| Statistical audit | Not applicable |
| Code audit | One Scheduler/BatchProvider construction precedes the loop; `if use_gpu` selects GPU generate versus CPU function plus transfer |
| Semantic/fairness audit | Selection is configuration-driven, not automatic injection or semantic equivalence |
| Conclusion | **A** |
| Paper-safe wording | “Both configured sampling paths consume batches from the same Scheduler and BatchProvider interfaces, while the training loop explicitly selects the CPU or GPU sampling backend.” |
| Minimum remedy | None; prohibit “transparent”, “automatic”, and “drop-in” wording |
| Responsible/status | Phase X / closed |

### C2.4 — Deterministic CostModel construction and lookup

| Field | Audit result |
|---|---|
| Claim ID | C2.4 |
| Inventory status | ACTIVE |
| Claim type | Implementation |
| Claim statement | `build_cost_table()` deterministically constructs a float32 array from static features and constants; scheduling then uses array lookup. |
| Frozen protocol | Two identical CPU calls with a six-entity fixture; `neg_num=150`; source AST |
| Primary evidence | `features.py`, `cost_model.py`, `schedulers.py`, checked-in `cost_table.npy` |
| Derived evidence | Fixture flags and source schema in `audit_checks.json`/`source_manifest.json` |
| Key variables/interfaces | `candidate_size`, `neg_num`, `max_try`, `b3_const`, `cost_table[e]` |
| Variable trustworthiness | Fixture confirms byte identity, dtype `float32`, and shape `[6]`; formula is explicit |
| Metric/estimand | Implementation determinism and access form, not prediction error or end-to-end overhead |
| Statistical audit | Not applicable; predictive validity belongs to Part 4 |
| Code audit | Function allocates `np.float32`, iterates deterministically, and stores one value per entity |
| Semantic/fairness audit | No “zero overhead”, “negligible end-to-end overhead”, or learned-regression wording |
| Conclusion | **A** |
| Paper-safe wording | “The implemented cost model deterministically constructs a float32 cost table from static features and configured constants, after which scheduling uses array lookups.” |
| Minimum remedy | None for implementation behavior; Part 4 must audit feature legitimacy and predictive validity |
| Responsible/status | Phase X / closed |

### C2.5 — Iterator and rank-strided partition API

| Field | Audit result |
|---|---|
| Claim ID | C2.5 |
| Inventory status | ACTIVE |
| Claim type | Interface / partition behavior |
| Claim statement | BatchProvider yields preassembled positive-triple lists and supports rank-strided batch partitioning. |
| Frozen protocol | Eleven unique triples, batch size four, three CPU ranks, logging disabled |
| Primary evidence | `BatchProvider.iterate()` and `set_rank()` source |
| Derived evidence | CPU fixtures: full coverage, pairwise-disjoint rank partitions, complete union |
| Key variables/interfaces | `triples_list`, `all_batches`, `_rank`, `_world_size`, `all_batches[rank::world_size]` |
| Variable trustworthiness | Fixture triples and batches are discrete and exactly comparable |
| Metric/estimand | Set coverage and partition disjointness |
| Statistical audit | Deterministic functional property; no repeat inference required |
| Code audit | `iterate()` yields each selected batch; the rank slice is explicit |
| Semantic/fairness audit | BatchProvider is not DataLoader and does not yield or select a sampler |
| Conclusion | **A** |
| Paper-safe wording | “BatchProvider yields preassembled triple-list batches and exposes rank-strided batch partitioning whose tested partitions are disjoint and collectively complete.” |
| Minimum remedy | None for the API claim; do not claim DDP readiness without end-to-end distributed implementation and validation |
| Responsible/status | Phase X / closed |

### C2.6 — Historical scheduler-overhead claim

| Field | Audit result |
|---|---|
| Claim ID | C2.6 |
| Inventory status | RETRACTED |
| Claim type | Performance / overhead |
| Claim statement | The historical “~0.5ms per epoch” scheduler-overhead claim is invalid and remains excluded. |
| Frozen protocol | Existing Phase 6 logs and C1-R1 throughput `per_epoch.csv`; no cross-phase pooling |
| Primary evidence | Phase 6 Baseline/CBP training logs; 6 seeds × 5 epochs per C1-R1 BL/GPU configuration |
| Derived evidence | `recomputed_metrics.csv` |
| Key variables/interfaces | `scheduler_overhead_ns`, `epoch_time_ns`, logged `overhead=...ms` |
| Variable trustworthiness | C1-R1 fields have raw integer nanoseconds; Phase 6 values are printed to three decimals |
| Metric/estimand | Within-protocol scheduler time and ratio of mean scheduler time to mean epoch time |
| Statistical audit | Phase 6 values are single recorded epochs; C1-R1 has 30 epoch observations per config nested in six runs; values are descriptive, not an independent-repeat performance Claim |
| Code audit | Phase 6 log values parsed directly; C1-R1 means, sample SDs, ranges, and ratios independently recomputed |
| Semantic/fairness audit | Phase 6 and C1-R1 timing boundaries/configurations differ and are never concatenated or directly compared |
| Conclusion | **D** |
| Paper-safe wording | “No universal scheduler-overhead claim is retained; observed overheads are reported only with their phase, configuration, timing boundary, and aggregation.” |
| Minimum remedy | Keep `~0.5ms` removed; any future claim needs a frozen timing boundary, raw precision, and independent repeated runs |
| Responsible/status | Retraction closed; replacement claim intentionally not created |

## 4. Protocol-Specific Overhead Descriptions

Every number below maps to a `metric_id` in `recomputed_metrics.csv`.

| Metric ID | Protocol-specific result | Interpretation limit |
|---|---:|---|
| `phase6_bl_scheduler_overhead_ms` | 64.757 ms | One Phase 6 Random+Chunk observation |
| `phase6_cbp_scheduler_overhead_ms` | 1165 ms | One Phase 6 Cost+FFD observation |
| `c1_r1_bl_scheduler_mean_ms` | 73.0879968667 ms | Mean of 30 BL throughput epochs |
| `c1_r1_bl_scheduler_sd_ms` | 11.727129931 ms | Epoch-level sample SD; epochs are nested within six runs |
| `c1_r1_bl_scheduler_min_ms` / `c1_r1_bl_scheduler_max_ms` | 61.714467–105.364501 ms | Observed epoch range |
| `c1_r1_bl_scheduler_epoch_pct` | 0.2799040681% | Ratio of mean scheduler time to mean BL epoch time |
| `c1_r1_gpu_scheduler_mean_ms` | 66.8442073333 ms | Mean of 30 GPU throughput epochs |
| `c1_r1_gpu_scheduler_sd_ms` | 7.28550183457 ms | Epoch-level sample SD; epochs are nested within six runs |
| `c1_r1_gpu_scheduler_min_ms` / `c1_r1_gpu_scheduler_max_ms` | 61.424144–95.476033 ms | Observed epoch range |
| `c1_r1_gpu_scheduler_epoch_pct` | 1.52941163216% | Ratio of mean scheduler time to mean GPU epoch time |

The BL and GPU C1-R1 configurations both use RandomSorter+ChunkPacker; the
different percentage is largely a denominator effect because GPU epochs are
shorter. The Phase 6 Cost+FFD number uses a different scheduling strategy and
protocol. These facts prohibit cross-phase splicing and do not establish a new
C2.6 claim.

## 5. CPU Fixture Results and Part 5 Blocker

The deterministic fixtures verify:

- repeated CostModel inputs produce byte-identical `float32` arrays of the same
  shape;
- all four sorter × packer combinations can be constructed manually;
- BatchProvider covers every fixture triple;
- three rank-strided partitions are pairwise disjoint and collectively cover
  all batches;
- Phase 9 Step 2 contains exactly the four registered labels;
- `RuntimePolicy` and `GPUExecution` implementations are absent.

The fixture also confirms `FFDPacker.pack() == ChunkPacker.pack()` for the frozen
cost-ordered input. The current FFDPacker scans bins from index zero and fills
each bin before moving to the next, which is sequential chunking. This is a
**Part 5 blocker**: Part 3 records it but does not modify the algorithm or
downgrade architecture-composition claims beyond their actual scope.

Accordingly, the safe scheduler statement is that two sorter classes and two
packer classes are manually composable. Part 3 does not claim that the packers
produce distinct layouts or that all four combinations were experimentally
validated.

## 6. Part 7 Correction Register

Part 7 must update Method/figure language to reflect these corrections:

1. BatchProvider is not a PyTorch DataLoader and does not yield
   `(pos_triples, neg_sampler)`.
2. Scheduler uses constructor composition and `pack_batches()`, not
   `configure()`/`schedule()`.
3. Scheduling runs on every `iterate()` call, rather than being permanently
   cached before training.
4. The driver/training loop, not BatchProvider, chooses the GPU backend.
5. `RuntimePolicy` and `GPUExecution` do not currently exist.
6. FeatureExtractor outputs `candidate_size`, `degree`, and `hub_flag`.
7. CostModel is an explicit formula producing a float32 table, not a trainable
   linear-regression layer.
8. Four sorter × packer combinations are manually constructible, but neither
   the factory nor the experiments validate all four.
9. Rank-strided partitioning must not be labeled “DDP-ready”.

The exact figure specification is frozen in
`docs/unified_runtime_architecture_freeze.md`. This Part does not edit the
Method, story freeze, historical runtime spec, or figure.

## 7. Risk Summary

| Contribution | Total | A | B | C | D | Verified |
|---|---:|---:|---:|---:|---:|---:|
| C2 Unified Runtime Framework | 6 | 4 | 1 | 0 | 1 | 66.7% |

The implementation architecture is now safe to describe using the frozen
wording. Paper sections relying on C2.2 remain Medium risk until the artifact
lineage is reconciled. Any section retaining the old C2.6 overhead claim is
Critical risk and must remove it.

## 8. Reproduction

```bash
python3 scripts/audit_c2_framework.py --self-test
python3 scripts/audit_c2_framework.py \
  --repo-root . \
  --output-dir output/results/evidence_audit_part3
python3 -m unittest tests/test_audit_c2_framework.py
```

The commands are CPU-only. The generated artifacts contain no timestamps and
must be byte-identical across repeated runs against the same repository state.
