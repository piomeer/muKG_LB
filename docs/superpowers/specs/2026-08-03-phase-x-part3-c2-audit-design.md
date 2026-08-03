# Phase X Part 3 — C2 Unified Runtime Framework Evidence Audit Design

**Date**: 2026-08-03
**Status**: Approved (A方案)
**Scope**: C2.1–C2.6 read-only evidence audit

## 1. Objective

Freeze an implementation-faithful architecture and audit the six C2 claims using
source inspection, existing artifacts, deterministic CPU fixtures, and independent
recomputation. This part does not run training or GPU experiments and does not
change runtime implementation, the paper Method, the story freeze, the historical
runtime specification, or figures.

## 2. Canonical Architecture

The implemented framework is a **two-stage architecture with five implemented
roles**, not a five-layer online stack.

### Offline control plane

1. **FeatureExtractor**: derives `candidate_size`, `degree`, and `hub_flag`.
2. **CostModel**: `build_cost_table()` maps static features and constants to an
   `np.float32` array.
3. **Cost Table**: the materialized array consumed by scheduling-time lookups.

### Online path

4. **Scheduler**: constructor composition of a sorter and packer, exposed through
   `pack_batches(triples_list, cost_table, batch_size)`.
5. **BatchProvider**: calls the Scheduler on every `iterate()` invocation, applies
   rank-strided batch slicing, and yields lists of positive triples.

The external training loop consumes these batches and explicitly selects either
the CPU negative-sampling path or `GPUNegativeSampler`. Backend selection is
configuration-driven composition; it is not automatic injection by BatchProvider
and is not claimed to be a transparent drop-in replacement.

`RuntimePolicy` and `GPUExecution` are design-only future extensions. No current
class or module implements them. DDP support is limited to the rank-strided
partition API; the framework is not described as DDP-ready.

## 3. Claim Decisions

- **C2.1-R1 — A**: the canonical two-stage, five-role implementation architecture
  replaces the held, inconsistent layer description.
- **C2.2 — B**: one driver enumerates BL, CBP, GPU, and CBP+GPU, but its artifact
  lineage is inconsistent because per-config output is written to `.md` while the
  final aggregation reads `.csv`.
- **C2.3 — A**: Scheduler and BatchProvider are shared; sampler selection occurs
  explicitly in the training loop.
- **C2.4 — A**: deterministic table construction and array lookup are verified as
  implementation properties only. Predictive validity belongs to Part 4.
- **C2.5 — A**: iterator behavior and rank-strided batch partitioning are verified;
  no PyTorch DataLoader or DDP-readiness claim is made.
- **C2.6 — D / RETRACTED**: the historical “~0.5ms per epoch” claim remains
  invalid. Recorded overheads are protocol-specific descriptions, not a
  replacement C2.6 claim.

## 4. Audit Mechanism

`scripts/audit_c2_framework.py` will:

- accept `--repo-root`, `--output-dir`, and `--self-test`;
- use AST checks for interfaces, configuration enumeration, backend selection,
  output/read suffixes, and absent design-only classes;
- hash source evidence with SHA-256;
- run CPU-only deterministic fixtures for CostModel, Scheduler, BatchProvider,
  rank slicing, and packer equivalence;
- recompute existing Phase 6 and C1-R1 scheduling overhead metrics;
- emit deterministic JSON/CSV with no timestamps.

The output directory is `output/results/evidence_audit_part3/` and contains:

- `source_manifest.json`
- `architecture_mapping.csv`
- `recomputed_metrics.csv`
- `audit_checks.json`

## 5. Non-Claims and Deferred Corrections

Part 3 does not claim:

- BatchProvider is a PyTorch DataLoader;
- BatchProvider yields `(pos_triples, neg_sampler)`;
- Scheduler exposes `configure()` or `schedule()`;
- scheduling is computed once before training and permanently cached;
- GPU backend selection is internal to BatchProvider;
- `RuntimePolicy` or `GPUExecution` exists;
- CostModel is a trainable linear-regression layer;
- the factory or experiments validate all four sorter × packer combinations;
- rank slicing alone makes the framework DDP-ready;
- a universal scheduler-overhead percentage.

Part 7 must correct the paper-facing architecture text and figure using the
canonical mapping. Part 5 must investigate the implementation fact that
`FFDPacker.pack()` equals `ChunkPacker.pack()` on the frozen ordered fixtures.

## 6. Acceptance Criteria

- Every C2 claim has exactly one grade, evidence chain, paper-safe wording, and
  remediation condition.
- Every report number maps to one derived CSV row and its original source.
- Two consecutive audit runs produce byte-identical output trees.
- JSON/CSV parsing, source paths, and source hashes validate.
- CPU fixtures pass, including the intentional FFD/Chunk equivalence blocker.
- `python -m py_compile`, `--self-test`, and `git diff --check` pass.
- No GPU process or training experiment is started.
- Part 1 and all explicitly frozen runtime/paper files remain unchanged.
