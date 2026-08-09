# Canonical Unified Runtime Architecture Freeze

**Version**: 1.0
**Date**: 2026-08-03
**Status**: Canonical for Evidence Audit Part 3 and Part 7 corrections
**Definition**: Two-stage architecture with five implemented roles

## 1. Frozen Definition

The current implementation is described as a **two-stage architecture with five
implemented roles**. “Role” is intentional: the Cost Table is a materialized
array rather than a class, and the roles do not form five permanently active
runtime layers.

| Stage | Order | Implemented role | Actual interface | Output |
|---|---:|---|---|---|
| Offline control plane | 1 | FeatureExtractor | `FeatureExtractor(triples_list, num_entities).build(force_recompute=False)` | `candidate_size`, `degree`, `hub_flag` |
| Offline control plane | 2 | CostModel | `build_cost_table(features, neg_num=150, max_try=10, b3_const=51.8)` | `np.float32` cost array |
| Offline control plane | 3 | Cost Table | `cost_table[entity_id]` | Precomputed scalar cost |
| Online per epoch | 4 | Scheduler | `Scheduler(sorter, packer).pack_batches(triples_list, cost_table, batch_size)` | List of triple batches |
| Online per epoch | 5 | BatchProvider | `iterate(triples_list)`; `set_rank(rank, world_size)` | Iterator of triple-list batches |

The external training loop is the consumer/orchestrator. It receives a positive
triple batch from BatchProvider and explicitly chooses the CPU negative-sampling
function or `GPUNegativeSampler.generate()`.

## 2. Data and Control Flow

```text
Offline control plane
training triples
  → FeatureExtractor
  → {candidate_size, degree, hub_flag}
  → build_cost_table(explicit formula and constants)
  → np.float32 Cost Table

Online path, repeated for each BatchProvider.iterate() call
epoch triples + Cost Table
  → Scheduler(sorter, packer).pack_batches(...)
  → BatchProvider rank-strided selection
  → yield List[(head, relation, tail)]
  → external training loop
       ├─ CPU negative-sampling function
       └─ GPUNegativeSampler.generate(batch)
```

Scheduling is triggered inside every `BatchProvider.iterate()` invocation. The
implementation does not permanently cache a layout computed before training.
The BatchProvider docstring sentence suggesting reuse by “subsequent iterations”
does not match the method body and is not part of the canonical definition.

## 3. Implementation Boundaries

### FeatureExtractor

`FeatureExtractor.build()` outputs three arrays: `candidate_size`, `degree`, and
`hub_flag`. It may load or save the global feature cache, but its implemented
output is not a learned representation.

### CostModel and Cost Table

CostModel is the function `build_cost_table()`. It applies an explicit formula
with configured constants and returns a `float32` NumPy array. It is not a
trainable linear-regression layer. Part 3 verifies deterministic construction
and array access only; predictive validity and variable legitimacy are Part 4.

### Scheduler

Scheduler is configured through constructor composition:
`Scheduler(sorter, packer)`. Its batching entry point is `pack_batches()`.
There is no `configure()` or `schedule()` public API.

Two sorter and two packer classes can be composed manually. The factory exposes
only Random+Chunk and Cost+FFD aliases, and Phase 9 experiments benchmark those
paired strategies rather than all four sorter × packer combinations.

### BatchProvider

BatchProvider is an iterator adapter, not a PyTorch DataLoader. It yields a list
of positive triples, not `(pos_triples, neg_sampler)`. It neither constructs nor
injects the sampling backend.

`set_rank(rank, world_size)` causes `all_batches[rank::world_size]` slicing. This
is a rank-strided partition API only; the implementation does not establish
distributed initialization, synchronization, sharding policy, or end-to-end DDP
readiness.

### Sampling backend

The experiment driver/training loop owns backend selection through an explicit
`use_gpu` branch. This is configuration-driven composition. It must not be
described as transparent, automatic, or semantically drop-in.

## 4. Future Extensions

`RuntimePolicy` and `GPUExecution` are design-only terms retained as possible
future extensions. No corresponding current module or class exists, so neither
is included among the five implemented roles.

## 5. Frozen Non-Claims

The architecture freeze does not claim:

- five concurrently active runtime layers;
- a PyTorch DataLoader replacement;
- automatic sampler injection by BatchProvider;
- permanent pre-training schedule caching;
- a `RuntimePolicy` or `GPUExecution` implementation;
- DDP readiness;
- a learned CostModel;
- factory or experimental validation of all four scheduling combinations;
- negligible, zero, or universal scheduler overhead;
- semantic equivalence between the CPU and GPU sampling paths.

## 6. Part 7 Paper and Figure Specification

Part 7 must update the paper-facing architecture, but Part 3 does not perform
those edits. The corrected figure must:

1. use two stage containers: “Offline control plane” and “Online per epoch”;
2. show FeatureExtractor → CostModel → Cost Table in the offline container;
3. show Cost Table and epoch triples entering Scheduler, followed by
   BatchProvider in the online container;
4. place the training loop outside the five-role framework;
5. branch from the training loop to CPU and GPU sampling backends;
6. label Scheduler with constructor composition plus `pack_batches()`;
7. label BatchProvider as an iterator of positive-triple lists;
8. annotate rank-strided slicing as an API, not DDP readiness;
9. render RuntimePolicy and GPUExecution, if shown at all, in a visually separate
   “Future Extensions” region;
10. avoid DataLoader, automatic injection, permanent cache, learned-model, or
    transparent-switching symbols.

## 7. Machine-Readable Ground Truth

The role mapping and audit facts are serialized in:

- `output/results/evidence_audit_part3/architecture_mapping.csv`
- `output/results/evidence_audit_part3/audit_checks.json`

These artifacts are generated by `scripts/audit_c2_framework.py` from source AST,
source hashes, and deterministic CPU fixtures.
