# Phase X Part 3 — C2 Audit Implementation Plan

**Date**: 2026-08-03
**Design**: `docs/superpowers/specs/2026-08-03-phase-x-part3-c2-audit-design.md`

## Task 1: Establish deterministic audit tests

Create `tests/test_audit_c2_framework.py` before the implementation. Cover:

- deterministic CostModel bytes, dtype, and shape;
- four manual Scheduler compositions and full BatchProvider coverage;
- mutually exclusive rank partitions whose union covers all batches;
- exact Phase 9 driver labels;
- absence of implemented `RuntimePolicy` and `GPUExecution`;
- deterministic equivalence of FFDPacker and ChunkPacker on frozen fixtures;
- expected C2 grades and deterministic output serialization.

Run the test once and confirm it fails because the audit module is not yet
implemented.

## Task 2: Implement the CPU-only audit script

Create `scripts/audit_c2_framework.py` with:

- `--repo-root`, `--output-dir`, and `--self-test`;
- AST and SHA-256 source inspection;
- deterministic CPU fixtures;
- Phase 6 and C1-R1 metric recomputation;
- deterministic writers for source manifest, architecture mapping, metrics, and
  audit checks.

Run unit tests and the self-test until green.

## Task 3: Generate and verify audit artifacts

Run the script twice into separate temporary output directories and compare all
relative paths and SHA-256 hashes. Then generate the canonical
`output/results/evidence_audit_part3/` output.

Parse all JSON/CSV files and verify manifest paths/hashes.

## Task 4: Write architecture freeze and Part 3 report

Create:

- `docs/unified_runtime_architecture_freeze.md`
- `docs/evidence_audit_part3_c2_framework.md`

The report must give C2.1-R1 and C2.2–C2.6 exactly one grade, evidence chain,
paper-safe wording, and minimum remedy. It must record C2.2 artifact lineage,
C2.6 retraction, and the Part 5 packer blocker.

## Task 5: Update progress records

Update `PROGRESS.md` and `mukg-memory.json` without changing the frozen Part 1
registry.

## Task 6: Final verification and publication

Run:

- unit tests;
- audit `--self-test`;
- two-run byte comparison;
- JSON/CSV/path/hash validation;
- `python -m py_compile`;
- `git diff --check`;
- frozen-file and no-GPU-process checks.

Review the final diff, stage only scoped files, commit on `production`, and push
`production` to `origin`.
