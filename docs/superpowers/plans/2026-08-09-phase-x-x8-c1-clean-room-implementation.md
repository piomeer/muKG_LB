# Phase X8-C1 — C1-R1 Clean-room Implementation Plan

## Global Constraints

- Preserve the C1-R1-v1.1 training, sampler, scheduler, data split, seeds, and
  estimands. Do not modify runtime or training code.
- Use the existing C1-R1 runner with SHA-256
  `2556df6aa6e50d20ae2c188fe987a7694dff1743473aaf0dd62a4e96615710ab`.
- The execution capsule must exclude historical result directories. Independent
  analysis must complete and be sealed before comparison code may read the
  original C1-R1 result root.
- E1 and E2 form one joint primary family. Report per-estimand 95% intervals and
  require Bonferroni 97.5% simultaneous lower bounds above one.
- Numerical-fidelity clean/original ratios are inclusive: E1 `[0.90, 1.10]`,
  E2 `[0.75, 1.25]`, E3 `[0.90, 1.10]`.
- Do not pool original and clean-room runs. X8 is an internal same-host artifact
  verification, not an external or cross-hardware replication.
- No network access. No paper-body or Part 1 changes. Raw runtime timestamps are
  lineage data; derived outputs must be byte deterministic for a fixed snapshot.
- Use standard-library `unittest` and test-driven development for new behavior.

## Task 1: Freeze the tracked protocol and statistical contract

Create the X8 protocol document and a machine-readable contract. Define the
source/input hashes, allowlisted capsule contents, environment requirements,
execution matrix, estimands, filters, retry policy, numerical thresholds,
verdict states, Material Passport semantics, and expected artifact schemas.

Add initial unit tests that load the contract and assert behavior through the
future audit/executor public interfaces rather than grepping source text. Tests
must initially fail because the interfaces do not exist.

## Task 2: Implement the isolated executor and raw sealing

Add `scripts/run_x8_c1_r1_clean_room.py` with commands `prepare`, `status`,
`preflight`, `run`, `remediate`, and `seal`.

The executor must build an allowlist source capsule, clone the active Conda
environment locally without network access, validate input/source hashes, invoke
the unchanged C1-R1 runner subcommands in the frozen serial order, persist a
resumable manifest, enforce paired remediation rules, and seal raw artifacts.
It must fail closed on environment/protocol drift, GPU contention, telemetry
failure, invalid retry requests, or attempts to overwrite an existing root.

Complete the RED/GREEN cycle for executor behavior using filesystem fixtures and
injected subprocess transports; tests must not require a GPU or clone a real
Conda environment.

## Task 3: Implement independent analysis and delayed comparison

Add `scripts/audit_x8_c1_r1_clean_room.py` with `--stage independent`,
`--stage compare`, and `--self-test`.

Independent mode must validate the raw seal and artifact schemas, select only
predeclared valid attempts, recompute E1/E2/E3 from raw integer-nanosecond data,
produce 95% and 97.5% intervals, seed-level effects, direction consistency, and
leave-one-seed-out diagnostics. It must reject any original-result argument.

Compare mode may run only after the independent outputs have their own seal. It
must compare against the frozen original estimates, apply the three numerical
tolerances, run the 11-item statistical fallacy scan, and emit exactly one of:
`VERIFIED`, `SUPPORTED_WITH_NUMERICAL_DRIFT`, `NOT_REPRODUCED`, `INCOMPLETE`, or
`BLOCKED_ENVIRONMENT`.

Complete TDD with hand-derived fixtures for all statistics, filters, seal
tampering, blind-boundary enforcement, and verdict branches.

## Task 4: Integrate, verify, and execute when GPU is available

Run all X8 unit/self-tests, compile checks, JSON/CSV validation, source/hash
checks, deterministic-output checks, quarantine checks, and `git diff --check`.
Document the existing clean-worktree baseline failures caused by untracked
historical evidence files.

Commit and push the implementation commit, then prepare the clean-room capsule
from that exact clean commit. If GPU preflight succeeds, run the complete 1 + 24
+ 6 job matrix, seal raw data, perform blind independent analysis, then compare
and write `docs/phase_x_x8_c1_r1_clean_room_report.md`. If GPU preflight is
blocked, produce a deterministic blocked status without fabricating results.

Update `PROGRESS.md` and project memory without overwriting unrelated concurrent
work. Commit and push the resulting implementation/result state to `production`
only through the approved branch-integration workflow.
