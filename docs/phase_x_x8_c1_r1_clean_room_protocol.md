# Phase X8 — C1-R1 clean-room protocol

## Purpose and boundary

X8 is an internal, same-host clean-room artifact verification of the frozen
`C1-R1-v1.1` runtime protocol. It is not an external or cross-hardware
replication, and it does not pool original and clean-room observations. The
unchanged runner is
`src/py/experiments/c1_r1_combined_rerun.py` at SHA-256
`2556df6aa6e50d20ae2c188fe987a7694dff1743473aaf0dd62a4e96615710ab`.

The machine-readable authority is
`output/results/evidence_audit_x8_c1_r1/clean_room_contract.json`. Its
allowlist is the complete execution capsule; its denylist explicitly excludes
the original C1-R1 result root and all X8 outputs from capsule construction.
No network access is permitted. The active Conda environment is cloned locally,
and source, input, environment, and raw-artifact hashes must validate before a
stage is accepted.

## Frozen execution

The preserved C1-R1 parameters are batch size 5000, 150 negatives per positive,
five epochs per measured job, and seeds 42 through 47. BL/GPU order alternates
by seed exactly as recorded in the contract. The initial matrix is one
preflight, 24 primary jobs (two passes × two configurations × six seeds), and
six GPU compute-only diagnostics. Preflight failure stops the run; batch size
is never reduced. Reserved VRAM must remain strictly below 90% of physical
VRAM, and GPU contention or telemetry failure invalidates the affected attempt.

Each primary job retains the runner’s raw `status.json`, `per_epoch.csv`,
`per_step.csv` where applicable, and GPU telemetry. The contract defines their
minimum schemas. Runtime timestamps and command captures are lineage evidence;
they must not make derived outputs non-deterministic for a fixed sealed
snapshot.

## Retry and sealing

Only an explicitly invalid infrastructure attempt may be remediated. One retry
is available for the complete BL/GPU pair for the same pass and seed. The
invalid attempt remains in the manifest and is excluded from analysis. A
single-config retry, changed parameters, overwrite, or a third attempt is
forbidden.

A Material Passport binds the contract hash, capsule manifest, cloned
environment, stage, and raw/derived artifact seal. Every sealed artifact has a
path, byte count, and SHA-256. Missing or mismatched lineage fails closed.
Independent analysis must create and seal its own passport before any comparison
code may read the original C1-R1 result root.

## Statistical contract

E1 is the seed-paired BL/GPU ratio of five-epoch mean throughput time. E2 is
the seed-paired BL/GPU ratio of the mean of five within-epoch population SDs of
full-batch negative-sampling time. E3 is the six-run arithmetic summary of GPU
full-batch negative-sampling time. E1 and E2 are one joint primary family:
report their per-estimand 95% intervals and require Bonferroni 97.5%
simultaneous lower bounds strictly above one. E2 and E3 use only rows with
`is_partial == False` and `batch_size_actual == 5000`; E2 uses `ddof=0`.

The inclusive clean/original numerical-fidelity ratio ranges are E1
`[0.90, 1.10]`, E2 `[0.75, 1.25]`, and E3 `[0.90, 1.10]`. Comparison may emit
only `VERIFIED`, `SUPPORTED_WITH_NUMERICAL_DRIFT`, `NOT_REPRODUCED`,
`INCOMPLETE`, or `BLOCKED_ENVIRONMENT`.
