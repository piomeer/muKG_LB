# Task 4 report — X8 C1-R1 clean-room integration

## Result

Status: **BLOCKED_ENVIRONMENT** before preflight. No experiment or analysis was
run and no E1/E2/E3 value or VERIFIED claim was produced.

## Commits

- Baseline implementation supplied for this task: `f6eb568c6f73b4c50e99a49f2726257dc91d9331`.
- Executor compatibility repair: `ed403bd4ce059a0571f18fd680930f5ef89c4cd1` (`fix: support offline Conda package manifests`).
- Failed-prepare lineage repair: `d0fbe4ed0737f27ccca0674fdcc7af061453efda` (`fix: retain failed prepare lineage`).

The repair was TDD: a regression test for Conda versions that reject
`conda list --offline` was added, observed failing, then passed after the two
read-only package-list calls stopped passing that unsupported flag. The executor
continues to set `CONDA_OFFLINE=true`, `PIP_NO_INDEX=1`, and empty proxy values.

## Verification

- Full X8 suite after the lineage repair: 41 tests passed.
- Audit self-test: PASS.
- `py_compile` of X8 executor/audit and frozen runner: PASS.
- Contract JSON, all 11 source/input SHA-256 entries, and CSV schemas: valid.
- `git diff --check`: PASS.
- X0.5 quarantine checker: baseline FAIL in this isolated clean worktree because
  `output/results/phase9_step4_5` is absent. The shared checkout has untracked
  historical/in-progress evidence; none was copied or used.
- The appended final `mukg-memory.json` record parses as JSON. Existing unrelated
  historical lines 75 and 139 are malformed JSONL and were deliberately not
  rewritten by this task.

## Executor state and environment evidence

The authoritative third executor attempt began at
`2026-08-09 16:05:56.248923981 +0900` from captured commit `d0fbe4e`, with
active prefix `/home/hma/miniconda3` and root
`output/results/x8_c1_r1_clean_room_v1_attempt3`. Before the fallible GPU query,
`raw/prepare_attempt.json` captured Git HEAD, active-prefix and clone probes,
and `argv`/exit/stdout/stderr for Git, active probe, Conda clone, Conda list,
clone probe, and GPU identity. It then stopped at required GPU identity capture:

```text
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
exit 9: could not communicate with the NVIDIA driver
```

The cloned runtime is PyTorch `2.7.1+cu118`, CUDA runtime `11.8`, and
`torch.cuda.is_available()==False`. Since prepare did not accept the GPU
environment, it did not write `execution_manifest.json`. Both `status` and
`preflight` then refused with `clean-room root is not prepared` (exit 1), before
any GPU dispatch. `run`, `remediate`, `seal`, independent analysis, and compare
were intentionally not invoked.

The artifact-backed deterministic closure is
`output/results/x8_c1_r1_clean_room_v1_attempt3/blocked_environment_closure.json`;
it regenerates byte-identically from the raw capture. The compact canonical
index is `output/results/x8_c1_r1_clean_room_v1/blocked_environment_closure.json`.
The frozen contract SHA-256 is
`32396312a947ef24e937c873d70f28b725c9e82aa5102d14bd1f6c449033b46a` and its
capsule manifest SHA-256 is
`5bc4830f23ff61b8f8598b08e38b5be3830331543919319a39508059b7ac108f`.

## Concern and resume condition

The authoritative third root is a deliberately unprepared, blocked capsule and
contains the local clone; it must not be overwritten. The first partial
preparation (`output/results/x8_c1_r1_clean_room_v1_prepare_defect_attempt1`)
and the former canonical root (`output/results/x8_c1_r1_clean_room_v1`) lack a
raw `prepare_attempt.json`; both are retained as explicitly incomplete
historical lineage rather than reconstructed evidence. The implementation was
committed locally but not pushed before attempt3 because no-network was
enforced; this is a procedural deviation, not retroactive compliance.

After the NVIDIA driver returns, confirm `nvidia-smi` identity and
`torch.cuda.is_available()==True` in the active prefix, create a new root, and
rerun `prepare`, `status`, and `preflight`. Only a passing preflight authorizes
the frozen 1 + 24 + 6 matrix, sealing, blind independent analysis, and delayed
comparison.
