# Phase X8 — C1-R1 clean-room execution report

## Material Passport

Verification Status: **BLOCKED_ENVIRONMENT**. This is an internal same-host
clean-room execution attempt, not an external replication. No formal Material
Passport exists: the executor failed closed during `prepare`, before it could
write `execution_manifest.json`, `environment_manifest.json`, or a raw-artifact
seal. This report and `blocked_environment_closure.json` preserve the available
capsule, environment, and command lineage; they are not a substitute for the
protocol Material Passport.

| Field | Value |
| --- | --- |
| Contract | `X8-C1-R1-clean-room-v1` |
| Attempt3 contract SHA-256 | `32396312a947ef24e937c873d70f28b725c9e82aa5102d14bd1f6c449033b46a` |
| Authoritative attempt | `prepare-x8_c1_r1_clean_room_v1_attempt3-1786259156248923157` |
| Attempt start (JST) | `2026-08-09 16:05:56.248923981 +0900` |
| Executor commit (captured) | `d0fbe4ed0737f27ccca0674fdcc7af061453efda` |
| Capsule manifest SHA-256 | `5bc4830f23ff61b8f8598b08e38b5be3830331543919319a39508059b7ac108f` |
| Raw prepare capture SHA-256 | `6536b020e277dbf31ead888bed54951f0a278a8199bec650410f03e4875857fe` |
| Derived closure SHA-256 | `af92487f80aa73cb0c1b85644c0b4574a99fa533e898b0b313f5b7be460bdc46` |
| Requested clean-room root | `output/results/x8_c1_r1_clean_room_v1_attempt3` |
| Active Conda prefix | `/home/hma/miniconda3` |
| Attempt3 network controls | Package-manager offline variables and proxy-environment changes only; no kernel network namespace was present |

The 11-file capsule contains only the contract allowlist, including the frozen
runner whose SHA-256 remains
`2556df6aa6e50d20ae2c188fe987a7694dff1743473aaf0dd62a4e96615710ab`.

## Exact state and blocker

The authoritative third preparation constructed the allowlisted capsule and
locally cloned the specified Conda environment. Before its fallible GPU
identity query, `raw/prepare_attempt.json` retained the captured Git HEAD,
active-prefix probe, clone probe, and complete `argv`/exit/stdout/stderr records
for six external commands: Git HEAD, active-prefix probe, Conda clone, Conda
list, clone probe, and GPU identity. It then failed closed while collecting the
contract-required GPU identity:

```text
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
exit code: 9
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

The clone runtime probe reports PyTorch `2.7.1+cu118`, CUDA runtime `11.8`, and
`cuda_available: false`. The original active prefix reports the same CUDA
unavailability. Accordingly, `status` and `preflight` each exited 1 with
`clean-room root is not prepared`; neither could dispatch a GPU command because
the executor had not accepted a prepared environment.

`run`, `remediate`, `seal`, independent analysis, and comparison were not
invoked. No primary or diagnostic jobs ran. No raw data were sealed, no E1/E2/E3
estimands were computed, and no numerical-fidelity or **VERIFIED** claim is
made.

The artifact-backed closure is
`output/results/x8_c1_r1_clean_room_v1_attempt3/blocked_environment_closure.json`.
Regenerating it from `raw/prepare_attempt.json` produced byte-identical output.
The compact canonical index at
`output/results/x8_c1_r1_clean_room_v1/blocked_environment_closure.json` points
to these hashes. Both earlier roots are explicitly marked
`HISTORICAL_LINEAGE_INCOMPLETE`: neither contains a retained raw
`prepare_attempt.json`, so their command/environment capture is not
reconstructed or asserted retroactively.

## Executor compatibility repair

The first preparation attempt exposed a separate Conda compatibility defect:
Conda `26.1.1` rejects `conda list --offline`, although it accepts the offline
clone operation. The initial partial attempt is preserved at
`output/results/x8_c1_r1_clean_room_v1_prepare_defect_attempt1` and was not
used as an execution capsule. Root-cause evidence was the command's exit 2 and
its `conda list --help` output, which has no `--offline` option.

A regression test was written and observed failing before the two package-list
calls were changed to rely on the already-enforced offline environment rather
than the unsupported subcommand flag. Commit
`ed403bd4ce059a0571f18fd680930f5ef89c4cd1` contains that repair. Commit
`d0fbe4ed0737f27ccca0674fdcc7af061453efda` adds retained failed-prepare
lineage and tests both package-list call sites. The full X8 suite then passed:
41 tests, executor/audit compilation, and `git diff --check`.

## Procedural deviation

The implementation commits were made locally but were not pushed before the
blocked preparation. Network use was prohibited, but attempt3 relied on
application environment controls rather than an OS network namespace; those
variables are not firewall evidence. This is not retroactive compliance with
the plan's pre-prepare push instruction. Final branch integration and any push
remain separate work.

## Non-GPU verification and baseline limitation

The following checks passed before the live preparation attempt:

- `python -m unittest -v tests.test_x8_c1_clean_room_contract tests.test_x8_c1_clean_room_executor tests.test_x8_c1_clean_room_audit` — 39 tests after the repair.
- `python scripts/audit_x8_c1_r1_clean_room.py --self-test` — PASS.
- `python -m py_compile scripts/run_x8_c1_r1_clean_room.py scripts/audit_x8_c1_r1_clean_room.py src/py/experiments/c1_r1_combined_rerun.py`.
- JSON parsing, CSV-schema checks, and SHA-256 validation of all 11 frozen source/input entries.
- `git diff --check`.

The X0.5 quarantine checker was deliberately executed using temporary output.
It fails in this isolated clean worktree because
`output/results/phase9_step4_5` is absent. The shared checkout instead contains
untracked historical evidence and unrelated in-progress files; neither was
copied, modified, or used by X8. This is a baseline clean-worktree limitation,
not an X8 result.

`mukg-memory.json` received one append-only X8 record. Its new final line
parses as JSON; the pre-existing file has unrelated malformed historical JSONL
lines (75 and 139), so this task did not rewrite or normalize that shared
memory history.

## Re-review correction: prepare-failure classification

Commit `cd924935b014d5bb1c1318864b30ce422cdc0c57` corrects an executor defect
found after attempt3. Previously, every prepare exception was labelled
`BLOCKED_ENVIRONMENT`, and an early failure could be masked by a closure builder
that required unavailable Git/clone-probe evidence. The executor now emits
`BLOCKED_ENVIRONMENT` only for sufficiently evidenced GPU/runtime environment
failures at the GPU-identity stage. Earlier Git, capsule/hash, Conda/clone,
probe-parse, and integrity failures retain their available
`raw/prepare_attempt.json` command lineage as `PREPARE_FAILED`, emit no
scientific blocked-environment closure, and re-raise the original failure.

Two regression tests force an early Git failure and an offline Conda clone
failure; both preserve the original error and partial raw lineage without a
closure. The existing GPU-identity failure regression remains blocked with an
artifact-backed closure. The suite now has 43 passing X8 tests. No new live
prepare was run and attempt3 artifacts were not modified by this correction.

## Final-review hardening after attempt3

Attempt3 predates the final-review hardening. The current tracked contract has
SHA-256 `b76f14a5439a11ccfe1a3bee1ab574f739a7291aeaf9a0362d679d8e0a12ff99`;
the attempt3 closure correctly retains its earlier contract hash instead of
being rewritten. The post-attempt implementation now:

- freezes six-of-six direction consistency as part of the E1/E2 primary gate;
- keeps exact original estimates only in the separately tracked, hash-bound
  comparison reference and opens it only after the independent seal validates;
- gives every subprocess an explicit non-inherited environment and wraps every
  command in `unshare --user --map-root-user --net --`, with a fail-closed
  namespace probe during `prepare`;
- validates the actual capsule file set/hashes, cloned-environment identity, and
  every execution-manifest binding before accepting a raw seal; and
- rejects `preflight` immediately after Material Passport creation without
  changing the execution manifest or raw seal.

No new live `prepare`, preflight, GPU job, seal, independent analysis,
comparison, E1/E2/E3 estimate, or verdict was produced during this hardening.
The tests used injected transports and synthetic sealed fixtures only.

## Resume only after environment repair

1. Restore NVIDIA-driver communication and confirm the expected RTX 3070 GPU
   identity through `nvidia-smi`.
2. Confirm `torch.cuda.is_available()` is true in `/home/hma/miniconda3`.
3. Confirm the host permits the frozen user/network namespace command, then use
   a new root (the executor must not overwrite a blocked root) and run `prepare`.
   The namespace probe must pass before Git, Conda, runtime, or GPU preparation
   proceeds.
4. Run `status`, then `preflight`.
5. Only after a passing preflight, execute the frozen 1 + 24 + 6 matrix, seal
   raw data, run blind independent analysis, and then permit comparison.
