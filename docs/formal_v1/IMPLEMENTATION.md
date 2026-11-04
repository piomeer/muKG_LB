# Formal v1 implementation ledger

Approved specification: user plan in this chat, 2026-11-04. Objective: recoverable archive and a minimal historical-TransE-based runtime, followed by diagnostic and budget-gated formal experiments.

## Binding protocol
Historical source fdfd880; TransE dim=250, L1, uniform initialization ±(7+2)/250, no normalization; logistic_adv margin=7, adv=2; Adam 0.0005; negatives=150; common batch preflight 5000→2500→1000. Official FB15K237 splits. CPU historical sampler versus GPU 75 head/75 tail with train-triple filtering; GPU tail-only is diagnostic. Full bidirectional average-tie filtered MRR and Hits@1/3/10; validation filter train+valid; test filter all splits. Diagnostic seed7 max100 epochs, no test. Formal quality seeds101–103, max2000, validation every10; min100, patience5, delta0.0001; checkpoint strict maximum independent of patience. Timing five paired seeds, five epochs after disposable warmup, throughput and trace separate. Quality target ratio≥0.95 with uncertainty, not a guaranteed outcome.

## Tasks
1. Archive snapshot, Git bundle, external CUDA evidence, hashes, restore and numeric replay.
2. Model/sampler/evaluator/data and historical parity tests.
3. Provenance-bound runner, preflight, diagnostics, budget freeze, formal subprocess matrix and analysis.
4. GPU verification, diagnostics, final review and handoff.

## Rulings
- Existing production worktree and uncommitted research files remain intact. Native managed worktree used.
- unittest is the test runner because pytest is absent; no dependency upgrade.
- GPU access is tested outside the sandbox; sandbox CUDA invisibility is not host driver failure.
- Formal long jobs require a diagnostic-derived compute budget with 30% reserve; GPU availability is currently unspecified.
- CPU reference retains historical final-attempt unfiltered fallback; its collision/duplicate behavior is disclosed, not silently fixed.
- GPU bounded rejection limit is 64 rounds; exhausted batches fail explicitly.
- Timing seeds are 201–205; diagnostic and quality seeds remain disjoint.

## Implementation verification and review

- Task 1 complete: 1590 snapshot files / 3,963,690,935 bytes, 15-ref Git bundle, external CUDA artifacts. Full snapshot restored into /tmp/mukg-formal-restore-20261104. C1-R1 and seed42 directional quality recomputed. Full snapshot plus sealed metadata verification passed.
- Task 2 complete: historical score/loss/gradient/Adam and CPU-sampler parity; full-triple GPU rejection; average-tie evaluation checked against exhaustive reference and real CUDA.
- Task 3 complete: isolated subprocess runner, provenance/environment bindings, no-overwrite results, checkpoint/early-stop separation, diagnostics, budget freeze, paired statistics and plots. Continuous core campaign is fail-fast and never retries a seed automatically.
- Fresh-context independent review found five important integrity issues: summary-only freeze, incomplete archive verification, missing frozen-protocol binding, incomplete result inventories, silently skipped directory symlinks. Seven regression tests were observed failing, then passing after fixes. No second review pass.
- Minor review item retained: standalone freeze output uses an exists check followed by atomic replace, so concurrent invocations targeting the identical file are unsupported. The supported campaign creates an exclusive root and has one controller. Do not concurrently freeze to the same path.
- Ruling: 2026-11-04 user confirmed continuous exclusive GPU access. Interpret the one-month window as 720 hours, deduct elapsed diagnostic/preflight time at freeze, and require the measured core projection plus30% reserve to fit. This authorizes continuous core execution; it does not imply the result will meet the research target.
- Ruling: leave codex/formal-v1 and its managed worktree in place, as specified; no merge/push or cleanup of the production workspace.
- Ruling: sensitivity and WN18RR remain post-core decisions; no additional model, dataset, or outcome-driven seed expansion is launched.
- Final validation: 159 unittest tests passed on the host with CUDA available; new CUDA parity test executed (not skipped). Archive verify returned an empty error list including all sealed external files and Git bundle. git diff --cached --check passed.
