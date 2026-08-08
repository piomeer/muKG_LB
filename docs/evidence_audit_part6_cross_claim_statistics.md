# Phase X X6 — Cross-Claim Statistical Integrity Audit

## Material Passport

Verification Status: **ANALYZED** (no independent clean-room rerun)

Overall X6a status: **BLOCKED_X5_5_INPUT**.

X6a is fail-closed until X5.5 provides finalized `contribution_triage.csv` and `gap_closing_decision.json`; no contribution promotion or novelty decision is made here. The available C1-R1 paired artifacts were re-computed read-only.

## C1 joint statistical layer

E1 (end-to-end epoch speedup) and E2 (full-batch negative-sampling dispersion compression) remain distinct estimands sharing six seed-level paired jobs. The frozen 95% intervals are retained. A Bonferroni 97.5% simultaneous interval gives E1 approximately 5.9276–6.1004× and E2 approximately 69.8452–110.5642×; both lower bounds exceed 1, so the joint statistical gate is `PASS_WITH_SCOPE_LIMITS`. This does not establish quality equivalence, variance of training quality, cross-model generality, or hardware portability.

The six paired effects are directionally consistent (6/6 > 1 for each); leave-one-seed-out geometric means and the log-effect correlation are diagnostics, not independent replication. BL-first/GPU-first strata have n=3 and are descriptive only. The seed-45 thermal attempt is retained in lineage and excluded according to the frozen protocol.

## Cross-claim dependence and eligibility

E1/E2 share seeds, split, environment, and code lineage despite distinct passes. E3 is derived from the E2 trace; C2 scheduler overhead reuses throughput epochs; C3.2/C4.1 share Phase 6 attribution rows; C3.6/C4.7 share a cost table; C4.4–C4.6 share a single-process quality protocol. These edges are not independent corroboration. Implementation facts are excluded from statistical multiplicity families.

Current paper eligibility is an overlay only: C1.2-R1 and C1.3-R1 are eligible primary claims with scope limits; C1.7-R1 is secondary descriptive; passed C2 and C3.3/C3.6 entries are implementation facts. Predictive C3, composite CBP/FFD, quality equivalence, VRAM, DDP, and generalization remain not eligible.

## X6.5 contract

Each future gap-closing branch must declare one primary promotion estimand before execution, independent units, filters, effect direction, missing-job rules, and a protocol SHA-256. Secondary outcomes cannot rescue a failed primary.

## Closure

X6 does not run GPU, training, network, runtime changes, paper-body edits, or Part 1 edits. X6b must be run after X5.5 and either an executed, hash-matching X6.5 artifact or a formal waiver.
