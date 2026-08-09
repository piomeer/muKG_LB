# Phase X X0.5 — Legacy Narrative Quarantine Register

## 1. Safe Writing Sources

Numerical or architectural wording may be drafted directly only from the
listed canonical sources. Any other source requires an explicit lookup in
the Claim Propagation Matrix before reuse.

- docs/phase_x_x0_research_freeze.md — X0 authority for research questions,
  scope, contribution hierarchy, and frozen estimands.
- docs/evidence_audit_part2_c1_gpu_runtime.md and
  output/results/evidence_audit_part2/ — Part 2 C1 audit and its derived
  artifacts.
- docs/evidence_audit_part3_c2_framework.md and
  output/results/evidence_audit_part3/ — Part 3 C2 audit and its derived
  artifacts.
- docs/evidence_audit_part4_c3_cost_model.md and
  output/results/evidence_audit_part4/ — Part 4 C3 audit and its derived
  artifacts.
- docs/evidence_audit_part5_c4_cbp.md and
  output/results/evidence_audit_part5/ — Part 5 C4 audit and its derived
  artifacts.
- docs/phase_x_x5_5_contribution_triage.md and
  output/results/evidence_audit_x5_5/ — final contribution triage and C3/C4
  waiver decision.
- docs/evidence_audit_part6_cross_claim_statistics.md and
  output/results/evidence_audit_part6/ — cross-Claim statistical overlay and
  final paper-eligibility propagation.
- docs/unified_runtime_architecture_freeze.md — canonical
  figure/interface boundary for the implemented framework.
- output/results/c1_r1_combined_rerun/ — C1-R1 source for unrounded
  performance observations.

## 2. Purpose, Boundary, and Authority Order

This register is the writing-facing evidence firewall for legacy narrative.
It preserves historical material for traceability while preventing it from
being reused as manuscript authority. Authority order is **X0 → completed
claim audit → Part 1 lineage only → legacy narrative**. Part 1 is a frozen
inventory and lineage index, not a replacement evidence source.

## 3. Document Status Register

| Status | Path | Writing role |
| --- | --- | --- |
| CANONICAL | docs/phase_x_x0_research_freeze.md | Scope, hierarchy, and estimands |
| CANONICAL | docs/evidence_audit_part2_c1_gpu_runtime.md | C1 audited wording |
| CANONICAL | docs/evidence_audit_part3_c2_framework.md | C2 audited wording |
| CANONICAL | docs/unified_runtime_architecture_freeze.md | Implemented architecture and figure boundary |
| FROZEN-INVENTORY | docs/evidence_audit_part1_claim_inventory.md | Claim lineage only |
| LEGACY-NON-AUTHORITATIVE | paper/draft/method.md | Historical draft; do not source claims |
| LEGACY-NON-AUTHORITATIVE | docs/paper_outline.md | Historical outline; do not source claims |
| LEGACY-NON-AUTHORITATIVE | docs/paper_story_freeze.md | Historical narrative; do not source claims |
| LEGACY-NON-AUTHORITATIVE | docs/runtime_framework_spec.md | Historical design narrative; do not source claims |
| LEGACY-NON-AUTHORITATIVE | docs/phase8_architecture_freeze.md | Historical interface plan; do not source claims |
| LEGACY-NON-AUTHORITATIVE | docs/baseline_freeze.md | Historical baseline record; do not source claims |
| LEGACY-NON-AUTHORITATIVE | docs/validation_plan.md | Historical validation plan; do not source claims |
| LEGACY-NON-AUTHORITATIVE | docs/evidence_matrix.md | Historical evidence matrix; do not source claims |

## 4. Data Source Status Register

Use these columns so the supersession lineage is explicit:

| Path | Status | Affected Claim IDs | Allowed Use | Superseded/Controlled By |
| --- | --- | --- | --- | --- |
| output/results/c1_r1_combined_rerun | CURRENT-EVIDENCE | C1.2, C1.3 | New C1 manuscript analysis and audited C1-R1 observations | Part 2 C1 audit |
| output/results/evidence_audit_part2 | CURRENT-EVIDENCE | C1.1-C1.9 lineage | Audited C1 derived artifacts | Part 2 C1 audit |
| output/results/evidence_audit_part3 | CURRENT-EVIDENCE | C2.1-C2.6 lineage | Audited C2 derived artifacts | Part 3 C2 audit |
| output/results/unified_runtime | LEGACY-DATA / AUDIT-ONLY | C1.1, C1.4, C2.3 | Historical auditing only | Part 2 and this register |
| output/results/phase9_step2 | LEGACY-DATA / AUDIT-ONLY | C1.2, C1.8, C2.2, C4.5, C4.6 | Historical auditing and lineage repair only | Part 2 and Part 7 propagation |
| output/results/phase9_step3 | LEGACY-DATA / AUDIT-ONLY | C1.3, C1.7, C4.4 | Historical auditing and lineage repair only | Part 2 and Part 7 propagation |
| output/results/runtime_attribution | LEGACY-DATA / AUDIT-ONLY | C1.9, C3.2, C4.1 | Part 4/5 historical auditing only | Part 4 and Part 5 audits |
| output/results/phase9_step4_5 | LEGACY-DATA / AUDIT-ONLY | C4.3 | Part 5 historical auditing only; rounded single-process trace | Part 5 audit |
| output/results/integration_validation | LEGACY-DATA / AUDIT-ONLY | C2.3, C2.5, C4.7 | Part 3/5 historical auditing only | Part 3 and Part 5 audits |
| output/results/evidence_audit_part4 | CURRENT-EVIDENCE | C3.1-C3.6 | Audited C3 wording and derived artifacts | Part 4 C3 audit |
| output/results/evidence_audit_part5 | CURRENT-EVIDENCE | C4.1-C4.7 | Audited C4 wording and derived artifacts | Part 5 C4 audit |
| output/results/evidence_audit_x5_5 | CURRENT-EVIDENCE | C1-C4 triage | Final contribution disposition and X6.5 waiver | X5.5 triage |
| output/results/evidence_audit_part6 | CURRENT-EVIDENCE | Cross-Claim overlay | Statistical integrity and paper eligibility | X6 audit |

New C1 manuscript analysis must read from output/results/c1_r1_combined_rerun/
or output/results/evidence_audit_part2/. Legacy data may be used only for
historical auditing, C2.2 lineage repair, or a newly registered Claim whose
consumer explicitly names the legacy protocol and limitation.

## 5. Claim Propagation Matrix

Each row names the legacy expression, its affected lineage, and the only
condition under which a paper-facing claim may be released.

## Quarantine Matrix

| Quarantine ID | Affected Claim IDs | Legacy Expression | Affected Documents/Data | Disposition | Canonical Source | Paper-Use Rule | Audit Owner / State | Explicit Release Condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q-01 | C1.1 | Historical 596ms to 3.0ms component speedup | Method; story freeze; unified runtime traces | PERMANENTLY-EXCLUDED | Part 2 C1 audit | Never release the historical number | Part 2 closed; legacy retained for audit | A new matched component/step claim requires a new Claim ID, a new Part 2 audit, and Part 7 propagation. |
| Q-02 | C1.4 | Historical 674ms to 79.7ms step-time speedup | Method; Phase 8 trace data | PERMANENTLY-EXCLUDED | Part 2 C1 audit | Never release the historical number | Part 2 closed; legacy retained for audit | A new matched component/step claim requires a new Claim ID, a new Part 2 audit, and Part 7 propagation. |
| Q-03 | C1.2 | Historical 25.1s to 4.4s epoch-time result | Outline; story freeze; Phase 9 Step 2 data | SUPERSEDED-PENDING-PROPAGATION | Part 2 C1 audit; C1.2-R1/E1 | Old number is never released | Part 7 pending | Part 7 must replace it with C1.2-R1/E1 wording. |
| Q-04 | C1.3 | Historical 28.5ms to 0.2ms dispersion result | Outline; story freeze; Phase 9 Step 3 data | SUPERSEDED-PENDING-PROPAGATION | Part 2 C1 audit; C1.3-R1/E2 | Old number is never released | Part 7 pending | Part 7 must replace it with C1.3-R1/E2 wording. |
| Q-05 | C3.1 | Cost-model R²=0.9008 efficacy narrative | Method; story freeze; cost-model materials | PREDICTIVE-RETRACTED-IMPLEMENTATION-ONLY | Part 4 C3 audit | Do not release predictive R² wording; implementation facts only | Part 4 closed; X5.5 triage pending | Release any predictive wording only after a new X6.5-approved out-of-sample Claim and Part 7 propagation. |
| Q-06 | C2.4, C2.6 | Negligible scheduler or runtime-overhead narrative | Method; runtime specification; legacy scheduler results | PARTIALLY-PROHIBITED | Part 3 C2 audit | Deterministic construction/O(1) lookup may use C2.4 wording now; do not claim negligible end-to-end overhead | C2.4 A; C2.6 D | An end-to-end negligible-overhead statement requires a new measured Claim graded A/B and Part 7 propagation; C2.6 remains D. |
| Q-07 | C4.1, C4.3, C4.4, C4.7 | CPU scheduler treatment/variance benefit narrative | Method; story freeze; Phase 9 and attribution data | COMPOSITE-FAIL-SORTER-CANDIDATE | Part 5 C4 audit | Do not release FFD/packing/CBP contribution wording | Part 5 closed; X5.5 pending | Release only after X5.5 selects a sorter-only or new CBP Claim, X6.5 supplies the approved evidence, and Part 7 propagates the wording. |
| Q-08 | C2.1 | Legacy four- or five-layer unified-runtime architecture | Method; story freeze; runtime specification; Phase 8 freeze | SUPERSEDED-PENDING-PROPAGATION | Part 3 C2 audit; C2.1-R1; unified architecture freeze | Use only the canonical implemented-layer wording | Part 7 pending | Part 7 replaces the old layer narrative with C2.1-R1; RuntimePolicy/GPUExecution remain future extensions. |
| Q-09 | C1.5, C1.8, C4.5, C4.6 | Quality preservation, convergence, or comparability narrative | Method; outline; story freeze; Phase 9 Step 2 data | PROHIBITED | X0 quality boundary; Part 2 C1 audit | Do not release quality claims | New quality audit required | Release requires a new full-convergence, valid official-test quality Claim audited at A/B and Part 7 propagation. |
| Q-10 | C1.6, C1.9, C2.5 | Sampler VRAM, bottleneck shift, DDP-ready, general, or SOTA narrative | Method; outline; runtime specification; legacy profiling data | PROHIBITED/OUT-OF-SCOPE | X0 scope and external-validity boundaries | Do not release any branch without its own evidence | Separate audits required | Sampler-only VRAM requires an isolated A/B Claim; bottleneck shift requires unified timing boundaries and A/B audit; DDP-ready requires a real distributed execution Claim; general/SOTA wording requires separately registered generalization/comparator evidence. Each branch also requires Part 7 propagation. |
| Q-11 | C2.3 | Transparent, automatic, or drop-in CPU-to-GPU backend wording | Method; runtime specification; Phase 8 design | PROHIBITED | Part 3 C2 audit | Current Part 7 wording must say the training loop explicitly selects the backend | Interface implementation pending | Transparent/automatic/drop-in backend wording requires a newly implemented and audited automatic-selection interface. |

## 6. Part 1 Claim-ID Reverse Index

The following entries preserve frozen Part 1 lineage only. Replacement IDs are
not Part 1 IDs and appear only in the propagation matrix.

## Claim Reverse Index

| Part 1 Claim ID | Quarantine ID |
| --- | --- |
| C1.1 | Q-01 |
| C1.4 | Q-02 |
| C1.2 | Q-03 |
| C1.3 | Q-04 |
| C3.1 | Q-05 |
| C2.4 | Q-06 |
| C2.6 | Q-06 |
| C4.1 | Q-07 |
| C4.3 | Q-07 |
| C4.4 | Q-07 |
| C4.7 | Q-07 |
| C2.1 | Q-08 |
| C1.5 | Q-09 |
| C1.8 | Q-09 |
| C4.5 | Q-09 |
| C4.6 | Q-09 |
| C1.6 | Q-10 |
| C1.9 | Q-10 |
| C2.5 | Q-10 |
| C2.3 | Q-11 |

## 7. Writing Prohibitions and Release-State Tracker

No legacy numerical, architectural, contribution, quality, memory,
distributed-execution, generalization, comparator, or automatic-backend claim
may enter a manuscript merely because it appears in a legacy document. The
matrix disposition is controlling: permanently excluded and prohibited claims
remain unavailable; pending claims require their named audit; superseded claims
require Part 7 propagation using their canonical replacement wording.

## 8. Part 7 Handoff and Release Procedure

Before paper or figure edits, identify the proposed Claim ID in the matrix,
confirm its release state, read the named canonical source, and propagate only
the audited wording. Record the consumer document and figure in Part 7. If the
claim is new, register it, audit it to the required grade, and then run Part 7
propagation; do not reuse a legacy number as a shortcut.

## 9. Reproduction

Run the deterministic gate twice and compare its output bytes:

    python3 scripts/check_x0_5_quarantine.py --repo-root . --output output/results/evidence_audit_x0_5/quarantine_checks.json
    cp output/results/evidence_audit_x0_5/quarantine_checks.json /tmp/x0_5_checks_first.json
    python3 scripts/check_x0_5_quarantine.py --repo-root . --output output/results/evidence_audit_x0_5/quarantine_checks.json
    cmp /tmp/x0_5_checks_first.json output/results/evidence_audit_x0_5/quarantine_checks.json
