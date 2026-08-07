# Phase X X0.5 — Legacy Narrative Quarantine Design

**Date**: 2026-08-07
**Status**: Approved design; implementation pending written-spec review
**Scope**: Documentation provenance and claim-propagation control only

## 1. Purpose

Phase X has established canonical evidence that supersedes portions of the
historical paper narrative.  This design prevents those legacy documents from
being accidentally treated as current paper truth while Parts 4 and 5 decide
the status of CostModel and CBP claims.

The quarantine is an evidence-control layer.  It does not revise the paper,
alter the frozen Part 1 inventory, create a new empirical result, or repair
the underlying implementation.

## 2. Authority Model

The implementation will define this precedence order:

1. `docs/phase_x_x0_research_freeze.md` for research questions, scope,
   contribution hierarchy, estimands, and reporting constraints.
2. Evidence-audit reports and their machine-readable artifacts for audited
   claim grades and numerical evidence: Part 2 (C1), Part 3 (C2), then future
   Parts 4 and 5.
3. `docs/evidence_audit_part1_claim_inventory.md` for frozen historical claim
   identifiers and lineage only; its ACTIVE/HOLD/RETRACTED labels are not
   A/B/C/D grades.
4. Historical narrative, draft, and architectural documents for context only.
   They are non-authoritative until a Part 7 propagation pass updates them.

No document can silently override a source that is higher in this order.

## 3. Deliverables

### 3.1 Central quarantine register

Create `docs/phase_x_x0_5_legacy_narrative_quarantine.md` with five sections:

1. purpose, execution boundary, and authority order;
2. document-status register;
3. high-risk claim propagation matrix;
4. writing prohibitions and permitted paper-safe wording sources; and
5. release conditions and Part 7 handoff.

The document-status register will classify each inspected document as one of:

- **CANONICAL**: current source of truth for its declared subject;
- **FROZEN-INVENTORY**: preserves historical claim identifiers and lineage,
  but does not itself establish validity;
- **PENDING-AUDIT**: may contain future candidates, but cannot supply a paper
  conclusion before its designated audit; or
- **LEGACY-NON-AUTHORITATIVE**: contains historical wording or values that must
  not be copied into the manuscript without a canonical-source check.

### 3.2 Legacy-document page headers

Add a short, identical-format quarantine header immediately after the title of
these legacy documents:

- `paper/draft/method.md`;
- `docs/paper_outline.md`;
- `docs/paper_story_freeze.md`;
- `docs/runtime_framework_spec.md`;
- `docs/phase8_architecture_freeze.md`.

The header will link to X0 and the central register, state that the file is
historical/non-authoritative during Phase X, and instruct readers to use the
audits before reusing any numerical, architecture, or contribution claim.  It
will not change substantive body text, headings, figures, code, or artifacts.

`docs/unified_runtime_architecture_freeze.md` is deliberately excluded from
this list: Part 3 declares it canonical for the implementation architecture,
so the status register will identify it as a supporting canonical source rather
than quarantine it.

### 3.3 Claim propagation matrix

Every row will contain: a stable quarantine ID, topic/legacy expression,
affected documents, current disposition, canonical source, paper-use rule,
and release condition.  The initial matrix covers:

| ID | Topic | Current disposition |
|---|---|---|
| Q-01 | 198x component speedup | Excluded; Phase 8 comparator is synthetic |
| Q-02 | 8.5x Phase 8 step speedup | Excluded; same comparator problem |
| Q-03 | 5.7x Phase 9 epoch speedup | Superseded by C1.2-R1 / E1 |
| Q-04 | 142x negative-time dispersion | Superseded by C1.3-R1 / E2 |
| Q-05 | R2=0.9008 / 90% explained | Pending Part 4 provenance and out-of-sample audit |
| Q-06 | CostModel zero/negligible overhead | Only deterministic lookup is established; end-to-end overhead claim prohibited |
| Q-07 | CBP 78% or 8.4% effects | Pending Part 5; fixture equivalence blocks contribution wording |
| Q-08 | Four/five-layer and RuntimePolicy/GPUExecution descriptions | Superseded by Part 3's two-stage, five-role implementation boundary |
| Q-09 | CPU/GPU quality equivalence or non-inferiority | Prohibited; current evaluations do not support it |
| Q-10 | Sampler-only VRAM, bottleneck shift, DDP-ready, general/SOTA claims | Unsupported or out of scope under X0 |

The exact wording will preserve mathematical symbols with plain-text fallbacks
where necessary (for example, `R^2`).

## 4. Data Flow and Interfaces

The quarantine register consumes only existing documents and audit artifacts.
It produces documentation-level routing instructions:

```
legacy narrative occurrence
  -> quarantine matrix row
  -> canonical evidence source
  -> permitted wording or prohibition
  -> Part 4/5/7 release condition
```

There is no training API, runtime API, GPU process, or experiment input/output
in this work.  `PROGRESS.md` and `mukg-memory.json` will record completion and
the next mandatory gate (X1.5) without changing the scientific claims.

## 5. Failure Handling

- If a legacy document contains a high-risk expression not covered by a matrix
  row, classify it as `UNMAPPED` and add a row before declaring X0.5 complete.
- If two canonical sources disagree, record the conflict and defer to the
  higher-precedence source; do not reconcile the science in X0.5.
- If a future audit changes a claim, Part 7 updates the quarantine row and
  lifts or retains the header; X0.5 does not silently rewrite the legacy body.
- If a document is absent, record it as `NOT-PRESENT` rather than inventing a
  lineage record.

## 6. Validation and Acceptance

Implementation is accepted only when:

1. every required legacy file exists and has exactly one quarantine header;
2. each of Q-01 through Q-10 has one disposition, canonical source, and
   release condition;
3. the register names the full authority order and the Part 1 status/grade
   distinction;
4. no substantive legacy body text, code, result artifact, figure, or paper
   claim is rewritten;
5. every referenced local path exists;
6. `rg` checks confirm that the designated high-risk terms are either mapped or
   present only in canonical evidence/register explanations;
7. `git diff --check` passes; and
8. the worktree has no unrelated changes before commit.

## 7. Explicit Non-Goals

- No GPU or CPU experiment, benchmark, profiling run, or code change.
- No rewrite of Method, Introduction, Results, figures, or the story freeze.
- No Part 4/Part 5 scientific conclusion, literature review, or contribution
  promotion.
- No alteration of Part 1's frozen rows or Claim IDs.
- No push to GitHub; publishing remains a separate user-authorized action.

## 8. Handoff

After X0.5, the mandatory research sequence remains X1.5 systematic mapping
and novelty audit, Part 4 C3 audit, Part 5 C4 audit, contribution triage, and
then Part 7 manuscript propagation.  The quarantine register is the required
input to Part 7 so that only released claims enter the regenerated manuscript.
