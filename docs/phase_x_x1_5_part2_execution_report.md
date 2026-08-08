# Phase X1.5 — Part 2 Execution Report

**Execution scope:** retrieval snapshot, automatic topic screening, and the
first conservative human-adjudication batch.  This report does not release a
global novelty claim and does not replace the frozen protocol.

## 1. Frozen corpus and retrieval state

The replay used the 51-page snapshot recorded in
`output/results/evidence_audit_x1_5/retrieval_manifest.json`.  The snapshot
contains 2,455 deduplicated records (`records.csv`) and preserves failed DBLP
pages rather than silently dropping them.  The current fallback table records
OpenAlex/Crossref coverage for the failed DBLP queries
(`fallback_coverage.csv`).

The retrieval channel remains open. The persistent retry ledger has migrated a
14-query universe: 3 queries are recovered from the prior snapshot and 11 are
pending at round 1; `completed_rounds` is 0. No DBLP query is treated as
resolved by absence; after three low-frequency retry rounds, any remaining
page will be recorded as `UNRESOLVED_MISSING_PAGE` and evaluated for qualified
OpenAlex/Crossref fallback.

## 2. Screening and adjudication

The automatic topic gate excluded only obvious metadata with no KGE/runtime
signal.  The current derivation reports 763 automatic exclusions, 1,682
uncertain records, 183 potential component records, and 110 potential direct
C1 records (`screening_summary.json`).  The uncertain queue is intentionally
not converted into exclusions.

The first manual batch contains eight records in
`output/results/evidence_audit_x1_5/manual_adjudications.json`.  Each has an
explicit peer-review status, topic boundary, overlap class, mechanism and
integration coding, a remote full-text locator, and locator-backed evidence.
The updated derivation records eight included records, eight
`REMOTE_LOCATED` manifest rows, and sixteen evidence-extraction rows.  Remote
locators are not local downloads and therefore carry an empty SHA-256 by
design; no local hash is fabricated.

The batch is deliberately conservative.  DGL-KE, SMORE, and the multiple-GPU
KGE framework are coded as functional overlaps; PyTorch-BigGraph, Marius, and
NSCaching are coded as strong components; Fast KGE is retained as adjacent
system context; MariusGNN is retained as an adjacent/strong-component system
record.  These are screening codes, not a final C1 verdict.

## 3. Current gate

`novelty_decision.json` emits exactly one C1 verdict:

```text
verdict: UNRESOLVED
confidence: LOW
blockers: human_adjudication_pending, peer_review_status_unverified, retrieval_channel_open
```

This is the required fail-closed state.  The remaining queue and unresolved
peer-review metadata prevent `RETAIN`, `NARROW`, `REFRAME`, or `DROP` from being
used as a paper-level conclusion.  The eight manually adjudicated records are
not counted as proof that C1 is novel or non-novel.

## 4. Replay and validation

The adjudication layer is replayable through `--adjudication-file`; targets are
matched by record ID, DOI, or normalized title/year and unknown targets are
reported as blockers.  `REMOTE_LOCATED` is accepted for evidence tracing but
does not create a local full-text hash.  The complete CPU-only test and
determinism results are recorded in the final handoff message and in
`audit_checks.json`.

## 5. Next gate

Continue high-priority manual adjudication for the remaining potential-direct
queue, then process the fixed DBLP retry batches at the protocol's low rate.
Only after all direct candidates have peer-review, full-text, and evidence
status can the C1 precedence rule issue a non-`UNRESOLVED` verdict.
