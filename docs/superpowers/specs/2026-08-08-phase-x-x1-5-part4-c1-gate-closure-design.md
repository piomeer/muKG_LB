# Phase X1.5 Part 4 — C1 Gate Closure Audit Design

## Purpose

Provide a deterministic, read-only closure audit for C1. The closure layer
does not adjudicate papers on behalf of a human and does not issue a substantive
novelty conclusion while any protocol blocker remains.

## Inputs and outputs

The script `scripts/close_x1_5_c1_gate.py` consumes the Part 2 and Part 3
artifacts under `output/results/evidence_audit_x1_5/` and writes:

- `c1_adjudication_queue.csv`: unresolved records ordered HIGH before MEDIUM,
  then by stable record ID, with issue and evidence status;
- `c1_source_verification_status.csv`: one row per C1/strong-component
  candidate, including peer-review, full-text, locator, retrieval, and human
  status checks;
- `c1_gate_closure.json`: mechanical gate decision, blockers, counts, and the
  inherited C1 verdict;
- `docs/phase_x_x1_5_part4_c1_gate_closure.md`: deterministic handoff report.

## Gate rules

The gate is `READY_FOR_HUMAN_DECISION` only when the retrieval channel is closed
or complete, no human adjudication is pending, every direct candidate has
verified peer-review and located/remote full text, and every candidate has
locator-backed evidence. Otherwise it remains `UNRESOLVED` with explicit
blockers. The script never changes `novelty_decision.json` and never infers a
verdict from missing data.

## Validation

CPU fixtures test deterministic queue ordering, blocker propagation, the
fail-closed state, and a fully resolved synthetic state. Outputs must be byte
identical across two runs; JSON/CSV/path checks, compilation, full tests, and
`git diff --check` must pass.
