# Phase X1.5 Part 3 — Mapping and Novelty Synthesis Design

## Purpose

Add a deterministic, read-only analysis layer over the X1.5 Part 2
retrieval/screening artifacts. The layer produces a systematic mapping for
MQ1–MQ4 and an auditable C1 novelty evidence matrix without performing new
retrieval, downloading papers, modifying runtime code, or releasing the C1
fail-closed gate.

## Architecture

`scripts/synthesize_x1_5_mapping.py` reads only the frozen output directory
`output/results/evidence_audit_x1_5/`. It normalizes CSV/JSON inputs, derives
topic facets from explicit record fields and locator-backed evidence, and writes
deterministic CSV/JSON/Markdown artifacts. It has no network transport and no
write path outside the requested output/report locations.

## Outputs

- `literature_mapping.csv`: one row per included or manually adjudicated record,
  with MQ1–MQ4 facet flags, study family, peer-review/full-text status, and
  provenance fields.
- `novelty_evidence_matrix.csv`: C1-relevant rows with overlap class,
  mechanism/integration/evidence flags, evidence locator count, and blocking
  conditions.
- `mapping_summary.json`: record and study-family counts, MQ coverage, overlap
  distribution, unresolved counts, and the inherited C1 gate state.
- `coverage_checks.json`: schema, path, count, provenance, and fail-closed
  checks.
- `docs/phase_x_x1_5_part3_mapping_synthesis.md`: human-readable synthesis
  whose claims are limited to derived artifacts and explicitly reports recall
  and adjudication limitations.

## Boundary rules

MQ flags are derived from explicit coded fields (`c1_relevance`, overlap and
integration fields, evidence facets) plus conservative metadata terms; they are
not novelty judgments. A missing locator, unresolved peer-review status,
failed retrieval page, or pending human adjudication remains a blocker. The
script must preserve exactly one inherited C1 verdict and must never convert
`UNRESOLVED` into a substantive conclusion.

## Validation

CPU fixtures test MQ mapping, study-family aggregation, evidence-blocker
propagation, deterministic output, and rejection of missing required inputs.
The full repository test suite, Python compilation, JSON/CSV parsing, path
checks, and `git diff --check` must pass.
