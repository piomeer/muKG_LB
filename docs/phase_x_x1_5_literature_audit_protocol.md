# Phase X1.5 — Literature Mapping and C1 Novelty Audit Protocol

**Protocol ID:** `phase-x-x1-5-literature-audit-v1`
**Version:** 1.0
**Freeze date:** 2026-08-07
**Status:** FROZEN FOR EXECUTION

This is a paper-grade internal systematic mapping and novelty audit. It is not
itself a standalone systematic-review paper. The machine-readable source of
truth is [`phase_x_x1_5_literature_audit_protocol.json`](phase_x_x1_5_literature_audit_protocol.json); this document explains the frozen decisions.

## 1. Objective and boundary

The audit maps peer-reviewed work relevant to GPU negative sampling, KGE
runtime integration, cost-aware scheduling, and reproducibility, then issues a
hard novelty verdict only for C1. C2, C3, and C4 are overlap-risk tracks; they
cannot be promoted to contributions without an X1.5 addendum after their
corresponding Part 4 or Part 5 audit.

The corpus is English-language, formally peer-reviewed work published from
2013 through 2026-08-07. DBLP, OpenAlex, and Crossref are the only indexes.
Google Scholar is excluded. Official documentation, repositories, and
preprints may help discovery but are not prior-art evidence and cannot support
a global “no existing work” statement.

The confidence ceiling is **MODERATE** because the search is English-only,
open-index-only, and seed-first, even though an independent wide sentinel is
required to reduce recall risk.

## 2. Questions and retrieval

The mapping questions are MQ1 (GPU negative sampling), MQ2 (runtime and
training-loop integration), MQ3 (cost models, scheduling, and packing), and
MQ4 (evidence and reproducibility). NQ1 asks whether C1 overlaps a prior
peer-reviewed mechanism and evidence protocol strongly enough to change the
claim.

Retrieval uses two complementary paths:

1. **Seed-and-snowball:** G0 contains the frozen seed set; G1 collects all
   backward references and OpenAlex forward citations; G2 continues only from
   records coded `DIRECT` or `STRONG-COMPONENT`. Snowballing stops after G2.
2. **Wide independent sentinel:** four fixed query families are run against
   each of DBLP, OpenAlex, and Crossref. Exact query strings, pagination,
   retrieval counts, and raw-response hashes are retained.

Records are deduplicated by DOI, then DBLP key, then normalized title plus year.
Formal versions remain linked as one study family while each manifestation is
retained for traceability.

## 3. Screening and extraction

Neutral eligibility and adversarial prior-art screening are separate channels.
Both emit `INCLUDE`, `EXCLUDE`, or `UNCERTAIN` using fixed reason codes E01–E06.
The adversarial channel additionally uses A01 for a direct C1 candidate that
requires full-text adjudication. Human confirmation is mandatory for channel
conflicts, uncertain boundaries, and direct C1 candidates.

An index row with unknown peer-review status is `UNCERTAIN` (E06), not
`EXCLUDE`; only explicit non-peer-reviewed metadata is excluded (E02).

The automatic topic gate excludes only records whose title and abstract have no
KGE or runtime signal (`AUTO_EXCLUDE_OBVIOUSLY_IRRELEVANT`). Records with a
possible direct C1 mechanism are flagged `POTENTIAL_DIRECT` and receive HIGH
priority in the human queue; this flag is not itself a novelty verdict.

Missing full text for a suspected direct C1 candidate is a blocker: the record
enters the acquisition queue and the C1 verdict remains `UNRESOLVED` until the
candidate is resolved. It is never silently excluded.

Located full texts are hashed in `fulltext_manifest.csv`. Evidence extraction
rows require a non-empty page/section/table locator; missing locators are
reported as a derivation blocker rather than inferred from titles or abstracts.

Extraction covers identity/provenance; model, dataset, and hardware; negative
operation semantics; device placement and transfer; runtime integration;
comparators, repeats, uncertainty, and artifacts; and the conditional C2/C3/C4
facets.

Overlap is coded as `DIRECT-EXACT`, `DIRECT-FUNCTIONAL`,
`STRONG-COMPONENT`, `ADJACENT-SYSTEM`, `SEMANTIC-BACKGROUND`, or `NO-OVERLAP`.

## 4. C1 verdict rule

Exactly one verdict is emitted from `RETAIN`, `NARROW`, `REFRAME`, `DROP`, or
`UNRESOLVED` using this precedence:

1. Incomplete search, unresolved conflict, or missing full text for a suspected
   direct candidate → `UNRESOLVED`.
2. Exact prior with matching integration and evidence → `DROP`.
3. Exact mechanism but a distinct audit/evidence contribution → `REFRAME`.
4. Functional prior → `NARROW`.
5. Only strong-component or weaker overlap → `RETAIN`.

The verdict is a claim-governance decision, not a count of papers and not a
global proof of originality.

## 5. Artifacts and gates

The replayable script is [`scripts/audit_x1_5_literature.py`](../scripts/audit_x1_5_literature.py).
It will produce deterministic records, screening, adjudication, full-text,
extraction, novelty, and audit-check artifacts under
`output/results/evidence_audit_x1_5/`. PDFs are not committed; their paths and
SHA-256 values belong in the full-text manifest.

The retrieval adapter uses only fixed, credential-free URLs for DBLP, OpenAlex,
and Crossref. Each page stores its query, pagination, retrieval stage, raw
payload SHA-256, and a normalized record list. Network transport is injectable
for tests and is not invoked by the CPU self-test.

DBLP failures are retried in batches of at most three queries. Each query gets a
deterministic 3–5 second jitter, and successive batches carry a 600-second
minimum wait marker. After three completed rounds, remaining failures are
closed as `UNRESOLVED_MISSING_PAGE`; `fallback_coverage.csv` records whether the
same title/query was available from OpenAlex or Crossref.

Given the same raw snapshot, normalization and derivation must be byte-identical
across two runs. Dynamic timestamps, temporary paths, and random UUIDs are
forbidden in derived outputs. The audit cannot modify training/runtime code or
paper-body text and does not require GPU execution.

The complete protocol, seed metadata, reason codes, verdict precedence, and
acceptance gates are machine-checked from the adjacent JSON file.
