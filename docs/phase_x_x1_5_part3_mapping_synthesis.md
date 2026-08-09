# Phase X1.5 Part 3 — Systematic Mapping and Novelty Synthesis

This is a deterministic synthesis of the Part 2 artifacts. It does not
perform new retrieval and does not constitute a global novelty proof.

## Mapping coverage

The mapping contains 8 neutral-included records across 8 study families.
Overlap distribution: ADJACENT-SYSTEM=1, DIRECT-FUNCTIONAL=3, STRONG-COMPONENT=4. Locator-backed evidence appears for 8 mapped records.

MQ coverage is recorded in `mapping_summary.json`; facet flags are derived
from explicit coding and conservative metadata signals, not from inferred
paper conclusions.

## C1 novelty position

The inherited C1 gate is **UNRESOLVED** with blockers: human_adjudication_pending, peer_review_status_unverified, retrieval_channel_open.
The novelty matrix contains 7 candidate rows; 7 retain one or more blocking conditions.
No RETAIN, NARROW, REFRAME, or DROP conclusion is released while the
inherited gate is unresolved. Direct and strong-component rows are evidence
for boundary review, not a claim of exact equivalence.

## Limitations and next action

The corpus is English-only and limited to DBLP, OpenAlex, and Crossref;
The DBLP retrieval channel is currently OPEN; 11 DBLP pages remain failed in the preserved snapshot, and most records
remain pending human adjudication. Continue high-priority adjudication and
the protocol-defined DBLP retry rounds before making paper-level novelty
language decisions.

### Candidate matrix trace

| Record | Overlap | Mechanism | Integration | Evidence locators | Blocking conditions |
|---|---|---:|---:|---:|---|
| rec-287dd98564698ec3 | DIRECT-FUNCTIONAL | true | true | 2 | human_adjudication_pending;peer_review_status_unverified;retrieval_channel_open |
| rec-37283377cfa3766c | DIRECT-FUNCTIONAL | true | true | 2 | human_adjudication_pending;peer_review_status_unverified;retrieval_channel_open |
| rec-49d37e2cf4399d16 | DIRECT-FUNCTIONAL | false | false | 2 | human_adjudication_pending;peer_review_status_unverified;retrieval_channel_open |
| rec-4f24693ca07812a8 | STRONG-COMPONENT | true | false | 2 | human_adjudication_pending;peer_review_status_unverified;retrieval_channel_open |
| rec-6eb043c469e35a30 | STRONG-COMPONENT | true | true | 2 | human_adjudication_pending;peer_review_status_unverified;retrieval_channel_open |
| rec-94cf218c869781d0 | STRONG-COMPONENT | true | true | 2 | human_adjudication_pending;peer_review_status_unverified;retrieval_channel_open |
| rec-c92dbdfaf6eeb135 | STRONG-COMPONENT | true | true | 2 | human_adjudication_pending;peer_review_status_unverified;retrieval_channel_open |
