# X1.5 DBLP Retry and Fallback Closure Addendum

This addendum supplements, but does not rewrite, the frozen
`phase-x-x1-5-literature-audit-v1` protocol. It governs the operational closure
of the DBLP channel after the first snapshot exposed partial DBLP availability.

## Retry contract

- The retry universe is fixed by query identity `(retrieval_stage, exact_query)`.
- Each query receives at most one DBLP request per round and at most three rounds.
- A batch contains at most three queries. Query-level waits use deterministic
  3–5 second jitter; adjacent batches are separated by at least 600 seconds.
- The persistent `dblp_retry_state.json` ledger selects the next batch. A
  shifting `batch_index` is not an execution contract.
- If the minimum interval has not elapsed, the command returns `NOT_DUE` without
  network access or artifact mutation.

## Fallback qualification

For a G0 seed, an OpenAlex or Crossref page qualifies only when the response is
successful, the raw payload hash is valid, and at least one record has an exact
normalized-title or canonical DOI match. For a wide sentinel, the response must
be successful, hashed, and contain at least one normalized record. One qualified
alternative index is sufficient.

## Closure and C1 propagation

`retrieval_cutoff.json.status` is one of `OPEN`, `COMPLETE`,
`CLOSED_WITH_FALLBACK`, or `CLOSED_BLOCKED`; the per-query ledger still records
`UNRESOLVED_MISSING_PAGE` before the aggregate disposition is written.

After three rounds, unresolved DBLP pages retain the per-query disposition
`UNRESOLVED_MISSING_PAGE`. If every such query has qualified fallback, the
channel is `CLOSED_WITH_FALLBACK`; the report carries an advisory
`dblp_missing_page_with_qualified_fallback`, but C1 receives no retrieval hard
blocker. If any unresolved query lacks qualified fallback, the channel is
`CLOSED_BLOCKED` and C1 receives `retrieval_gap_uncovered`. Before closure, the
channel is `OPEN` and C1 receives `retrieval_channel_open`.

This addendum does not decide C1 novelty. Human adjudication, full-text evidence
and the final RETAIN/NARROW/REFRAME/DROP decision remain separate gates.
