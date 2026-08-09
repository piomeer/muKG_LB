from __future__ import annotations

import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_x1_5_literature import (
    assess_c1_verdict,
    build_query_url,
    build_retrieval_plan,
    deduplicate_records,
    fetch_index_page,
    fetch_protocol_queries,
    retry_dblp_batch,
    build_evidence_extraction,
    build_fulltext_manifest,
    build_dblp_retry_schedule,
    build_fallback_coverage,
    qualify_fallback_coverage,
    initialize_dblp_retry_state,
    select_next_dblp_batch,
    assess_dblp_retry_closure,
    retry_dblp_next,
    assess_retrieval_cutoff,
    run_dual_screening,
    load_protocol,
    normalize_record,
    run_self_test,
    screen_record,
    classify_c1_potential,
    apply_adjudications,
    write_retrieval_snapshot,
    write_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "docs/phase_x_x1_5_literature_audit_protocol.json"


class AuditX15LiteratureTests(unittest.TestCase):
    def test_retry_state_migrates_fourteen_initial_failures_with_three_recovered(self) -> None:
        pages = [
            {"index": "DBLP", "query": f"recovered-{i}", "retrieval_stage": "G0_SEEDS", "status": "OK", "retry_round": 1, "hit_count": 1, "raw_payload_sha256": "a" * 64}
            for i in range(3)
        ] + [
            {"index": "DBLP", "query": f"failed-{i}", "retrieval_stage": "G0_SEEDS", "status": "FAILED", "retry_round": 0, "hit_count": 0, "raw_payload_sha256": "b" * 64}
            for i in range(11)
        ]
        state = initialize_dblp_retry_state(pages, batch_size=3, max_rounds=3)
        self.assertEqual(state["initial_query_count"], 14)
        self.assertEqual(state["recovered_query_count"], 3)
        self.assertEqual(state["pending_query_count"], 11)
        self.assertEqual(state["completed_rounds"], 0)

    def test_next_batch_is_stable_and_does_not_use_reindexed_batch_numbers(self) -> None:
        pages = [
            {"index": "DBLP", "query": f"q{i}", "retrieval_stage": "G0_SEEDS", "status": "FAILED", "retry_round": 0}
            for i in range(5)
        ]
        state = initialize_dblp_retry_state(pages)
        first = select_next_dblp_batch(state)
        self.assertEqual([row["query"] for row in first["queries"]], ["q0", "q1", "q2"])
        state["queries"][0]["next_round"] = 2
        state["queries"][1]["next_round"] = 2
        state["queries"][2]["next_round"] = 2
        second = select_next_dblp_batch(state)
        self.assertEqual([row["query"] for row in second["queries"]], ["q3", "q4"])

    def test_fallback_qualification_requires_exact_seed_match_or_nonempty_wide_result(self) -> None:
        pages = [
            {"index": "DBLP", "query": "Seed Paper", "retrieval_stage": "G0_SEEDS", "status": "FAILED"},
            {"index": "OpenAlex", "query": "Seed Paper", "retrieval_stage": "G0_SEEDS", "status": "OK", "hit_count": 1, "raw_payload_sha256": "a" * 64, "records": [{"title": "Seed Paper", "doi": "10.1/seed"}]},
            {"index": "DBLP", "query": "wide query", "retrieval_stage": "WIDE_SENTINEL", "status": "FAILED"},
            {"index": "Crossref", "query": "wide query", "retrieval_stage": "WIDE_SENTINEL", "status": "OK", "hit_count": 1, "raw_payload_sha256": "b" * 64, "records": [{"title": "A result"}]},
            {"index": "DBLP", "query": "zero query", "retrieval_stage": "G0_SEEDS", "status": "FAILED"},
            {"index": "Crossref", "query": "zero query", "retrieval_stage": "G0_SEEDS", "status": "OK", "hit_count": 0, "raw_payload_sha256": "c" * 64, "records": []},
        ]
        rows = qualify_fallback_coverage(pages)
        by_query = {row["query"]: row for row in rows}
        self.assertEqual(by_query["Seed Paper"]["qualification"], "QUALIFIED")
        self.assertEqual(by_query["wide query"]["qualification"], "QUALIFIED")
        self.assertEqual(by_query["zero query"]["qualification"], "MISSING")

    def test_g0_fallback_accepts_known_doi_when_title_differs(self) -> None:
        rows = qualify_fallback_coverage([
            {"index": "DBLP", "query": "Historical title", "expected_doi": "10.1/known", "retrieval_stage": "G0_SEEDS", "status": "FAILED"},
            {"index": "Crossref", "query": "Historical title", "retrieval_stage": "G0_SEEDS", "status": "OK", "hit_count": 1, "raw_payload_sha256": "d" * 64, "records": [{"title": "Updated title", "doi": "https://doi.org/10.1/known"}]},
        ])
        self.assertEqual(rows[0]["qualification"], "QUALIFIED")

    def test_fallback_closure_is_nonblocking_only_when_all_missing_pages_are_qualified(self) -> None:
        qualified = {"status": "CLOSED", "unresolved_count": 1}
        rows = [{"query": "q", "qualification": "QUALIFIED"}]
        result = assess_dblp_retry_closure(qualified, rows)
        self.assertEqual(result["status"], "CLOSED_WITH_FALLBACK")
        self.assertEqual(result["blocking_conditions"], [])
        rows[0]["qualification"] = "MISSING"
        result = assess_dblp_retry_closure(qualified, rows)
        self.assertEqual(result["status"], "CLOSED_BLOCKED")
        self.assertEqual(result["blocking_conditions"], ["retrieval_gap_uncovered"])
        self.assertEqual(assess_dblp_retry_closure({"status": "CLOSED_WITH_FALLBACK"}, [])["status"], "CLOSED_WITH_FALLBACK")
        self.assertEqual(assess_dblp_retry_closure({"status": "CLOSED_BLOCKED"}, [])["blocking_conditions"], ["retrieval_gap_uncovered"])

    def test_retry_next_refuses_before_interval_without_transport_call(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            pages = [{"index": "DBLP", "query": "q", "page": 0, "rows": 1, "retrieval_stage": "G0_SEEDS", "status": "FAILED", "error": "429", "url": "https://dblp.org", "retrieved_at": "2026-08-08T00:00:00Z", "hit_count": 0, "raw_payload_sha256": "0" * 64, "raw_payload": {}, "records": []}]
            write_retrieval_snapshot(root, pages)
            state = initialize_dblp_retry_state(pages)
            state["last_batch_completed_at"] = "2026-08-08T00:00:00Z"
            state_path = root / "dblp_retry_state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            before = state_path.read_bytes()
            calls = []
            result = retry_dblp_next(root, transport=lambda url: calls.append(url) or {"result": {"hits": {"@total": 0, "hit": []}}}, now="2026-08-08T00:05:00Z", sleep_fn=lambda _: None, repo_root=REPO_ROOT)
            self.assertEqual(result["status"], "NOT_DUE")
            self.assertEqual(calls, [])
            self.assertEqual(state_path.read_bytes(), before)

    def test_retry_next_uses_one_request_per_query_and_advances_without_skipping(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            pages = [
                {
                    "index": "DBLP", "query": f"q{i}", "page": 0, "rows": 1,
                    "retrieval_stage": "G0_SEEDS", "status": "FAILED", "error": "429",
                    "url": "https://dblp.org", "retrieved_at": "", "hit_count": 0,
                    "raw_payload_sha256": "0" * 64, "raw_payload": {}, "records": [],
                }
                for i in range(4)
            ]
            write_retrieval_snapshot(root, pages)
            calls: list[str] = []
            sleeps: list[float] = []
            def transport(url: str) -> dict[str, object]:
                calls.append(url)
                return {"result": {"hits": {"@total": 0, "hit": []}}}

            first = retry_dblp_next(root, transport=transport, now="2026-08-08T00:00:00Z", sleep_fn=sleeps.append, repo_root=REPO_ROOT)
            self.assertEqual(first["status"], "BATCH_COMPLETED")
            self.assertEqual(first["updated_query_count"], 3)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(sleeps), 3)
            self.assertTrue(all(3.0 <= value <= 5.0 for value in sleeps))
            schedule = json.loads((root / "retry_schedule.json").read_text(encoding="utf-8"))
            self.assertEqual([row["query"] for row in schedule["next_batch"]], ["q3"])

            second = retry_dblp_next(root, transport=transport, now="2026-08-08T00:10:01Z", sleep_fn=lambda _: None, repo_root=REPO_ROOT)
            self.assertEqual(second["status"], "BATCH_COMPLETED")
            self.assertEqual(second["updated_query_count"], 1)
            self.assertEqual(len(calls), 4)
            state = json.loads((root / "dblp_retry_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["recovered_query_count"], 4)
            self.assertEqual(state["completed_rounds"], 1)
            self.assertEqual(state["status"], "COMPLETE")
    def test_protocol_freezes_scope_and_retrieval_boundaries(self) -> None:
        protocol = load_protocol(REPO_ROOT)

        self.assertEqual(protocol["protocol_id"], "phase-x-x1-5-literature-audit-v1")
        self.assertEqual(protocol["publication_window"], [2013, "2026-08-07"])
        self.assertEqual(protocol["language_scope"], ["English"])
        self.assertEqual(protocol["databases"], ["DBLP", "OpenAlex", "Crossref"])
        self.assertNotIn("Google Scholar", protocol["databases"])
        self.assertEqual(protocol["hard_novelty_claim"], "C1")
        self.assertEqual(protocol["confidence_ceiling"], "MODERATE")
        self.assertEqual(protocol["retrieval"]["max_snowball_generation"], 2)

    def test_normalization_and_deduplication_use_declared_precedence(self) -> None:
        raw = [
            {
                "title": "GPU Negative Sampling for KGE",
                "year": "2022",
                "doi": "https://doi.org/10.1000/XYZ",
                "source_index": "Crossref",
                "peer_reviewed": True,
                "language": "English",
            },
            {
                "title": "GPU negative sampling for KGE.",
                "year": 2022,
                "doi": "10.1000/xyz",
                "dblp_key": "conf.example/2022/foo",
                "source_index": "DBLP",
                "peer_reviewed": True,
                "language": "English",
            },
        ]

        normalized = [normalize_record(item) for item in raw]
        unique, decisions = deduplicate_records(normalized)

        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["dedup_key"], "doi:10.1000/xyz")
        self.assertEqual(decisions[0]["reason"], "DOI")

    def test_screening_returns_fixed_reason_codes_and_excludes_unqualified_records(self) -> None:
        record = normalize_record(
            {
                "title": "A preprint on GPU KGE",
                "year": 2024,
                "source_index": "OpenAlex",
                "peer_reviewed": False,
                "language": "English",
            }
        )

        decision = screen_record(record)

        self.assertEqual(decision["decision"], "EXCLUDE")
        self.assertIn("E02", decision["reason_codes"])

    def test_unverified_peer_review_status_is_uncertain_not_excluded(self) -> None:
        record = normalize_record(
            {
                "title": "Knowledge graph index metadata",
                "year": 2024,
                "source_index": "OpenAlex",
                "language": "English",
            }
        )

        decision = screen_record(record)

        self.assertEqual(decision["decision"], "UNCERTAIN")
        self.assertIn("E06", decision["reason_codes"])

    def test_obviously_irrelevant_metadata_is_auto_excluded(self) -> None:
        record = normalize_record(
            {
                "title": "Quantum chemistry of molecular spectra",
                "year": 2024,
                "source_index": "Crossref",
                "language": "English",
            }
        )

        decision = screen_record(record)

        self.assertEqual(decision["decision"], "EXCLUDE")
        self.assertIn("E04", decision["reason_codes"])
        self.assertIn("AUTO_EXCLUDE", decision["notes"])

    def test_potential_direct_c1_candidate_is_flagged_without_emitting_a_verdict(self) -> None:
        record = normalize_record(
            {
                "title": "GPU negative sampling for knowledge graph embeddings",
                "year": 2023,
                "source_index": "Crossref",
                "language": "English",
            }
        )

        self.assertEqual(classify_c1_potential(record), "POTENTIAL_DIRECT")
        self.assertEqual(record["c1_relevance"], "NONE")

    def test_adjudication_updates_a_record_replayably(self) -> None:
        raw = [{"record_id": "r1", "title": "GPU negative sampling for knowledge graph", "year": 2023, "source_index": "OpenAlex"}]

        updated, errors = apply_adjudications(
            raw,
            [{"record_id": "r1", "peer_reviewed": True, "language": "English", "full_text_status": "REMOTE_LOCATED", "full_text_url": "https://example.org/paper.pdf"}],
        )

        self.assertEqual(errors, [])
        self.assertTrue(updated[0]["peer_reviewed"])
        self.assertEqual(updated[0]["full_text_status"], "REMOTE_LOCATED")

    def test_adjudication_record_id_patch_propagates_to_all_same_doi_manifestations(self) -> None:
        raw = [
            {"record_id": "old-id", "title": "Paper", "year": 2024, "doi": "10.1000/x", "source_index": "Crossref"},
            {"record_id": "new-id", "title": "Paper.", "year": 2024, "doi": "10.1000/x", "source_index": "DBLP"},
        ]
        updated, errors = apply_adjudications(
            raw,
            [{"record_id": "old-id", "doi": "10.1000/x", "peer_reviewed": True, "peer_review_status": "VERIFIED"}],
        )
        self.assertEqual(errors, [])
        self.assertEqual([row["peer_review_status"] for row in updated], ["VERIFIED", "VERIFIED"])

    def test_remote_fulltext_is_located_without_fabricating_a_local_hash(self) -> None:
        record = normalize_record(
            {
                "record_id": "remote",
                "title": "GPU negative sampling for knowledge graph",
                "year": 2023,
                "source_index": "Crossref",
                "full_text_status": "REMOTE_LOCATED",
                "full_text_url": "https://example.org/paper.pdf",
            }
        )

        manifest = build_fulltext_manifest(REPO_ROOT, [record])

        self.assertEqual(manifest[0]["status"], "REMOTE_LOCATED")
        self.assertEqual(manifest[0]["sha256"], "")

    def test_c1_verdict_precedence_blocks_on_missing_direct_full_text(self) -> None:
        result = assess_c1_verdict(
            records=[
                {
                    "record_id": "r-direct",
                    "overlap_class": "DIRECT-FUNCTIONAL",
                    "full_text_status": "MISSING",
                    "c1_relevance": "DIRECT",
                }
            ],
            blockers=[],
        )

        self.assertEqual(result["verdict"], "UNRESOLVED")
        self.assertIn("missing_full_text", result["blockers"])

    def test_c1_verdict_is_retain_for_only_strong_component_prior_art(self) -> None:
        result = assess_c1_verdict(
            records=[
                {
                    "record_id": "r-component",
                    "overlap_class": "STRONG-COMPONENT",
                    "full_text_status": "LOCATED",
                    "c1_relevance": "STRONG-COMPONENT",
                }
            ],
            blockers=[],
        )

        self.assertEqual(result["verdict"], "RETAIN")

    def test_self_test_is_green(self) -> None:
        result = run_self_test(REPO_ROOT)

        self.assertTrue(result["passed"])
        self.assertEqual(result["checks_failed"], [])

    def test_query_url_is_protocol_bound_and_rejects_unknown_index(self) -> None:
        url = build_query_url("DBLP", "GPU negative sampling", page=2, rows=25)

        self.assertIn("dblp.org/search/publ/api", url)
        self.assertIn("f=50", url)
        self.assertIn("h=25", url)
        with self.assertRaises(ValueError):
            build_query_url("Google Scholar", "GPU", page=0, rows=10)

    def test_retrieval_plan_contains_all_frozen_seed_and_sentinel_queries(self) -> None:
        plan = build_retrieval_plan(load_protocol(REPO_ROOT))

        self.assertEqual(len(plan), 51)
        self.assertEqual({item["index"] for item in plan}, {"DBLP", "OpenAlex", "Crossref"})
        self.assertEqual({item["retrieval_stage"] for item in plan}, {"G0_SEEDS", "WIDE_SENTINEL"})

    def test_fetch_index_page_normalizes_all_three_index_shapes(self) -> None:
        payloads = {
            "DBLP": {
                "result": {
                    "hits": {
                        "@total": 1,
                        "hit": [{"info": {"title": "DBLP paper", "year": "2021", "key": "conf/x", "doi": "10.1/db"}}],
                    }
                }
            },
            "OpenAlex": {
                "meta": {"count": 1},
                "results": [{"id": "https://openalex.org/W1", "title": "OpenAlex paper", "publication_year": 2022, "doi": "https://doi.org/10.1/oa", "type": "article"}],
            },
            "Crossref": {
                "message": {
                    "total-results": 1,
                    "items": [{"title": ["Crossref paper"], "published": {"date-parts": [[2023]]}, "DOI": "10.1/cr", "type": "proceedings-article"}],
                }
            },
        }

        for index, payload in payloads.items():
            page = fetch_index_page(index, "GPU", page=0, rows=10, transport=lambda _url, p=payload: p)
            self.assertEqual(len(page["records"]), 1)
            self.assertEqual(page["records"][0]["source_index"], index)
            self.assertTrue(page["raw_payload_sha256"])
            self.assertEqual(page["query"], "GPU")

    def test_retrieval_snapshot_serialization_is_deterministic(self) -> None:
        pages = [
            fetch_index_page(
                "DBLP",
                "GPU",
                page=0,
                rows=10,
                transport=lambda _url: {"result": {"hits": {"@total": 0, "hit": []}}},
            )
        ]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths = write_retrieval_snapshot(Path(first), pages)
            second_paths = write_retrieval_snapshot(Path(second), pages)
            first_bytes = {name: path.read_bytes() for name, path in sorted(first_paths.items())}
            second_bytes = {name: path.read_bytes() for name, path in sorted(second_paths.items())}
            self.assertEqual(first_bytes, second_bytes)
            manifest = json.loads((Path(first) / "retrieval_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pages"][0]["retrieved_at"], "")

    def test_failed_retrieval_is_preserved_in_manifest(self) -> None:
        pages = fetch_protocol_queries(
            load_protocol(REPO_ROOT),
            rows=1,
            transport=lambda _url: (_ for _ in ()).throw(RuntimeError("offline")),
        )

        self.assertEqual(len(pages), 51)
        self.assertEqual(pages[0]["status"], "FAILED")
        self.assertEqual(pages[0]["error"], "RuntimeError: offline")
        with tempfile.TemporaryDirectory() as output:
            write_retrieval_snapshot(Path(output), pages)
            manifest = json.loads((Path(output) / "retrieval_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pages"][0]["status"], "FAILED")

    def test_retrieval_retries_transient_transport_failure(self) -> None:
        attempts = {"count": 0}

        def flaky(_url: str) -> dict[str, object]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary")
            return {"result": {"hits": {"@total": 0, "hit": []}}}

        pages = fetch_protocol_queries(
            {"retrieval": {"g0_seeds": [], "wide_sentinel_queries": ["GPU"]}},
            rows=1,
            transport=flaky,
            retries=1,
            retry_delay=0,
        )

        self.assertEqual(len(pages), 3)
        self.assertEqual(attempts["count"], 4)
        self.assertEqual(pages[0]["status"], "OK")

    def test_dblp_retry_schedule_is_batched_and_has_deterministic_3_to_5_second_jitter(self) -> None:
        pages = [
            {"index": "DBLP", "query": f"q{i}", "page": 0, "retrieval_stage": "G0_SEEDS", "status": "FAILED", "error": "429"}
            for i in range(7)
        ]

        schedule = build_dblp_retry_schedule(pages, round_number=1, batch_size=3)

        self.assertEqual([len(batch["queries"]) for batch in schedule["batches"]], [3, 3, 1])
        delays = [query["delay_seconds"] for batch in schedule["batches"] for query in batch["queries"]]
        self.assertTrue(all(3.0 <= delay <= 5.0 for delay in delays))
        self.assertEqual(schedule, build_dblp_retry_schedule(pages, round_number=1, batch_size=3))

    def test_retrieval_cutoff_marks_missing_pages_after_three_rounds(self) -> None:
        pages = [{"index": "DBLP", "query": "q", "status": "FAILED", "retry_round": 3}]

        cutoff = assess_retrieval_cutoff(pages, completed_rounds=3)

        self.assertEqual(cutoff["status"], "CLOSED")
        self.assertEqual(cutoff["unresolved_count"], 1)
        self.assertEqual(cutoff["unresolved_status"], "UNRESOLVED_MISSING_PAGE")

    def test_failed_dblp_query_can_be_covered_by_successful_alternative_source(self) -> None:
        pages = [
            {"index": "DBLP", "query": "same title", "retrieval_stage": "G0_SEEDS", "status": "FAILED"},
            {"index": "OpenAlex", "query": "same title", "retrieval_stage": "G0_SEEDS", "status": "OK", "hit_count": 1, "raw_payload_sha256": "a" * 64, "records": [{"title": "same title", "source_identifier": "oa:1"}]},
            {"index": "Crossref", "query": "other title", "retrieval_stage": "G0_SEEDS", "status": "OK"},
        ]

        coverage = build_fallback_coverage(pages)

        self.assertEqual(coverage[0]["fallback_status"], "AVAILABLE")
        self.assertEqual(coverage[0]["fallback_indexes"], "OpenAlex")

    def test_retry_dblp_batch_updates_only_selected_queries(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            pages = [
                {
                    "index": "DBLP",
                    "query": f"q{i}",
                    "page": 0,
                    "rows": 1,
                    "retrieval_stage": "G0_SEEDS",
                    "status": "FAILED",
                    "error": "429",
                    "url": "https://dblp.org",
                    "retrieved_at": "",
                    "hit_count": 0,
                    "raw_payload_sha256": "" * 64,
                    "raw_payload": {},
                    "records": [],
                }
                for i in range(4)
            ]
            for page in pages:
                page["raw_payload_sha256"] = "0" * 64
            from scripts.audit_x1_5_literature import write_retrieval_snapshot

            write_retrieval_snapshot(root, pages)
            result = retry_dblp_batch(
                root,
                round_number=1,
                batch_index=0,
                batch_size=2,
                rows=1,
                transport=lambda _url: {"result": {"hits": {"@total": 0, "hit": []}}},
                honor_wait=False,
                repo_root=REPO_ROOT,
            )

            manifest = json.loads((root / "retrieval_manifest.json").read_text(encoding="utf-8"))["pages"]
            status_by_query = {page["query"]: page["status"] for page in manifest}
            self.assertEqual(status_by_query["q0"], "OK")
            self.assertEqual(status_by_query["q1"], "OK")
            self.assertEqual(status_by_query["q2"], "FAILED")
            self.assertEqual(result["batch_index"], 0)

    def test_retry_dblp_batch_preserves_manual_adjudications(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            raw_record = {"title": "GPU negative sampling", "year": 2024, "source_index": "DBLP", "query": "q1", "retrieval_stage": "G0_SEEDS"}
            pages = [
                {
                    "index": "DBLP", "query": "q0", "page": 0, "rows": 1,
                    "retrieval_stage": "G0_SEEDS", "status": "FAILED", "error": "429",
                    "url": "https://dblp.org", "retrieved_at": "", "hit_count": 0,
                    "raw_payload_sha256": "0" * 64, "raw_payload": {}, "records": [],
                },
                {
                    "index": "DBLP", "query": "q1", "page": 0, "rows": 1,
                    "retrieval_stage": "G0_SEEDS", "status": "FAILED", "error": "429",
                    "url": "https://dblp.org", "retrieved_at": "", "hit_count": 0,
                    "raw_payload_sha256": "0" * 64, "raw_payload": {}, "records": [raw_record],
                },
            ]
            from scripts.audit_x1_5_literature import normalize_record, write_retrieval_snapshot

            write_retrieval_snapshot(root, pages)
            manual_id = normalize_record(raw_record)["record_id"]
            (root / "manual_adjudications.json").write_text(
                json.dumps([{"record_id": manual_id, "peer_reviewed": True, "language": "English", "peer_review_status": "VERIFIED"}]),
                encoding="utf-8",
            )
            retry_dblp_batch(
                root, round_number=1, batch_index=0, batch_size=1, rows=1,
                transport=lambda _url: {"result": {"hits": {"@total": 0, "hit": []}}},
                honor_wait=False, repo_root=REPO_ROOT,
            )
            with (root / "records.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["peer_review_status"], "VERIFIED")

    def test_unverified_index_metadata_cannot_emit_retain_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            paths = write_outputs(
                REPO_ROOT,
                Path(output),
                [
                    {
                        "title": "GPU negative sampling for knowledge graph",
                        "year": 2022,
                        "doi": "10.1/candidate",
                        "source_index": "OpenAlex",
                        "language": "English",
                        "overlap_class": "DIRECT-FUNCTIONAL",
                        "c1_relevance": "DIRECT",
                        "full_text_status": "LOCATED",
                    }
                ],
            )
            decision = json.loads(paths["novelty_decision.json"].read_text(encoding="utf-8"))
            self.assertEqual(decision["verdict"], "UNRESOLVED")
            self.assertIn("peer_review_status_unverified", decision["blockers"])

    def test_dual_screening_records_conflict_for_human_adjudication(self) -> None:
        record = normalize_record(
            {
                "title": "GPU negative sampling runtime",
                "year": 2022,
                "doi": "10.1/direct",
                "source_index": "OpenAlex",
                "peer_reviewed": True,
                "language": "English",
                "topic_relevant": True,
                "c1_relevance": "DIRECT",
                "overlap_class": "DIRECT-FUNCTIONAL",
                "full_text_status": "MISSING",
            }
        )

        result = run_dual_screening([record])

        self.assertEqual(len(result["decisions"]), 2)
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["conflicts"][0]["status"], "PENDING")

    def test_fulltext_manifest_hashes_located_files_and_blocks_direct_missing(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            pdf = root_path / "paper.pdf"
            pdf.write_bytes(b"fixture-pdf")
            records = [
                normalize_record({"record_id": "located", "title": "Located", "year": 2022, "source_index": "DBLP", "full_text_path": str(pdf), "full_text_status": "LOCATED"}),
                normalize_record({"record_id": "missing", "title": "Missing", "year": 2022, "source_index": "DBLP", "c1_relevance": "DIRECT", "overlap_class": "DIRECT-FUNCTIONAL", "full_text_status": "MISSING"}),
            ]

            manifest = build_fulltext_manifest(root_path, records)

            self.assertEqual(manifest[0]["status"], "LOCATED")
            self.assertTrue(manifest[0]["sha256"])
            self.assertEqual(manifest[1]["status"], "MISSING_DIRECT_CANDIDATE")

    def test_evidence_extraction_requires_nonempty_locators(self) -> None:
        records = [
            normalize_record(
                {
                    "record_id": "r1",
                    "title": "Evidence",
                    "year": 2022,
                    "source_index": "DBLP",
                    "evidence": {"device_placement": {"value": "GPU", "locator": "p. 4"}},
                }
            ),
            normalize_record(
                {
                    "record_id": "r2",
                    "title": "No locator",
                    "year": 2022,
                    "source_index": "DBLP",
                    "evidence": {"runtime": {"value": "integrated", "locator": ""}},
                }
            ),
        ]

        rows, errors = build_evidence_extraction(records)

        self.assertEqual(len(rows), 1)
        self.assertEqual(errors, ["r2:runtime:missing_locator"])

    def test_default_outputs_are_byte_deterministic_and_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths = write_outputs(REPO_ROOT, Path(first))
            second_paths = write_outputs(REPO_ROOT, Path(second))

            first_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(first).iterdir())
                if path.is_file()
            }
            second_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(second).iterdir())
                if path.is_file()
            }
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(set(first_paths), set(second_paths))
            json.loads((Path(first) / "novelty_decision.json").read_text(encoding="utf-8"))
            json.loads((Path(first) / "audit_checks.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
