from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.expand_x1_5_snowball import (
    build_openalex_cites_url,
    extract_seed_work_ids,
    build_snowball_plan,
    parse_forward_page,
    select_forward_tasks,
    write_snowball_outputs,
)


class SnowballExpansionTests(unittest.TestCase):
    def test_seed_matching_extracts_openalex_work_ids(self) -> None:
        pages = [{"index": "OpenAlex", "query": "seed", "retrieval_stage": "G0_SEEDS", "raw_payload": {"results": [{"id": "https://openalex.org/W1", "title": "DGL-KE: Training Knowledge Graph Embeddings at Scale"}]}}]
        result = extract_seed_work_ids(pages, ["DGL-KE: Training Knowledge Graph Embeddings at Scale"])
        self.assertEqual(result, [{"seed_title": "DGL-KE: Training Knowledge Graph Embeddings at Scale", "work_id": "W1"}])

    def test_plan_contains_g1_backward_forward_and_g2_component_tasks(self) -> None:
        plan = build_snowball_plan(
            [{"seed_title": "seed", "work_id": "W1", "referenced_works": ["https://openalex.org/W2"]}],
            [{"record_id": "r", "title": "component", "c1_relevance": "DIRECT", "openalex_work_id": "W3"}],
        )
        self.assertEqual([row["generation"] for row in plan["tasks"]], ["G1_BACKWARD", "G1_FORWARD", "G2_FORWARD"])
        self.assertIn("cites:W1", plan["tasks"][1]["filter"])
        self.assertIn("cites:W3", plan["tasks"][2]["filter"])

    def test_forward_parser_normalizes_records_and_preserves_parent(self) -> None:
        payload = {"meta": {"count": 1}, "results": [{"id": "https://openalex.org/W4", "title": "A paper", "publication_year": 2022, "doi": "https://doi.org/10.1000/x", "primary_location": {"source": {"display_name": "Venue"}}}]}
        page = parse_forward_page(payload, "W1", "G1_FORWARD", 1)
        self.assertEqual(page["hit_count"], 1)
        self.assertEqual(page["records"][0]["source_identifier"], "https://openalex.org/W4")
        self.assertEqual(page["records"][0]["snowball_parent_work_id"], "W1")

    def test_snowball_outputs_are_deterministic(self) -> None:
        plan = {"tasks": [{"generation": "G1_FORWARD", "work_id": "W1", "filter": "cites:W1"}]}
        pages = [parse_forward_page({"meta": {"count": 0}, "results": []}, "W1", "G1_FORWARD", 1)]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = write_snowball_outputs(Path(first), plan, pages, [])
            b = write_snowball_outputs(Path(second), plan, pages, [])
            self.assertEqual({k: p.read_bytes() for k, p in a.items()}, {k: p.read_bytes() for k, p in b.items()})
            json.loads((Path(first) / "snowball_status.json").read_text(encoding="utf-8"))

    def test_citation_url_is_stable(self) -> None:
        self.assertEqual(build_openalex_cites_url("W1", cursor="*", per_page=200), "https://api.openalex.org/works?filter=cites%3AW1&per-page=200&cursor=%2A")

    def test_forward_task_selection_can_bound_low_frequency_batches(self) -> None:
        tasks = [{"direction": "FORWARD", "generation": "G1_FORWARD", "work_id": f"W{i}"} for i in range(3)]
        self.assertEqual([row["work_id"] for row in select_forward_tasks(tasks, generation="G1_FORWARD", offset=1, limit=1)], ["W1"])

    def test_batch_status_records_selected_task_count_without_claiming_full_completion(self) -> None:
        plan = {"tasks": [
            {"generation": "G1_FORWARD", "direction": "FORWARD", "parent_work_id": "W1"},
            {"generation": "G1_FORWARD", "direction": "FORWARD", "parent_work_id": "W2"},
        ]}
        pages = [parse_forward_page({"meta": {"count": 0}, "results": []}, "W1", "G1_FORWARD", 1)]
        with tempfile.TemporaryDirectory() as output:
            write_snowball_outputs(
                Path(output), plan, pages, [], selected_tasks=[plan["tasks"][0]]
            )
            status = json.loads((Path(output) / "snowball_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["selected_forward_task_count"], 1)
        self.assertEqual(status["selected_fetched_forward_task_count"], 1)
        self.assertEqual(status["fetched_forward_task_count"], 1)
        self.assertEqual(status["status"], "PARTIAL")
