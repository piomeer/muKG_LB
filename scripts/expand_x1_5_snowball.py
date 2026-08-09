#!/usr/bin/env python3
"""Replayable G1/G2 OpenAlex citation expansion for X1.5."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from scripts.audit_x1_5_literature import _canonical_doi, _normalize_title, _openalex_abstract, _payload_hash
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    from audit_x1_5_literature import _canonical_doi, _normalize_title, _openalex_abstract, _payload_hash


DEFAULT_INPUT = Path("output/results/evidence_audit_x1_5")
DEFAULT_OUTPUT = Path("output/results/evidence_audit_x1_5/snowball")
PLAN_FIELDS = ["generation", "direction", "parent_work_id", "work_id", "filter", "seed_title", "page_limit"]
EDGE_FIELDS = ["generation", "direction", "parent_work_id", "child_work_id", "seed_title"]


def build_openalex_cites_url(work_id: str, cursor: str = "*", per_page: int = 200) -> str:
    params = {"filter": f"cites:{work_id.removeprefix('https://openalex.org/')}" , "per-page": per_page, "cursor": cursor}
    return "https://api.openalex.org/works?" + urllib.parse.urlencode(params)


def extract_seed_work_ids(pages: Iterable[dict[str, Any]], seed_titles: Iterable[str]) -> list[dict[str, str]]:
    wanted = {_normalize_title(title): title for title in seed_titles}
    found: dict[str, str] = {}
    for page in pages:
        if page.get("index") != "OpenAlex":
            continue
        payload = page.get("raw_payload", {})
        for item in payload.get("results", []):
            title = _normalize_title(item.get("title", ""))
            work_id = str(item.get("id", "")).removeprefix("https://openalex.org/")
            if title in wanted and work_id:
                found[wanted[title]] = work_id
    return [{"seed_title": title, "work_id": found[title]} for title in sorted(found)]


def build_snowball_plan(
    seed_works: Iterable[dict[str, Any]],
    component_records: Iterable[dict[str, Any]],
    page_limit: int = 0,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for seed in sorted(seed_works, key=lambda row: (str(row.get("seed_title", "")), str(row.get("work_id", "")))):
        parent = str(seed.get("work_id", ""))
        for child in sorted(set(str(item).removeprefix("https://openalex.org/") for item in seed.get("referenced_works", []) if item)):
            tasks.append({"generation": "G1_BACKWARD", "direction": "BACKWARD", "parent_work_id": parent, "work_id": child, "filter": "", "seed_title": str(seed.get("seed_title", "")), "page_limit": page_limit})
        tasks.append({"generation": "G1_FORWARD", "direction": "FORWARD", "parent_work_id": parent, "work_id": parent, "filter": f"cites:{parent}", "seed_title": str(seed.get("seed_title", "")), "page_limit": page_limit})
    for record in sorted(component_records, key=lambda row: (str(row.get("openalex_work_id", "")), str(row.get("record_id", "")))):
        work_id = str(record.get("openalex_work_id", "")).removeprefix("https://openalex.org/")
        if not work_id:
            continue
        tasks.append({"generation": "G2_FORWARD", "direction": "FORWARD", "parent_work_id": work_id, "work_id": work_id, "filter": f"cites:{work_id}", "seed_title": str(record.get("title", "")), "page_limit": page_limit})
    return {"max_generation": 2, "tasks": tasks}


def parse_forward_page(payload: dict[str, Any], parent_work_id: str, generation: str, page: int, seed_title: str = "", cursor: str = "") -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        source = (item.get("primary_location") or {}).get("source") or {}
        records.append({
            "title": item.get("title", ""),
            "year": item.get("publication_year"),
            "doi": item.get("doi", ""),
            "venue": source.get("display_name", ""),
            "source_index": "OpenAlex",
            "source_identifier": item.get("id", ""),
            "retrieval_stage": generation,
            "query": f"cites:{parent_work_id}",
            "language": item.get("language", ""),
            "abstract": _openalex_abstract(item),
            "snowball_parent_work_id": parent_work_id,
            "snowball_seed_title": seed_title,
        })
    return {
        "index": "OpenAlex",
        "generation": generation,
        "direction": "FORWARD",
        "parent_work_id": parent_work_id,
        "seed_title": seed_title,
        "page": page,
        "cursor": cursor,
        "status": "OK",
        "error": "",
        "hit_count": int(payload.get("meta", {}).get("count") or 0),
        "next_cursor": (payload.get("meta") or {}).get("next_cursor") or "",
        "raw_payload_sha256": _payload_hash(payload),
        "raw_payload": payload,
        "records": records,
    }


def _default_transport(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "muKG-LB-X1.5-snowball/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310 - fixed OpenAlex endpoint
        return json.loads(response.read().decode("utf-8"))


def fetch_forward_tasks(
    tasks: Iterable[dict[str, Any]],
    transport: Callable[[str], dict[str, Any]] | None = None,
    per_page: int = 200,
    max_pages: int = 0,
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("direction") != "FORWARD":
            continue
        cursor = "*"
        page_number = 1
        limit = int(max_pages or task.get("page_limit") or 0)
        while True:
            url = build_openalex_cites_url(str(task["work_id"]), cursor=cursor, per_page=per_page)
            try:
                payload = (transport or _default_transport)(url)
                page = parse_forward_page(payload, str(task["parent_work_id"]), str(task["generation"]), page_number, str(task.get("seed_title", "")), cursor)
            except Exception as exc:  # preserve retrieval loss explicitly
                page = {"index": "OpenAlex", "generation": task["generation"], "direction": "FORWARD", "parent_work_id": task["parent_work_id"], "seed_title": task.get("seed_title", ""), "page": page_number, "cursor": cursor, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "hit_count": 0, "next_cursor": "", "raw_payload_sha256": _payload_hash({}), "raw_payload": {}, "records": []}
            page["url"] = url
            pages.append(page)
            reached_limit = bool(limit and page_number >= limit and page.get("next_cursor"))
            page["truncated"] = reached_limit
            if page["status"] != "OK" or not page.get("next_cursor") or reached_limit:
                break
            cursor = str(page["next_cursor"])
            page_number += 1
    return pages


def select_forward_tasks(
    tasks: Iterable[dict[str, Any]],
    generation: str = "",
    offset: int = 0,
    limit: int = 0,
) -> list[dict[str, Any]]:
    selected = [task for task in tasks if task.get("direction") == "FORWARD" and (not generation or task.get("generation") == generation)]
    selected = selected[max(0, offset):]
    return selected if not limit else selected[:limit]


def write_snowball_outputs(
    output_dir: Path,
    plan: dict[str, Any],
    pages: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    selected_tasks: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, page in enumerate(sorted(pages, key=lambda row: (row.get("generation", ""), row.get("parent_work_id", ""), row.get("page", 0)))):
        filename = f"page-{ordinal:05d}-{page['raw_payload_sha256'][:12]}.json"
        (raw_dir / filename).write_text(json.dumps(page.get("raw_payload", {}), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan_path = output_dir / "snowball_plan.json"
    _write_json(plan_path, plan)
    pages_path = output_dir / "snowball_pages.json"
    _write_json(pages_path, [{key: value for key, value in page.items() if key != "raw_payload"} for page in sorted(pages, key=lambda row: (row.get("generation", ""), row.get("parent_work_id", ""), row.get("page", 0)))])
    records = [record for page in pages for record in page.get("records", [])]
    records_path = output_dir / "snowball_records.json"
    _write_json(records_path, sorted(records, key=lambda row: (_normalize_title(row.get("title", "")), str(row.get("year", "")), row.get("source_identifier", ""))))
    edges_path = output_dir / "citation_edges.csv"
    with edges_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for edge in sorted(edges, key=lambda row: (row.get("generation", ""), row.get("parent_work_id", ""), row.get("child_work_id", ""))):
            writer.writerow({field: edge.get(field, "") for field in EDGE_FIELDS})
    selected = list(selected_tasks) if selected_tasks is not None else [
        task for task in plan.get("tasks", []) if task.get("direction") == "FORWARD"
    ]
    selected_keys = {
        (task.get("generation"), task.get("parent_work_id")) for task in selected
    }
    fetched_keys = {
        (page.get("generation"), page.get("parent_work_id")) for page in pages
    }
    status = {
        "g1_seed_work_count": sum(task.get("generation") == "G1_FORWARD" for task in plan.get("tasks", [])),
        "g2_component_work_count": sum(task.get("generation") == "G2_FORWARD" for task in plan.get("tasks", [])),
        "task_count": len(plan.get("tasks", [])),
        "forward_page_count": len(pages),
        "expected_forward_task_count": sum(task.get("direction") == "FORWARD" for task in plan.get("tasks", [])),
        "fetched_forward_task_count": len({(page.get("generation"), page.get("parent_work_id")) for page in pages}),
        "selected_forward_task_count": len(selected),
        "selected_fetched_forward_task_count": len(selected_keys & fetched_keys),
        "failed_page_count": sum(page.get("status") == "FAILED" for page in pages),
        "truncated_task_count": sum(page.get("truncated", False) for page in pages),
        "edge_count": len(edges),
        "record_count": len(records),
        "status": "PARTIAL" if (
            any(page.get("status") == "FAILED" or page.get("truncated", False) for page in pages)
            or len({(page.get("generation"), page.get("parent_work_id")) for page in pages}) < sum(task.get("direction") == "FORWARD" for task in plan.get("tasks", []))
        ) else "COMPLETE",
    }
    status_path = output_dir / "snowball_status.json"
    _write_json(status_path, status)
    return {"snowball_plan.json": plan_path, "snowball_pages.json": pages_path, "snowball_records.json": records_path, "citation_edges.csv": edges_path, "snowball_status.json": status_path}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_input_pages(input_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((input_dir / "retrieval_manifest.json").read_text(encoding="utf-8"))
    pages = []
    for page in manifest.get("pages", []):
        raw_path = input_dir / page["raw_payload_path"]
        payload = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else {}
        pages.append({**page, "raw_payload": payload})
    return pages


def _build_from_input(input_dir: Path, protocol_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pages = _load_input_pages(input_dir)
    seeds = protocol.get("retrieval", {}).get("g0_seeds", [])
    seed_ids = extract_seed_work_ids(pages, [seed.get("title", "") for seed in seeds])
    seed_by_id = {row["work_id"]: row for row in seed_ids}
    for page in pages:
        if page.get("index") != "OpenAlex":
            continue
        for item in page.get("raw_payload", {}).get("results", []):
            work_id = str(item.get("id", "")).removeprefix("https://openalex.org/")
            if work_id in seed_by_id:
                seed_by_id[work_id]["referenced_works"] = item.get("referenced_works", [])
    novelty = []
    novelty_path = input_dir / "novelty_evidence_matrix.csv"
    if novelty_path.is_file():
        with novelty_path.open(encoding="utf-8", newline="") as handle:
            novelty = list(csv.DictReader(handle))
    raw_records = json.loads((input_dir / "retrieval_records.json").read_text(encoding="utf-8"))
    openalex_by_doi: dict[str, str] = {}
    for record in raw_records:
        if record.get("source_index") == "OpenAlex" and record.get("source_identifier"):
            doi = _canonical_doi(record.get("doi"))
            if doi:
                openalex_by_doi[doi] = str(record["source_identifier"])
    records_path = input_dir / "records.csv"
    records = []
    if records_path.is_file():
        with records_path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    by_id = {row.get("record_id", ""): row for row in records}
    components: list[dict[str, Any]] = []
    for row in novelty:
        source = by_id.get(row.get("record_id", ""), {})
        work_id = str(source.get("source_identifier", ""))
        if not work_id.startswith("https://openalex.org/"):
            work_id = openalex_by_doi.get(_canonical_doi(source.get("doi")), "")
        if work_id:
            components.append({"record_id": row.get("record_id", ""), "title": row.get("title", ""), "c1_relevance": row.get("c1_relevance", ""), "openalex_work_id": work_id})
    return build_snowball_plan(list(seed_by_id.values()), components), pages


def run_self_test() -> dict[str, Any]:
    plan = build_snowball_plan([{"seed_title": "seed", "work_id": "W1", "referenced_works": ["W2"]}], [{"record_id": "r", "title": "c", "c1_relevance": "DIRECT", "openalex_work_id": "W3"}])
    checks = {"plan": len(plan["tasks"]) == 3, "url": "cites%3AW1" in build_openalex_cites_url("W1"), "parse": parse_forward_page({"meta": {"count": 0}, "results": []}, "W1", "G1_FORWARD", 1)["hit_count"] == 0}
    return {"passed": all(checks.values()), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=Path("docs/phase_x_x1_5_literature_audit_protocol.json"))
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--generation", choices=("G1_FORWARD", "G2_FORWARD"), default="")
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        result = run_self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    plan, pages = _build_from_input(args.input_dir.resolve(), args.protocol.resolve())
    selected_tasks = select_forward_tasks(plan["tasks"], generation=args.generation, offset=args.task_offset, limit=args.max_tasks)
    forward_pages = fetch_forward_tasks(selected_tasks, per_page=args.per_page, max_pages=args.max_pages) if args.fetch else []
    edges = []
    for task in plan["tasks"]:
        if task["generation"] == "G1_BACKWARD":
            edges.append({"generation": task["generation"], "direction": task["direction"], "parent_work_id": task["parent_work_id"], "child_work_id": task["work_id"], "seed_title": task["seed_title"]})
    for page in forward_pages:
        for record in page.get("records", []):
            edges.append({"generation": page["generation"], "direction": page["direction"], "parent_work_id": page["parent_work_id"], "child_work_id": str(record.get("source_identifier", "")).removeprefix("https://openalex.org/"), "seed_title": page.get("seed_title", "")})
    paths = write_snowball_outputs(args.output_dir.resolve(), plan, forward_pages, edges, selected_tasks=selected_tasks)
    print(json.dumps({key: str(value) for key, value in sorted(paths.items())}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
