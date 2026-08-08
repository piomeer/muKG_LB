#!/usr/bin/env python3
"""Deterministic, read-only X1.5 Part 4 C1 closure audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.audit_x1_5_literature import assess_dblp_retry_closure, qualify_fallback_coverage
except ModuleNotFoundError:  # direct ``python scripts/close_x1_5_c1_gate.py`` invocation
    from audit_x1_5_literature import assess_dblp_retry_closure, qualify_fallback_coverage


DEFAULT_INPUT = Path("output/results/evidence_audit_x1_5")
DEFAULT_REPORT = Path("docs/phase_x_x1_5_part4_c1_gate_closure.md")
FULLTEXT_LOCATED = {"LOCATED", "REMOTE_LOCATED"}
PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "": 3}

QUEUE_FIELDS = [
    "record_id", "issue_type", "priority", "status", "decision", "adjudicator_note",
]
STATUS_FIELDS = [
    "record_id", "title", "study_family_id", "c1_relevance", "overlap_class",
    "peer_review_status", "full_text_status", "evidence_locator_count",
    "human_status", "retrieval_status", "blocking_conditions",
]


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else str(row.get(field, "")) for field in fields})


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_adjudication_queue(human_rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Retain unresolved human rows and sort by frozen priority."""

    rows = [
        {field: str(row.get(field, "")) for field in QUEUE_FIELDS}
        for row in human_rows
        if str(row.get("status", "")).upper() not in {"RESOLVED", "EXCLUDED", "RETRACTED"}
    ]
    return sorted(rows, key=lambda row: (PRIORITY_ORDER.get(row["priority"].upper(), 3), row["record_id"], row["issue_type"]))


def _human_status_by_record(human_rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    statuses: dict[str, list[str]] = {}
    for row in human_rows:
        record_id = str(row.get("record_id", ""))
        if record_id:
            statuses.setdefault(record_id, []).append(str(row.get("status", "")).upper())
    result: dict[str, str] = {}
    for record_id, values in statuses.items():
        result[record_id] = "PENDING" if any(value not in {"RESOLVED", "EXCLUDED", "RETRACTED"} for value in values) else "RESOLVED"
    return result


def build_source_verification_status(
    novelty_rows: Iterable[dict[str, Any]],
    human_rows: Iterable[dict[str, Any]],
    retrieval_pages: Iterable[dict[str, Any]],
    retrieval_closure: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build one auditable status row per C1/strong-component candidate."""

    human_status = _human_status_by_record(human_rows)
    pages = list(retrieval_pages)
    retrieval_failed = any(page.get("status") == "FAILED" for page in pages)
    closure_status = str((retrieval_closure or {}).get("status", ""))
    retrieval_status = closure_status or ("FAILED" if retrieval_failed else "OK")
    retrieval_conditions = set((retrieval_closure or {}).get("blocking_conditions", []))
    if retrieval_closure is None and retrieval_failed:
        retrieval_conditions.add("retrieval_failed")
    rows: list[dict[str, str]] = []
    for source in sorted(novelty_rows, key=lambda row: str(row.get("record_id", ""))):
        record_id = str(source.get("record_id", ""))
        global_conditions = {
            "human_adjudication_pending", "peer_review_status_unverified",
            "retrieval_failed", "retrieval_channel_open", "retrieval_gap_uncovered",
            "UNRESOLVED_MISSING_PAGE",
        }
        conditions = {
            item for item in str(source.get("blocking_conditions", "")).split(";")
            if item and item not in global_conditions
        }
        if human_status.get(record_id) == "PENDING":
            conditions.add("human_adjudication_pending")
        conditions.update(retrieval_conditions)
        if str(source.get("peer_review_status", "")) != "VERIFIED":
            conditions.add("peer_review_status_unverified")
        if str(source.get("full_text_status", "")) not in FULLTEXT_LOCATED:
            conditions.add("missing_full_text")
        if int(str(source.get("evidence_locator_count", "0") or 0)) == 0:
            conditions.add("evidence_locator_missing")
        rows.append({
            "record_id": record_id,
            "title": str(source.get("title", "")),
            "study_family_id": str(source.get("study_family_id", "")),
            "c1_relevance": str(source.get("c1_relevance", "")),
            "overlap_class": str(source.get("overlap_class", "")),
            "peer_review_status": str(source.get("peer_review_status", "")),
            "full_text_status": str(source.get("full_text_status", "")),
            "evidence_locator_count": str(source.get("evidence_locator_count", "0")),
            "human_status": human_status.get(record_id, "NOT_QUEUED"),
            "retrieval_status": retrieval_status,
            "blocking_conditions": ";".join(sorted(conditions)),
        })
    return rows


def assess_gate_closure(
    novelty_decision: dict[str, Any],
    queue: Iterable[dict[str, Any]],
    status_rows: Iterable[dict[str, Any]],
    retrieval_pages: Iterable[dict[str, Any]],
    cutoff: dict[str, Any],
    retrieval_closure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return closure readiness while preserving the inherited C1 verdict."""

    blockers = {str(item) for item in novelty_decision.get("blockers", []) if item}
    retrieval_blockers = {"retrieval_failed", "retrieval_channel_open", "retrieval_gap_uncovered", "UNRESOLVED_MISSING_PAGE"}
    if retrieval_closure is not None:
        # Recompute retrieval conditions from the closure ledger; stale Part 2/3
        # inherited blockers must not survive a qualified fallback closure.
        blockers.difference_update(retrieval_blockers)
        blockers.update(str(item) for item in retrieval_closure.get("blocking_conditions", []) if item)
    queue_rows = list(queue)
    status = list(status_rows)
    pages = list(retrieval_pages)
    if queue_rows:
        blockers.add("human_adjudication_pending")
    if retrieval_closure is None:
        if any(page.get("status") == "FAILED" for page in pages):
            blockers.add("retrieval_failed")
        if str(cutoff.get("status", "")) not in {"CLOSED", "COMPLETE"}:
            blockers.add("retrieval_channel_open")
    for row in status:
        blockers.update(item for item in str(row.get("blocking_conditions", "")).split(";") if item)
    return {
        "claim_id": "C1",
        "closure_status": "UNRESOLVED" if blockers else "READY_FOR_HUMAN_DECISION",
        "c1_verdict": str(novelty_decision.get("verdict", "UNRESOLVED")),
        "blockers": sorted(blockers),
        "pending_adjudication_count": len(queue_rows),
        "candidate_count": len(status),
        "failed_retrieval_page_count": sum(page.get("status") == "FAILED" for page in pages),
        "retrieval_status": str((retrieval_closure or {}).get("status", cutoff.get("status", "OPEN"))),
        "note": "Closure readiness is mechanical; it never issues a substantive novelty verdict.",
    }


def _report(closure: dict[str, Any], queue: list[dict[str, str]], status_rows: list[dict[str, str]]) -> str:
    blockers = ", ".join(closure["blockers"]) or "none"
    lines = [
        "# Phase X1.5 Part 4 — C1 Gate Closure Audit",
        "",
        "This report is a deterministic, read-only closure check. It does not",
        "replace human adjudication and does not declare global novelty.",
        "",
        "## Gate state",
        "",
        f"Closure status: **{closure['closure_status']}**; inherited C1 verdict: **{closure['c1_verdict']}**.",
        f"Blockers: {blockers}.",
        f"Pending adjudication rows: {closure['pending_adjudication_count']}; candidate rows: {closure['candidate_count']}; failed retrieval pages: {closure['failed_retrieval_page_count']}.",
        "",
        "The gate remains fail-closed until all protocol conditions are satisfied.",
        "A READY_FOR_HUMAN_DECISION state would only permit a final human novelty",
        "decision; it would not select RETAIN, NARROW, REFRAME, or DROP automatically.",
        "",
        "## Queue summary",
        "",
        f"The unresolved queue contains {len(queue)} rows, ordered HIGH before MEDIUM and then by stable record ID.",
        "",
        "## Candidate status",
        "",
        "| Record | Human | Retrieval | Peer review | Full text | Evidence locators | Blockers |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in status_rows:
        lines.append(
            f"| {row['record_id']} | {row['human_status']} | {row['retrieval_status']} | {row['peer_review_status']} | {row['full_text_status']} | {row['evidence_locator_count']} | {row['blocking_conditions'] or 'none'} |"
        )
    lines.extend([
        "",
        "## Required next actions",
        "",
        "1. Resolve the HIGH-priority human queue with locator-backed evidence.",
        "2. Run the protocol-defined DBLP retry batches and preserve any failures.",
        "3. Re-run this closure audit and then Part 3 mapping after each accepted batch.",
    ])
    return "\n".join(lines) + "\n"


def write_closure_outputs(
    input_dir: Path,
    output_dir: Path,
    *,
    novelty: list[dict[str, Any]] | None = None,
    human: list[dict[str, Any]] | None = None,
    pages: list[dict[str, Any]] | None = None,
    novelty_decision: dict[str, Any] | None = None,
    cutoff: dict[str, Any] | None = None,
    report_path: Path | None = None,
) -> dict[str, Path]:
    def read_csv(name: str) -> list[dict[str, str]]:
        with (input_dir / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    novelty = novelty if novelty is not None else read_csv("novelty_evidence_matrix.csv")
    human = human if human is not None else read_csv("human_adjudication.csv")
    if pages is None:
        manifest = json.loads((input_dir / "retrieval_manifest.json").read_text(encoding="utf-8"))
        pages = manifest.get("pages", [])
    if novelty_decision is None:
        novelty_decision = json.loads((input_dir / "novelty_decision.json").read_text(encoding="utf-8"))
    if cutoff is None:
        cutoff_path = input_dir / "retrieval_cutoff.json"
        cutoff = json.loads(cutoff_path.read_text(encoding="utf-8")) if cutoff_path.is_file() else {"status": "OPEN"}
    fallback_path = input_dir / "fallback_coverage.csv"
    if fallback_path.is_file():
        with fallback_path.open(encoding="utf-8", newline="") as handle:
            fallback_rows = list(csv.DictReader(handle))
    else:
        fallback_rows = qualify_fallback_coverage(pages)
    retrieval_closure = assess_dblp_retry_closure(cutoff, fallback_rows)
    queue = build_adjudication_queue(human)
    status_rows = build_source_verification_status(novelty, human, pages, retrieval_closure)
    closure = assess_gate_closure(novelty_decision, queue, status_rows, pages, cutoff, retrieval_closure)
    closure["retrieval_status"] = retrieval_closure["status"]
    closure["retrieval_advisories"] = retrieval_closure.get("advisories", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    queue_path = output_dir / "c1_adjudication_queue.csv"
    status_path = output_dir / "c1_source_verification_status.csv"
    closure_path = output_dir / "c1_gate_closure.json"
    _write_csv(queue_path, QUEUE_FIELDS, queue)
    _write_csv(status_path, STATUS_FIELDS, status_rows)
    _write_json(closure_path, closure)
    paths.update({"c1_adjudication_queue.csv": queue_path, "c1_source_verification_status.csv": status_path, "c1_gate_closure.json": closure_path})
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_report(closure, queue, status_rows), encoding="utf-8")
        paths["part4_report.md"] = report_path
    return paths


def run_self_test() -> dict[str, Any]:
    novelty = [{"record_id": "r", "title": "GPU sampler", "study_family_id": "f", "c1_relevance": "DIRECT", "overlap_class": "DIRECT-FUNCTIONAL", "peer_review_status": "VERIFIED", "full_text_status": "REMOTE_LOCATED", "evidence_locator_count": "1", "blocking_conditions": ""}]
    human = [{"record_id": "r", "issue_type": "POTENTIAL_C1", "priority": "HIGH", "status": "PENDING"}]
    pages = [{"index": "DBLP", "status": "FAILED"}]
    queue = build_adjudication_queue(human)
    status = build_source_verification_status(novelty, human, pages)
    closure = assess_gate_closure({"verdict": "UNRESOLVED", "blockers": []}, queue, status, pages, {"status": "OPEN"})
    checks = {"priority": queue[0]["priority"] == "HIGH", "retrieval_blocked": "retrieval_failed" in closure["blockers"], "fail_closed": closure["closure_status"] == "UNRESOLVED"}
    return {"passed": all(checks.values()), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        result = run_self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    paths = write_closure_outputs(input_dir, output_dir, report_path=args.report_path.resolve())
    print(json.dumps({key: str(value) for key, value in sorted(paths.items())}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
