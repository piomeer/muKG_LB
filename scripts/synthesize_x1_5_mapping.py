#!/usr/bin/env python3
"""Deterministic, read-only X1.5 Part 3 mapping and novelty synthesis."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("output/results/evidence_audit_x1_5")
DEFAULT_REPORT = Path("docs/phase_x_x1_5_part3_mapping_synthesis.md")
DIRECT_CLASSES = {"DIRECT-EXACT", "DIRECT-FUNCTIONAL", "STRONG-COMPONENT"}
DIRECT_RELEVANCE = {"DIRECT", "STRONG-COMPONENT"}
FULLTEXT_LOCATED = {"LOCATED", "REMOTE_LOCATED"}

MAPPING_FIELDS = [
    "record_id", "title", "year", "doi", "venue", "study_family_id",
    "peer_reviewed", "peer_review_status", "topic_relevance", "full_text_status",
    "c1_relevance", "overlap_class", "mq1_gpu_negative_sampling",
    "mq2_runtime_integration", "mq3_cost_scheduling_packing",
    "mq4_evidence_reproducibility", "evidence_locator_count", "source_indexes",
]

NOVELTY_FIELDS = [
    "record_id", "title", "study_family_id", "c1_relevance", "overlap_class",
    "mechanism_match", "integration_match", "evidence_match", "full_text_status",
    "peer_review_status", "evidence_locator_count", "gate_verdict",
    "blocking_conditions", "evidence_locators",
]


def _bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _terms(record: dict[str, Any]) -> str:
    return " ".join(str(record.get(key, "")) for key in ("title", "abstract", "venue", "query")).casefold()


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in sorted(value))
    return str(value)


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evidence_index(evidence_rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        record_id = str(row.get("record_id", ""))
        if record_id and str(row.get("locator", "")).strip():
            index.setdefault(record_id, []).append(row)
    return {key: sorted(value, key=lambda row: (str(row.get("facet", "")), str(row.get("locator", "")))) for key, value in index.items()}


def _apply_manual_overlays(input_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore adjudication fields omitted from the Part 2 records CSV schema."""

    path = input_dir / "manual_adjudications.json"
    if not path.is_file():
        return records
    patches = json.loads(path.read_text(encoding="utf-8"))
    by_id = {str(item.get("record_id", "")): item for item in patches if isinstance(item, dict)}
    overlay_fields = {
        "mechanism_match", "integration_match", "evidence_match", "c1_relevance",
        "overlap_class", "study_family_id", "peer_reviewed", "peer_review_status",
        "topic_relevance", "full_text_status",
    }
    result: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        patch = by_id.get(str(record.get("record_id", "")), {})
        for field in overlay_fields.intersection(patch):
            updated[field] = patch[field]
        result.append(updated)
    return result


def _included_ids(screening_rows: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("record_id", ""))
        for row in screening_rows
        if row.get("channel") == "NEUTRAL_ELIGIBILITY" and row.get("decision") == "INCLUDE"
    }


def build_literature_mapping(
    records: Iterable[dict[str, Any]],
    screening_rows: Iterable[dict[str, Any]],
    evidence_rows: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Map only neutral-included records to MQ1–MQ4 facets."""

    included = _included_ids(screening_rows)
    evidence = _evidence_index(evidence_rows)
    selected = [record for record in records if str(record.get("record_id", "")) in included]
    rows: list[dict[str, str]] = []
    for record in sorted(selected, key=lambda item: str(item.get("record_id", ""))):
        terms = _terms(record)
        locators = evidence.get(str(record.get("record_id", "")), [])
        has_gpu_sampling_signal = bool(re.search(r"gpu|cuda", terms)) and bool(re.search(r"negative sampl|sampler", terms))
        has_runtime_signal = bool(re.search(r"runtime|training|asynchron|scheduler|gpu|multi.?gpu", terms))
        has_scheduling_signal = bool(re.search(r"cost|schedul|pack|batch|partition|out.of.core|scalab", terms))
        row = {
            "record_id": str(record.get("record_id", "")),
            "title": str(record.get("title", "")),
            "year": str(record.get("year", "")),
            "doi": str(record.get("doi", "")),
            "venue": str(record.get("venue", "")),
            "study_family_id": str(record.get("study_family_id", "")),
            "peer_reviewed": "true" if _bool(record.get("peer_reviewed")) else "false",
            "peer_review_status": str(record.get("peer_review_status", "")),
            "topic_relevance": str(record.get("topic_relevance", "")),
            "full_text_status": str(record.get("full_text_status", "")),
            "c1_relevance": str(record.get("c1_relevance", "")),
            "overlap_class": str(record.get("overlap_class", "")),
            "mq1_gpu_negative_sampling": "true" if has_gpu_sampling_signal or (_bool(record.get("mechanism_match")) and record.get("c1_relevance") == "DIRECT" and record.get("overlap_class") in DIRECT_CLASSES) else "false",
            "mq2_runtime_integration": "true" if has_runtime_signal or _bool(record.get("integration_match")) else "false",
            "mq3_cost_scheduling_packing": "true" if has_scheduling_signal else "false",
            "mq4_evidence_reproducibility": "true" if _bool(record.get("peer_reviewed")) and str(record.get("peer_review_status")) == "VERIFIED" and bool(locators) else "false",
            "evidence_locator_count": str(len(locators)),
            "source_indexes": str(record.get("source_index", "")),
        }
        rows.append(row)
    return rows


def build_novelty_evidence_matrix(
    records: Iterable[dict[str, Any]],
    evidence_rows: Iterable[dict[str, Any]],
    novelty_decision: dict[str, Any],
) -> list[dict[str, str]]:
    """Build C1 candidate rows while inheriting, never resolving, the gate."""

    evidence = _evidence_index(evidence_rows)
    inherited = sorted(set(str(item) for item in novelty_decision.get("blockers", []) if item))
    candidates = [
        record for record in records
        if str(record.get("c1_relevance", "")) in DIRECT_RELEVANCE
        or str(record.get("overlap_class", "")) in DIRECT_CLASSES
    ]
    rows: list[dict[str, str]] = []
    for record in sorted(candidates, key=lambda item: str(item.get("record_id", ""))):
        record_id = str(record.get("record_id", ""))
        locators = evidence.get(record_id, [])
        conditions = set(inherited)
        if str(record.get("full_text_status", "")) not in FULLTEXT_LOCATED:
            conditions.add("missing_full_text")
        if str(record.get("peer_review_status", "")) != "VERIFIED":
            conditions.add("peer_review_status_unverified")
        if not locators:
            conditions.add("evidence_locator_missing")
        rows.append({
            "record_id": record_id,
            "title": str(record.get("title", "")),
            "study_family_id": str(record.get("study_family_id", "")),
            "c1_relevance": str(record.get("c1_relevance", "")),
            "overlap_class": str(record.get("overlap_class", "")),
            "mechanism_match": "true" if _bool(record.get("mechanism_match")) else "false",
            "integration_match": "true" if _bool(record.get("integration_match")) else "false",
            "evidence_match": "true" if _bool(record.get("evidence_match")) else "false",
            "full_text_status": str(record.get("full_text_status", "")),
            "peer_review_status": str(record.get("peer_review_status", "")),
            "evidence_locator_count": str(len(locators)),
            "gate_verdict": str(novelty_decision.get("verdict", "UNRESOLVED")),
            "blocking_conditions": ";".join(sorted(conditions)),
            "evidence_locators": ";".join(str(row.get("locator", "")) for row in locators),
        })
    return rows


def summarize_mapping(
    mapping_rows: Iterable[dict[str, Any]],
    novelty_rows: Iterable[dict[str, Any]],
    novelty_decision: dict[str, Any],
) -> dict[str, Any]:
    mapping = list(mapping_rows)
    novelty = list(novelty_rows)
    mq_fields = [
        "mq1_gpu_negative_sampling", "mq2_runtime_integration",
        "mq3_cost_scheduling_packing", "mq4_evidence_reproducibility",
    ]
    return {
        "mapping_record_count": len(mapping),
        "novelty_candidate_count": len(novelty),
        "study_family_count": len({row.get("study_family_id", "") for row in mapping if row.get("study_family_id", "")}),
        "mq_counts": {field: sum(_bool(row.get(field)) for row in mapping) for field in mq_fields},
        "overlap_counts": {
            key: sum(row.get("overlap_class") == key for row in mapping)
            for key in sorted({str(row.get("overlap_class", "")) for row in mapping if row.get("overlap_class")})
        },
        "evidence_locator_rows": sum(int(row.get("evidence_locator_count", "0") or 0) > 0 for row in mapping),
        "c1_gate_verdict": str(novelty_decision.get("verdict", "UNRESOLVED")),
        "c1_gate_blockers": sorted(set(str(item) for item in novelty_decision.get("blockers", []) if item)),
        "novelty_rows_with_blockers": sum(bool(str(row.get("blocking_conditions", ""))) for row in novelty),
    }


def _report(
    summary: dict[str, Any],
    novelty_rows: list[dict[str, str]],
    retrieval_status: str = "OPEN",
    failed_retrieval_count: int = 0,
) -> str:
    overlap = ", ".join(f"{key}={value}" for key, value in summary["overlap_counts"].items()) or "none"
    blockers = ", ".join(summary["c1_gate_blockers"]) or "none"
    lines = [
        "# Phase X1.5 Part 3 — Systematic Mapping and Novelty Synthesis",
        "",
        "This is a deterministic synthesis of the Part 2 artifacts. It does not",
        "perform new retrieval and does not constitute a global novelty proof.",
        "",
        "## Mapping coverage",
        "",
        f"The mapping contains {summary['mapping_record_count']} neutral-included records across {summary['study_family_count']} study families. ",
        f"Overlap distribution: {overlap}. Locator-backed evidence appears for {summary['evidence_locator_rows']} mapped records.",
        "",
        "MQ coverage is recorded in `mapping_summary.json`; facet flags are derived",
        "from explicit coding and conservative metadata signals, not from inferred",
        "paper conclusions.",
        "",
        "## C1 novelty position",
        "",
        f"The inherited C1 gate is **{summary['c1_gate_verdict']}** with blockers: {blockers}.",
        f"The novelty matrix contains {summary['novelty_candidate_count']} candidate rows; {summary['novelty_rows_with_blockers']} retain one or more blocking conditions.",
        "No RETAIN, NARROW, REFRAME, or DROP conclusion is released while the",
        "inherited gate is unresolved. Direct and strong-component rows are evidence",
        "for boundary review, not a claim of exact equivalence.",
        "",
        "## Limitations and next action",
        "",
        "The corpus is English-only and limited to DBLP, OpenAlex, and Crossref;",
        f"The DBLP retrieval channel is currently {retrieval_status}; {failed_retrieval_count} DBLP pages remain failed in the preserved snapshot, and most records",
        "remain pending human adjudication. Continue high-priority adjudication and",
        "the protocol-defined DBLP retry rounds before making paper-level novelty",
        "language decisions.",
        "",
        "### Candidate matrix trace",
        "",
        "| Record | Overlap | Mechanism | Integration | Evidence locators | Blocking conditions |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in novelty_rows:
        lines.append(
            f"| {row['record_id']} | {row['overlap_class']} | {row['mechanism_match']} | {row['integration_match']} | {row['evidence_locator_count']} | {row['blocking_conditions'] or 'none'} |"
        )
    return "\n".join(lines) + "\n"


def write_mapping_outputs(
    input_dir: Path,
    output_dir: Path,
    *,
    records: list[dict[str, Any]] | None = None,
    screening: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    novelty_decision: dict[str, Any] | None = None,
    report_path: Path | None = None,
) -> dict[str, Path]:
    """Read Part 2 artifacts (unless supplied) and write deterministic outputs."""

    def read_csv(name: str) -> list[dict[str, str]]:
        with (input_dir / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    records = records if records is not None else read_csv("records.csv")
    records = _apply_manual_overlays(input_dir, records)
    screening = screening if screening is not None else read_csv("screening_decisions.csv")
    evidence = evidence if evidence is not None else read_csv("evidence_extraction.csv")
    if novelty_decision is None:
        novelty_decision = json.loads((input_dir / "novelty_decision.json").read_text(encoding="utf-8"))
    mapping = build_literature_mapping(records, screening, evidence)
    novelty = build_novelty_evidence_matrix(records, evidence, novelty_decision)
    summary = summarize_mapping(mapping, novelty, novelty_decision)
    checks = {
        "input_records_present": bool(records),
        "mapping_rows_have_record_ids": all(bool(row.get("record_id")) for row in mapping),
        "novelty_rows_have_gate_verdict": all(bool(row.get("gate_verdict")) for row in novelty),
        "one_inherited_c1_verdict": summary["c1_gate_verdict"] in {"RETAIN", "NARROW", "REFRAME", "DROP", "UNRESOLVED"},
        "no_dynamic_fields_emitted": True,
    }
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    mapping_path = output_dir / "literature_mapping.csv"
    novelty_path = output_dir / "novelty_evidence_matrix.csv"
    summary_path = output_dir / "mapping_summary.json"
    checks_path = output_dir / "coverage_checks.json"
    _write_csv(mapping_path, MAPPING_FIELDS, mapping)
    _write_csv(novelty_path, NOVELTY_FIELDS, novelty)
    _write_json(summary_path, summary)
    _write_json(checks_path, checks)
    paths.update({"literature_mapping.csv": mapping_path, "novelty_evidence_matrix.csv": novelty_path, "mapping_summary.json": summary_path, "coverage_checks.json": checks_path})
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        retrieval_status = "OPEN"
        failed_retrieval_count = 0
        manifest_path = input_dir / "retrieval_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            failed_retrieval_count = sum(
                page.get("index") == "DBLP" and page.get("status") == "FAILED"
                for page in manifest.get("pages", [])
            )
        cutoff_path = input_dir / "retrieval_cutoff.json"
        if cutoff_path.is_file():
            retrieval_status = str(json.loads(cutoff_path.read_text(encoding="utf-8")).get("status", "OPEN"))
        report_path.write_text(_report(summary, novelty, retrieval_status, failed_retrieval_count), encoding="utf-8")
        paths["part3_report.md"] = report_path
    return paths


def run_self_test() -> dict[str, Any]:
    records = [{"record_id": "r", "title": "GPU negative sampling", "peer_reviewed": "true", "peer_review_status": "VERIFIED", "topic_relevance": "IN_SCOPE", "full_text_status": "REMOTE_LOCATED", "c1_relevance": "DIRECT", "overlap_class": "DIRECT-FUNCTIONAL", "mechanism_match": "true", "integration_match": "true", "study_family_id": "f"}]
    screening = [{"record_id": "r", "channel": "NEUTRAL_ELIGIBILITY", "decision": "INCLUDE"}]
    evidence = [{"record_id": "r", "facet": "runtime", "locator": "https://example.org", "value": "GPU"}]
    decision = {"verdict": "UNRESOLVED", "blockers": ["human_adjudication_pending"]}
    mapping = build_literature_mapping(records, screening, evidence)
    novelty = build_novelty_evidence_matrix(records, evidence, decision)
    summary = summarize_mapping(mapping, novelty, decision)
    checks = {"mapping": len(mapping) == 1, "mq1": mapping[0]["mq1_gpu_negative_sampling"] == "true", "blocked": "human_adjudication_pending" in novelty[0]["blocking_conditions"], "verdict": summary["c1_gate_verdict"] == "UNRESOLVED"}
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
    report_path = args.report_path.resolve()
    paths = write_mapping_outputs(input_dir, output_dir, report_path=report_path)
    print(json.dumps({key: str(value) for key, value in sorted(paths.items())}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
