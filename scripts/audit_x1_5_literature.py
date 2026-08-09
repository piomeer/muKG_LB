#!/usr/bin/env python3
"""Deterministic first-stage X1.5 literature-audit pipeline.

The first implementation stage is deliberately offline.  It validates the
frozen protocol, ingests a local raw metadata snapshot when supplied, and
derives normalized records, screening decisions, and the C1 novelty gate.  It
does not contact indexes, download papers, run training, or access a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


PROTOCOL_RELATIVE = "docs/phase_x_x1_5_literature_audit_protocol.json"
DEFAULT_OUTPUT_RELATIVE = "output/results/evidence_audit_x1_5"
RETRY_STATE_FILENAME = "dblp_retry_state.json"
RETRY_BATCH_INTERVAL_SECONDS = 600
ALLOWED_DATABASES = ("DBLP", "OpenAlex", "Crossref")
OVERLAP_CLASSES = (
    "DIRECT-EXACT",
    "DIRECT-FUNCTIONAL",
    "STRONG-COMPONENT",
    "ADJACENT-SYSTEM",
    "SEMANTIC-BACKGROUND",
    "NO-OVERLAP",
)
VERDICTS = ("RETAIN", "NARROW", "REFRAME", "DROP", "UNRESOLVED")
CORE_TOPIC_TERMS = (
    "knowledge graph",
    "knowledge representation learning",
    "knowledge graph embedding",
    " kge ",
    "transe",
    "rotate",
    "entity embedding",
    "relation embedding",
    "link prediction",
    "openke",
    "pykeen",
    "dgl-ke",
    "libkge",
    "pykg2vec",
)
ADJACENT_TOPIC_TERMS = (
    "negative sampling",
    "negative sample",
    "graph embedding",
    "gpu",
    "cuda",
    "sampler",
    "batch",
    "scheduling",
    "runtime",
    "reproducib",
    "framework",
    "artifact",
)

CSV_SCHEMAS: dict[str, list[str]] = {
    "records.csv": [
        "record_id",
        "dedup_key",
        "title",
        "year",
        "doi",
        "dblp_key",
        "venue",
        "source_index",
        "source_identifier",
        "query",
        "retrieval_stage",
        "language",
        "peer_reviewed",
        "peer_review_status",
        "topic_relevant",
        "topic_relevance",
        "full_text_status",
        "c1_relevance",
        "c1_potential",
        "overlap_class",
        "study_family_id",
    ],
    "screening_decisions.csv": [
        "record_id",
        "channel",
        "decision",
        "reason_codes",
        "notes",
    ],
    "human_adjudication.csv": [
        "record_id",
        "issue_type",
        "priority",
        "status",
        "decision",
        "adjudicator_note",
    ],
    "fulltext_manifest.csv": [
        "record_id",
        "status",
        "path",
        "sha256",
        "locator_scope",
    ],
    "evidence_extraction.csv": [
        "record_id",
        "facet",
        "value",
        "locator",
        "source_type",
    ],
    "novelty_matrix.csv": [
        "record_id",
        "overlap_class",
        "c1_relevance",
        "c1_potential",
        "mechanism_match",
        "integration_match",
        "evidence_match",
        "full_text_status",
        "notes",
    ],
    "fallback_coverage.csv": [
        "failed_index",
        "query",
        "retrieval_stage",
        "fallback_status",
        "fallback_indexes",
        "qualification",
        "matched_record_ids",
        "raw_evidence",
        "disposition",
    ],
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_doi(value: Any) -> str:
    if value is None:
        return ""
    doi = str(value).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.rstrip(" .;,")


def _normalize_title(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_language(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    return "English" if text in {"en", "eng", "english"} else str(value).strip()


def _as_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_id(title: str, year: int | None, doi: str, dblp_key: str, source: str) -> str:
    seed = "|".join((title, str(year or ""), doi, dblp_key, source))
    return "rec-" + _sha256_text(seed)[:16]


def _dedup_key(record: dict[str, Any]) -> tuple[str, str]:
    if record["doi"]:
        return "DOI", "doi:" + record["doi"]
    if record["dblp_key"]:
        return "DBLP_KEY", "dblp:" + record["dblp_key"].casefold()
    return "NORMALIZED_TITLE_YEAR", f"title:{record['title_normalized']}:{record['year'] or ''}"


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one index row without adding nondeterministic identifiers."""

    title = re.sub(r"\s+", " ", str(raw.get("title", "")).strip())
    year = _as_int(raw.get("year"))
    doi = _canonical_doi(raw.get("doi"))
    dblp_key = str(raw.get("dblp_key", "")).strip()
    source_index = str(raw.get("source_index", "")).strip()
    title_normalized = _normalize_title(title)
    record_id = str(raw.get("record_id") or _record_id(title_normalized, year, doi, dblp_key, source_index))
    result: dict[str, Any] = {
        "record_id": record_id,
        "title": title,
        "title_normalized": title_normalized,
        "year": year,
        "doi": doi,
        "dblp_key": dblp_key,
        "venue": str(raw.get("venue", "")).strip(),
        "source_index": source_index,
        "source_identifier": str(raw.get("source_identifier", "")).strip(),
        "query": str(raw.get("query", "")).strip(),
        "retrieval_stage": str(raw.get("retrieval_stage", "WIDE_SENTINEL")).strip(),
        "language": _normalize_language(raw.get("language")),
        "peer_reviewed": _as_bool(raw.get("peer_reviewed")),
        "peer_review_status": str(raw.get("peer_review_status", "UNVERIFIED")).strip().upper(),
        "topic_relevant": _as_bool(raw.get("topic_relevant")),
        "topic_relevance": str(raw.get("topic_relevance", "")).strip().upper(),
        "full_text_status": str(raw.get("full_text_status", "UNKNOWN")).strip().upper(),
        "full_text_path": str(raw.get("full_text_path", "")).strip(),
        "full_text_url": str(raw.get("full_text_url", "")).strip(),
        "full_text_sha256": str(raw.get("full_text_sha256", "")).strip().lower(),
        "c1_relevance": str(raw.get("c1_relevance", "NONE")).strip().upper(),
        "c1_potential": str(raw.get("c1_potential", "")).strip().upper(),
        "overlap_class": str(raw.get("overlap_class", "NO-OVERLAP")).strip().upper(),
        "mechanism_match": bool(_as_bool(raw.get("mechanism_match")) or False),
        "integration_match": bool(_as_bool(raw.get("integration_match")) or False),
        "evidence_match": bool(_as_bool(raw.get("evidence_match")) or False),
        "study_family_id": str(raw.get("study_family_id", "")).strip(),
        "abstract": str(raw.get("abstract", "")).strip(),
        "evidence": raw.get("evidence", {}) if isinstance(raw.get("evidence", {}), dict) else {},
        "manual_adjudicated": bool(_as_bool(raw.get("manual_adjudicated")) or False),
    }
    kind, key = _dedup_key(result)
    result["dedup_key"] = key
    result["dedup_basis"] = kind
    if result["overlap_class"] not in OVERLAP_CLASSES:
        result["overlap_class"] = "NO-OVERLAP"
    if result["topic_relevance"] not in {"IN_SCOPE", "ADJACENT", "IRRELEVANT", "UNKNOWN"}:
        result["topic_relevance"] = classify_topic_relevance(result)
    if result["c1_potential"] not in {"POTENTIAL_DIRECT", "POTENTIAL_COMPONENT", "NONE"}:
        result["c1_potential"] = classify_c1_potential(result)
    return result


def classify_topic_relevance(record: dict[str, Any]) -> str:
    """Conservatively remove only metadata with no KGE/runtime signal."""

    text = " ".join((str(record.get("title", "")), str(record.get("abstract", "")))).casefold()
    if not text.strip():
        return "UNKNOWN"
    if any(term in f" {text} " for term in CORE_TOPIC_TERMS):
        return "IN_SCOPE"
    if any(term in text for term in ADJACENT_TOPIC_TERMS):
        return "ADJACENT"
    return "IRRELEVANT"


def classify_c1_potential(record: dict[str, Any]) -> str:
    """Flag likely C1 candidates for priority review without assigning C1 overlap."""

    text = " ".join((str(record.get("title", "")), str(record.get("abstract", "")))).casefold()
    core = any(term in f" {text} " for term in CORE_TOPIC_TERMS)
    sampling_or_gpu = any(term in text for term in ("negative sampling", "negative sample", "gpu", "cuda", "sampler"))
    runtime = any(term in text for term in ("runtime", "throughput", "scheduler", "batch", "framework"))
    if core and sampling_or_gpu:
        return "POTENTIAL_DIRECT"
    if core and runtime:
        return "POTENTIAL_COMPONENT"
    return "NONE"


def _completeness(record: dict[str, Any]) -> int:
    fields = ("title", "year", "doi", "dblp_key", "venue", "abstract", "full_text_path")
    return sum(bool(record.get(field)) for field in fields)


def deduplicate_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Deduplicate with DOI > DBLP key > normalized title/year precedence."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["dedup_key"], []).append(record)
    unique: list[dict[str, Any]] = []
    decisions: list[dict[str, str]] = []
    for key in sorted(groups):
        members = sorted(
            groups[key],
            key=lambda item: (-_completeness(item), item["record_id"]),
        )
        retained = members[0]
        unique.append(retained)
        if len(members) > 1:
            for duplicate in members[1:]:
                decisions.append(
                    {
                        "duplicate_record_id": duplicate["record_id"],
                        "retained_record_id": retained["record_id"],
                        "dedup_key": key,
                        "reason": retained["dedup_basis"],
                    }
                )
    return sorted(unique, key=lambda item: (item["dedup_key"], item["record_id"])), decisions


def load_protocol(repo_root: Path) -> dict[str, Any]:
    path = repo_root / PROTOCOL_RELATIVE
    protocol = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("invalid X1.5 protocol: " + "; ".join(errors))
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("protocol_id") != "phase-x-x1-5-literature-audit-v1":
        errors.append("protocol_id")
    if protocol.get("publication_window") != [2013, "2026-08-07"]:
        errors.append("publication_window")
    if protocol.get("language_scope") != ["English"]:
        errors.append("language_scope")
    if protocol.get("databases") != list(ALLOWED_DATABASES):
        errors.append("databases")
    if protocol.get("hard_novelty_claim") != "C1":
        errors.append("hard_novelty_claim")
    if protocol.get("retrieval", {}).get("max_snowball_generation") != 2:
        errors.append("max_snowball_generation")
    if protocol.get("confidence_ceiling") != "MODERATE":
        errors.append("confidence_ceiling")
    if len(protocol.get("retrieval", {}).get("g0_seeds", [])) < 10:
        errors.append("seed_count")
    return errors


def screen_record(record: dict[str, Any], channel: str = "NEUTRAL_ELIGIBILITY") -> dict[str, Any]:
    """Apply fixed eligibility rules and E01-E06 reason codes."""

    reasons: list[str] = []
    uncertain = False
    auto_excluded = False
    year = record.get("year")
    if year is None or not record.get("title") or not record.get("source_index"):
        reasons.append("E06")
        uncertain = True
    elif year < 2013 or year > 2026:
        reasons.append("E01")
    if record.get("peer_reviewed") is False:
        reasons.append("E02")
    elif record.get("peer_reviewed") is None:
        reasons.append("E06")
        uncertain = True
    if record.get("language") and record.get("language") != "English":
        reasons.append("E03")
    elif not record.get("language"):
        reasons.append("E06")
        uncertain = True
    if record.get("source_index") not in ALLOWED_DATABASES:
        reasons.append("E04")
    topic_relevance = record.get("topic_relevance") or classify_topic_relevance(record)
    if topic_relevance == "IRRELEVANT" or record.get("topic_relevant") is False:
        reasons.append("E04")
        auto_excluded = topic_relevance == "IRRELEVANT"
    notes = "" if not reasons else "Eligibility rule failure"
    if auto_excluded:
        notes = "AUTO_EXCLUDE_OBVIOUSLY_IRRELEVANT"
    return {
        "record_id": record["record_id"],
        "channel": channel,
        "decision": "EXCLUDE" if any(code in reasons for code in ("E01", "E02", "E03", "E04")) else ("UNCERTAIN" if uncertain else "INCLUDE"),
        "reason_codes": ";".join(sorted(set(reasons))),
        "notes": notes,
    }


def _is_direct_candidate(record: dict[str, Any]) -> bool:
    return record.get("c1_relevance") == "DIRECT" or record.get("overlap_class") in {
        "DIRECT-EXACT",
        "DIRECT-FUNCTIONAL",
    }


def adversarial_screen_record(record: dict[str, Any]) -> dict[str, Any]:
    """Look specifically for C1 prior-art risk and route uncertainty to humans."""

    neutral = screen_record(record, channel="ADVERSARIAL_PRIOR_ART")
    if neutral["decision"] == "EXCLUDE":
        return neutral
    if _is_direct_candidate(record) and record.get("full_text_status") not in {"LOCATED", "REMOTE_LOCATED"}:
        neutral.update(
            {
                "decision": "UNCERTAIN",
                "reason_codes": "A01",
                "notes": "Direct C1 candidate requires full-text adjudication.",
            }
        )
    return neutral


def run_dual_screening(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Run neutral and adversarial channels and emit a human conflict queue."""

    rows = sorted(records, key=lambda item: item["record_id"])
    decisions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    adjudication: list[dict[str, Any]] = []
    for record in rows:
        neutral = screen_record(record, channel="NEUTRAL_ELIGIBILITY")
        adversarial = adversarial_screen_record(record)
        decisions.extend((neutral, adversarial))
        if neutral["decision"] != adversarial["decision"]:
            conflicts.append(
                {
                    "record_id": record["record_id"],
                    "issue_type": "SCREENING_CONFLICT",
                    "priority": "HIGH",
                    "status": "PENDING",
                    "decision": "",
                    "adjudicator_note": "Neutral and adversarial channels disagree.",
                }
            )
        if (neutral["decision"] == "UNCERTAIN" or adversarial["decision"] == "UNCERTAIN") and not record.get("manual_adjudicated"):
            adjudication.append(
                {
                    "record_id": record["record_id"],
                    "issue_type": "SCREENING_UNCERTAIN",
                    "priority": "HIGH" if record.get("c1_potential") == "POTENTIAL_DIRECT" else "MEDIUM",
                    "status": "PENDING",
                    "decision": "",
                    "adjudicator_note": "Peer-review, language, metadata, or boundary status requires human confirmation.",
                }
            )
        elif record.get("c1_potential") == "POTENTIAL_DIRECT" and not record.get("manual_adjudicated"):
            adjudication.append(
                {
                    "record_id": record["record_id"],
                    "issue_type": "POTENTIAL_C1",
                    "priority": "HIGH",
                    "status": "PENDING",
                    "decision": "",
                    "adjudicator_note": "Potential direct C1 prior art; verify full text and mechanism boundary.",
                }
            )
    by_id = {row["record_id"]: row for row in adjudication}
    by_id.update({row["record_id"]: row for row in conflicts})
    return {"decisions": decisions, "conflicts": conflicts, "adjudication": [by_id[key] for key in sorted(by_id)]}


def build_fulltext_manifest(repo_root: Path, records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Hash located full texts and preserve missing direct-candidate blockers."""

    manifest: list[dict[str, str]] = []
    for record in sorted(records, key=lambda item: item["record_id"]):
        raw_path = record.get("full_text_path", "")
        path = Path(raw_path) if raw_path else Path()
        if raw_path and not path.is_absolute():
            path = repo_root / path
        if raw_path and path.is_file():
            manifest.append(
                {
                    "record_id": record["record_id"],
                    "status": "LOCATED",
                    "path": str(path),
                    "sha256": _sha256_bytes(path.read_bytes()),
                    "locator_scope": "FULL_TEXT",
                }
            )
        elif record.get("full_text_status") == "REMOTE_LOCATED" and record.get("full_text_url"):
            manifest.append(
                {
                    "record_id": record["record_id"],
                    "status": "REMOTE_LOCATED",
                    "path": record["full_text_url"],
                    "sha256": "",
                    "locator_scope": "REMOTE",
                }
            )
        elif _is_direct_candidate(record):
            manifest.append(
                {
                    "record_id": record["record_id"],
                    "status": "MISSING_DIRECT_CANDIDATE",
                    "path": "",
                    "sha256": "",
                    "locator_scope": "",
                }
            )
        else:
            manifest.append(
                {
                    "record_id": record["record_id"],
                    "status": "NOT_REQUESTED",
                    "path": "",
                    "sha256": "",
                    "locator_scope": "",
                }
            )
    return manifest


def build_evidence_extraction(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
    """Flatten only locator-backed extraction fields; report missing locators."""

    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for record in sorted(records, key=lambda item: item["record_id"]):
        evidence = record.get("evidence", {})
        if not isinstance(evidence, dict):
            continue
        for facet in sorted(evidence):
            item = evidence[facet]
            if not isinstance(item, dict) or not str(item.get("locator", "")).strip():
                errors.append(f"{record['record_id']}:{facet}:missing_locator")
                continue
            rows.append(
                {
                    "record_id": record["record_id"],
                    "facet": str(facet),
                    "value": str(item.get("value", "")),
                    "locator": str(item["locator"]).strip(),
                    "source_type": str(item.get("source_type", "FULL_TEXT")),
                }
            )
    return rows, sorted(errors)


def assess_c1_verdict(records: Iterable[dict[str, Any]], blockers: Iterable[str]) -> dict[str, Any]:
    """Emit exactly one C1 verdict using the frozen fail-closed precedence."""

    rows = list(records)
    blocker_list = sorted(set(str(item) for item in blockers if item))
    for record in rows:
        direct = record.get("c1_relevance") == "DIRECT" or record.get("overlap_class") in {
            "DIRECT-EXACT",
            "DIRECT-FUNCTIONAL",
        }
        if direct and record.get("full_text_status") not in {"LOCATED", "REMOTE_LOCATED"}:
            blocker_list.append("missing_full_text")
    blocker_list = sorted(set(blocker_list))
    if blocker_list:
        return {
            "claim_id": "C1",
            "verdict": "UNRESOLVED",
            "confidence": "LOW",
            "blockers": blocker_list,
            "supporting_record_ids": [],
            "reason": "Fail-closed because the audit is incomplete or a direct candidate lacks located full text.",
        }

    exact_matching = [
        record
        for record in rows
        if record.get("overlap_class") == "DIRECT-EXACT"
        and record.get("mechanism_match")
        and record.get("integration_match")
        and record.get("evidence_match")
    ]
    exact_distinct_evidence = [
        record
        for record in rows
        if record.get("overlap_class") == "DIRECT-EXACT"
        and record.get("mechanism_match")
        and record.get("integration_match")
        and not record.get("evidence_match")
    ]
    functional = [record for record in rows if record.get("overlap_class") == "DIRECT-FUNCTIONAL"]
    strong_component = [record for record in rows if record.get("overlap_class") == "STRONG-COMPONENT"]
    if exact_matching:
        verdict, reason, selected = "DROP", "Exact prior matches mechanism, integration, and evidence.", exact_matching
    elif exact_distinct_evidence:
        verdict, reason, selected = "REFRAME", "Exact mechanism exists, but the audited evidence object is distinct.", exact_distinct_evidence
    elif functional:
        verdict, reason, selected = "NARROW", "A functional prior art overlap requires a narrower C1 wording.", functional
    else:
        verdict, reason, selected = "RETAIN", "No exact or functional prior was found; only strong-component or weaker overlap remains.", strong_component
    return {
        "claim_id": "C1",
        "verdict": verdict,
        "confidence": "MODERATE",
        "blockers": [],
        "supporting_record_ids": sorted(record["record_id"] for record in selected),
        "reason": reason,
    }


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_raw_snapshot(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", [])
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("raw snapshot must be a JSON list of objects or {records: [...]}")
    return data


def _retrieval_blockers_from_output(output_dir: Path) -> list[str]:
    """Carry preserved retrieval failures into every derived replay."""

    manifest_path = output_dir / "retrieval_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cutoff_path = output_dir / "retrieval_cutoff.json"
    cutoff = json.loads(cutoff_path.read_text(encoding="utf-8")) if cutoff_path.is_file() else {"status": "OPEN"}
    fallback_path = output_dir / "fallback_coverage.csv"
    fallback_rows: list[dict[str, str]] = []
    if fallback_path.is_file():
        with fallback_path.open(encoding="utf-8", newline="") as handle:
            fallback_rows = list(csv.DictReader(handle))
    closure = assess_dblp_retry_closure(cutoff, fallback_rows)
    if closure["status"] == "CLOSED_BLOCKED":
        return ["retrieval_gap_uncovered"]
    if closure["status"] == "OPEN":
        return ["retrieval_channel_open"]
    blockers: list[str] = []
    return sorted(set(blockers))


_ADJUDICATION_FIELDS = {
    "peer_reviewed",
    "peer_review_status",
    "language",
    "topic_relevant",
    "topic_relevance",
    "full_text_status",
    "full_text_path",
    "full_text_url",
    "c1_relevance",
    "overlap_class",
    "mechanism_match",
    "integration_match",
    "evidence_match",
    "evidence",
    "study_family_id",
}


def apply_adjudications(
    raw_records: Iterable[dict[str, Any]], adjudications: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply an explicit, replayable manual-adjudication patch to raw records.

    Records are addressed by ``record_id`` when available, then DOI, then the
    normalized title/year pair.  Unknown or ambiguous targets are reported as
    errors instead of being silently ignored.  The internal marker is retained
    through normalization so a manually adjudicated candidate is not routed
    back into the pending queue on the next deterministic replay.
    """

    records = [dict(record) for record in raw_records]
    normalized = [normalize_record(record) for record in records]
    by_id: dict[str, list[int]] = {}
    by_doi: dict[str, list[int]] = {}
    by_title_year: dict[tuple[str, int | None], list[int]] = {}
    for index, record in enumerate(normalized):
        by_id.setdefault(record["record_id"], []).append(index)
        if record.get("doi"):
            by_doi.setdefault(record["doi"], []).append(index)
        by_title_year.setdefault((record.get("title_normalized", ""), record.get("year")), []).append(index)

    errors: list[str] = []
    for ordinal, patch in enumerate(adjudications):
        if not isinstance(patch, dict):
            errors.append(f"adjudication[{ordinal}]:not_object")
            continue
        target_indices: list[int] = []
        target = str(patch.get("record_id", "")).strip()
        if target and target in by_id:
            target_indices = list(by_id[target])
            # A historical record ID may identify only one manifestation. If
            # the adjudication also carries a DOI, propagate the patch to all
            # manifestations of that DOI so deduplication cannot discard the
            # adjudicated state when a more complete index row appears.
            doi = _canonical_doi(patch.get("doi"))
            if doi:
                target_indices = sorted(set(target_indices).union(by_doi.get(doi, [])))
        else:
            doi = _canonical_doi(patch.get("doi"))
            if doi and by_doi.get(doi):
                target_indices = list(by_doi[doi])
            else:
                title = _normalize_title(patch.get("title", ""))
                year = _as_int(patch.get("year"))
                candidates = by_title_year.get((title, year), []) if title else []
                target_indices = list(candidates)
        if not target_indices:
            errors.append(f"adjudication[{ordinal}]:unknown_or_ambiguous_target")
            continue
        unknown_fields = sorted(set(patch) - _ADJUDICATION_FIELDS - {"record_id", "doi", "title", "year", "note"})
        if unknown_fields:
            errors.append(f"adjudication[{ordinal}]:unknown_fields:{','.join(unknown_fields)}")
            continue
        for target_index in target_indices:
            record = records[target_index]
            for field in sorted(_ADJUDICATION_FIELDS.intersection(patch)):
                record[field] = patch[field]
            record["manual_adjudicated"] = True
    return records, sorted(errors)


def build_query_url(index: str, query: str, page: int, rows: int = 100) -> str:
    """Build a fixed, credential-free URL for one allowed open index."""

    if index not in ALLOWED_DATABASES:
        raise ValueError(f"unsupported index: {index}")
    if page < 0 or rows <= 0:
        raise ValueError("page must be non-negative and rows must be positive")
    if index == "DBLP":
        params = {"q": query, "h": rows, "f": page * rows, "format": "json"}
        base = "https://dblp.org/search/publ/api"
    elif index == "OpenAlex":
        params = {
            "search": query,
            "filter": "from_publication_date:2013-01-01,to_publication_date:2026-08-07",
            "page": page + 1,
            "per-page": rows,
        }
        base = "https://api.openalex.org/works"
    else:
        params = {
            "query.bibliographic": query,
            "filter": "from-pub-date:2013-01-01,until-pub-date:2026-08-07",
            "rows": rows,
            "offset": page * rows,
        }
        base = "https://api.crossref.org/works"
    return base + "?" + urllib.parse.urlencode(params)


def _default_transport(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "muKG-LB-X1.5-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310 - fixed allow-listed URLs
        return json.loads(response.read().decode("utf-8"))


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _crossref_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        date_parts = item.get(key, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            return _as_int(date_parts[0][0])
    return None


def _openalex_abstract(item: dict[str, Any]) -> str:
    inverted = item.get("abstract_inverted_index") or {}
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for position in positions:
            words.append((int(position), word))
    return " ".join(word for _, word in sorted(words))


def _parse_index_payload(index: str, payload: dict[str, Any], query: str, page: int, stage: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if index == "DBLP":
        hits = payload.get("result", {}).get("hits", {}).get("hit", [])
        if isinstance(hits, dict):
            hits = [hits]
        for hit in hits:
            info = hit.get("info", {})
            records.append(
                {
                    "title": info.get("title", ""),
                    "year": info.get("year"),
                    "doi": info.get("doi", ""),
                    "dblp_key": info.get("key", ""),
                    "venue": info.get("venue", ""),
                    "source_index": index,
                    "source_identifier": info.get("key", ""),
                    "retrieval_stage": stage,
                    "query": query,
                    "peer_reviewed": None,
                    "peer_review_status": "UNVERIFIED",
                    "language": "",
                }
            )
    elif index == "OpenAlex":
        for item in payload.get("results", []):
            source = (item.get("primary_location") or {}).get("source") or {}
            records.append(
                {
                    "title": item.get("title", ""),
                    "year": item.get("publication_year"),
                    "doi": item.get("doi", ""),
                    "venue": source.get("display_name", ""),
                    "source_index": index,
                    "source_identifier": item.get("id", ""),
                    "retrieval_stage": stage,
                    "query": query,
                    "peer_reviewed": None,
                    "peer_review_status": "UNVERIFIED",
                    "language": item.get("language", ""),
                    "abstract": _openalex_abstract(item),
                }
            )
    else:
        for item in payload.get("message", {}).get("items", []):
            titles = item.get("title") or [""]
            containers = item.get("container-title") or [""]
            records.append(
                {
                    "title": titles[0],
                    "year": _crossref_year(item),
                    "doi": item.get("DOI", ""),
                    "venue": containers[0],
                    "source_index": index,
                    "source_identifier": item.get("URL", ""),
                    "retrieval_stage": stage,
                    "query": query,
                    "peer_reviewed": None,
                    "peer_review_status": "UNVERIFIED",
                    "language": "",
                    "abstract": item.get("abstract", ""),
                }
            )
    return records


def fetch_index_page(
    index: str,
    query: str,
    page: int,
    rows: int = 100,
    transport: Callable[[str], dict[str, Any]] | None = None,
    retrieval_stage: str = "WIDE_SENTINEL",
    retrieved_at: str = "",
) -> dict[str, Any]:
    """Fetch and normalize one page; transport is injectable for CPU fixtures."""

    url = build_query_url(index, query, page, rows)
    payload = (transport or _default_transport)(url)
    if not isinstance(payload, dict):
        raise ValueError("index response must be a JSON object")
    records = _parse_index_payload(index, payload, query, page, retrieval_stage)
    if index == "DBLP":
        hit_count = _as_int(payload.get("result", {}).get("hits", {}).get("@total")) or 0
    elif index == "OpenAlex":
        hit_count = _as_int(payload.get("meta", {}).get("count")) or 0
    else:
        hit_count = _as_int(payload.get("message", {}).get("total-results")) or 0
    return {
        "index": index,
        "query": query,
        "page": page,
        "rows": rows,
        "retrieval_stage": retrieval_stage,
        "status": "OK",
        "error": "",
        "url": url,
        "retrieved_at": retrieved_at,
        "hit_count": hit_count,
        "raw_payload_sha256": _payload_hash(payload),
        "raw_payload": payload,
        "records": records,
    }


def build_retrieval_plan(protocol: dict[str, Any]) -> list[dict[str, str]]:
    """Return the frozen G0 and wide-sentinel query plan in stable order."""

    plan: list[dict[str, str]] = []
    seeds = protocol.get("retrieval", {}).get("g0_seeds", [])
    wide = protocol.get("retrieval", {}).get("wide_sentinel_queries", [])
    for index in ALLOWED_DATABASES:
        for seed in seeds:
            plan.append({"index": index, "query": seed["title"], "retrieval_stage": "G0_SEEDS"})
        for query in wide:
            plan.append({"index": index, "query": query, "retrieval_stage": "WIDE_SENTINEL"})
    return plan


def fetch_protocol_queries(
    protocol: dict[str, Any],
    rows: int = 100,
    transport: Callable[[str], dict[str, Any]] | None = None,
    retrieved_at: str = "",
    retries: int = 0,
    retry_delay: float = 0.0,
) -> list[dict[str, Any]]:
    """Fetch the frozen G0 and wide-sentinel plan in stable order."""

    pages: list[dict[str, Any]] = []
    for item in build_retrieval_plan(protocol):
        last_error: Exception | None = None
        for attempt in range(max(0, retries) + 1):
            try:
                pages.append(
                    fetch_index_page(
                        item["index"],
                        item["query"],
                        page=0,
                        rows=rows,
                        transport=transport,
                        retrieval_stage=item["retrieval_stage"],
                        retrieved_at=retrieved_at,
                    )
                )
                last_error = None
                break
            except Exception as exc:  # preserve a failed page instead of hiding recall loss
                last_error = exc
                if attempt < max(0, retries) and retry_delay > 0:
                    time.sleep(retry_delay * (attempt + 1))
        if last_error is not None:
            url = build_query_url(item["index"], item["query"], page=0, rows=rows)
            pages.append(
                {
                    "index": item["index"],
                    "query": item["query"],
                    "page": 0,
                    "rows": rows,
                    "retrieval_stage": item["retrieval_stage"],
                    "status": "FAILED",
                    "error": f"{type(last_error).__name__}: {last_error}",
                    "url": url,
                    "retrieved_at": retrieved_at,
                    "hit_count": 0,
                    "raw_payload_sha256": _payload_hash({}),
                    "raw_payload": {},
                    "records": [],
                }
            )
    return pages


def build_dblp_retry_schedule(
    pages: Iterable[dict[str, Any]],
    round_number: int,
    batch_size: int = 3,
) -> dict[str, Any]:
    """Plan at most ``batch_size`` failed DBLP queries per low-frequency batch."""

    if round_number < 1:
        raise ValueError("round_number must be positive")
    if batch_size < 1 or batch_size > 3:
        raise ValueError("batch_size must be between 1 and 3")
    failed = sorted(
        (
            page
            for page in pages
            if page.get("index") == "DBLP"
            and page.get("status") == "FAILED"
            and int(page.get("retry_round", 0)) < round_number
        ),
        key=lambda page: (page.get("query", ""), page.get("retrieval_stage", "")),
    )
    batches: list[dict[str, Any]] = []
    for batch_index in range(0, len(failed), batch_size):
        batch_pages = failed[batch_index : batch_index + batch_size]
        queries: list[dict[str, Any]] = []
        for page in batch_pages:
            fingerprint = _sha256_text(f"{round_number}|{page.get('query', '')}|{page.get('retrieval_stage', '')}")
            jitter = 3.0 + (int(fingerprint[:8], 16) % 2001) / 1000.0
            queries.append(
                {
                    "index": "DBLP",
                    "query": page.get("query", ""),
                    "retrieval_stage": page.get("retrieval_stage", ""),
                    "retry_round": round_number,
                    "delay_seconds": round(jitter, 3),
                }
            )
        batches.append(
            {
                "batch_index": len(batches),
                "wait_before_batch_seconds": 0 if not batches else 600,
                "queries": queries,
            }
        )
    return {
        "round_number": round_number,
        "batch_size": batch_size,
        "batches": batches,
        "failed_count": len(failed),
    }


def _retry_query_key(query: Any, retrieval_stage: Any) -> str:
    return _sha256_text(f"{retrieval_stage}|{query}")[:24]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _retry_completed_rounds(state: dict[str, Any]) -> int:
    entries = list(state.get("queries", []))
    max_rounds = int(state.get("max_rounds", 3))
    observed_rounds = sorted({
        int(attempt.get("round", 0))
        for entry in entries
        for attempt in entry.get("attempts", [])
        if 0 < int(attempt.get("round", 0)) <= max_rounds
    })
    completed = 0
    for round_number in observed_rounds:
        if all(
            any(int(attempt.get("round", 0)) == round_number for attempt in entry.get("attempts", []))
            or (entry.get("status") == "RECOVERED" and any(int(attempt.get("round", 0)) < round_number for attempt in entry.get("attempts", [])))
            or (entry.get("status") == "UNRESOLVED_MISSING_PAGE" and any(int(attempt.get("round", 0)) <= round_number for attempt in entry.get("attempts", [])))
            for entry in entries
        ):
            completed = round_number if round_number == completed + 1 else completed
        else:
            break
    return completed


def _refresh_retry_state_counts(state: dict[str, Any]) -> dict[str, Any]:
    entries = list(state.get("queries", []))
    pending = [entry for entry in entries if entry.get("status") == "PENDING"]
    recovered = [entry for entry in entries if entry.get("status") == "RECOVERED"]
    unresolved = [entry for entry in entries if entry.get("status") == "UNRESOLVED_MISSING_PAGE"]
    state["initial_query_count"] = len(entries)
    state["pending_query_count"] = len(pending)
    state["recovered_query_count"] = len(recovered)
    state["unresolved_query_count"] = len(unresolved)
    state["completed_rounds"] = _retry_completed_rounds(state)
    state["current_round"] = min((int(entry["next_round"]) for entry in pending), default=state["max_rounds"] + 1)
    state["status"] = "OPEN" if pending else ("COMPLETE" if not unresolved else "CLOSED")
    return state


def initialize_dblp_retry_state(
    pages: Iterable[dict[str, Any]], batch_size: int = 3, max_rounds: int = 3
) -> dict[str, Any]:
    """Create a stable retry universe from the current DBLP snapshot.

    A DBLP page is part of the retry universe when it is currently failed or
    carries a positive retry_round, which preserves the three queries already
    recovered by the existing snapshot migration.
    """

    if batch_size < 1 or batch_size > 3:
        raise ValueError("batch_size must be between 1 and 3")
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    entries: list[dict[str, Any]] = []
    for page in sorted(
        (
            page
            for page in pages
            if page.get("index") == "DBLP"
            and (page.get("status") == "FAILED" or int(page.get("retry_round", 0)) > 0)
        ),
        key=lambda item: (str(item.get("query", "")), str(item.get("retrieval_stage", ""))),
    ):
        retry_round = int(page.get("retry_round", 0))
        status = "RECOVERED" if page.get("status") == "OK" else (
            "UNRESOLVED_MISSING_PAGE" if retry_round >= max_rounds else "PENDING"
        )
        attempts = []
        if retry_round > 0:
            attempts.append(
                {
                    "round": retry_round,
                    "status": page.get("status", "FAILED"),
                    "error": page.get("error", ""),
                    "hit_count": int(page.get("hit_count", 0) or 0),
                    "raw_payload_sha256": page.get("raw_payload_sha256", ""),
                    "retrieved_at": page.get("retrieved_at", ""),
                }
            )
        entries.append(
            {
                "query_key": _retry_query_key(page.get("query", ""), page.get("retrieval_stage", "")),
                "query": page.get("query", ""),
                "retrieval_stage": page.get("retrieval_stage", ""),
                "status": status,
                "next_round": None if status != "PENDING" else max(1, retry_round + 1),
                "attempts": attempts,
                "fallback": {},
            }
        )
    state = {
        "protocol_id": "phase-x-x1-5-literature-audit-v1",
        "state_version": 1,
        "batch_size": batch_size,
        "max_rounds": max_rounds,
        "min_batch_interval_seconds": RETRY_BATCH_INTERVAL_SECONDS,
        "last_batch_completed_at": "",
        "next_eligible_at": "",
        "batches": [],
        "queries": entries,
    }
    return _refresh_retry_state_counts(state)


def select_next_dblp_batch(state: dict[str, Any]) -> dict[str, Any]:
    """Select the next stable batch without exposing a shifting batch index."""

    pending = [entry for entry in state.get("queries", []) if entry.get("status") == "PENDING"]
    if not pending:
        return {"status": state.get("status", "COMPLETE"), "round_number": state.get("current_round", 1), "queries": []}
    round_number = min(int(entry.get("next_round", 1)) for entry in pending)
    selected = [entry for entry in pending if int(entry.get("next_round", 1)) == round_number]
    selected = sorted(selected, key=lambda entry: (str(entry.get("query", "")), str(entry.get("retrieval_stage", ""))))
    return {
        "status": "READY",
        "round_number": round_number,
        "batch_sequence": len(state.get("batches", [])) + 1,
        "queries": selected[: int(state.get("batch_size", 3))],
    }


def _fallback_page_is_valid(page: dict[str, Any]) -> bool:
    raw_hash = str(page.get("raw_payload_sha256", ""))
    return (
        page.get("status") == "OK"
        and int(page.get("hit_count", 0) or 0) > 0
        and bool(page.get("records"))
        and len(raw_hash) == 64
        and raw_hash != "0" * 64
    )


def qualify_fallback_coverage(pages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Qualify alternative-index coverage for every failed DBLP query."""

    rows = list(pages)
    result: list[dict[str, str]] = []
    for failed in sorted(
        (page for page in rows if page.get("index") == "DBLP" and page.get("status") == "FAILED"),
        key=lambda item: (str(item.get("query", "")), str(item.get("retrieval_stage", ""))),
    ):
        query = str(failed.get("query", ""))
        stage = str(failed.get("retrieval_stage", ""))
        alternatives = [
            page for page in rows
            if page.get("index") in {"OpenAlex", "Crossref"}
            and page.get("query") == query
            and page.get("retrieval_stage") == stage
            and _fallback_page_is_valid(page)
        ]
        matched_indexes: list[str] = []
        matched_ids: list[str] = []
        evidence: list[str] = []
        for page in alternatives:
            records = list(page.get("records", []))
            if stage == "G0_SEEDS":
                expected_doi = _canonical_doi(failed.get("expected_doi") or failed.get("doi"))
                if expected_doi:
                    matching = [record for record in records if _canonical_doi(record.get("doi")) == expected_doi]
                else:
                    matching = [record for record in records if _normalize_title(record.get("title", "")) == _normalize_title(query)]
                if not matching:
                    continue
            else:
                matching = records
            matched_indexes.append(str(page.get("index", "")))
            matched_ids.extend(str(record.get("source_identifier", "") or record.get("doi", "")) for record in matching)
            evidence.append(f"{page.get('index','')}:{page.get('raw_payload_sha256','')}")
        qualified = bool(matched_indexes)
        result.append(
            {
                "failed_index": "DBLP",
                "query": query,
                "retrieval_stage": stage,
                "fallback_status": "AVAILABLE" if qualified else "MISSING",
                "fallback_indexes": ";".join(sorted(set(matched_indexes))),
                "qualification": "QUALIFIED" if qualified else "MISSING",
                "matched_record_ids": ";".join(sorted(set(item for item in matched_ids if item))),
                "raw_evidence": ";".join(sorted(set(evidence))),
                "disposition": "ADVISORY_ONLY" if qualified else "BLOCKING",
            }
        )
    return result


def assess_dblp_retry_closure(cutoff: dict[str, Any], fallback_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Map retry cutoff plus qualified fallbacks to the hard-gate disposition."""

    rows = list(fallback_rows)
    cutoff_status = str(cutoff.get("status", "OPEN"))
    if cutoff_status == "COMPLETE":
        return {"status": "COMPLETE", "blocking_conditions": [], "advisories": []}
    if cutoff_status == "CLOSED_WITH_FALLBACK":
        return {"status": "CLOSED_WITH_FALLBACK", "blocking_conditions": [], "advisories": ["dblp_missing_page_with_qualified_fallback"], "uncovered_queries": []}
    if cutoff_status == "CLOSED_BLOCKED":
        return {"status": "CLOSED_BLOCKED", "blocking_conditions": ["retrieval_gap_uncovered"], "advisories": [], "uncovered_queries": []}
    if cutoff_status != "CLOSED":
        return {"status": "OPEN", "blocking_conditions": ["retrieval_channel_open"], "advisories": []}
    missing = [row for row in rows if row.get("qualification") != "QUALIFIED"]
    if missing:
        return {
            "status": "CLOSED_BLOCKED",
            "blocking_conditions": ["retrieval_gap_uncovered"],
            "advisories": [],
            "uncovered_queries": sorted(str(row.get("query", "")) for row in missing),
        }
    return {
        "status": "CLOSED_WITH_FALLBACK",
        "blocking_conditions": [],
        "advisories": ["dblp_missing_page_with_qualified_fallback"],
        "uncovered_queries": [],
    }


def assess_retrieval_cutoff(pages: Iterable[dict[str, Any]], completed_rounds: int, max_rounds: int = 3) -> dict[str, Any]:
    """Compatibility wrapper for legacy snapshots; stateful CLI uses the retry ledger."""

    failed = [page for page in pages if page.get("index") == "DBLP" and page.get("status") == "FAILED"]
    closed = completed_rounds >= max_rounds and bool(failed)
    return {
        "status": "CLOSED" if closed else ("COMPLETE" if not failed else "OPEN"),
        "completed_rounds": completed_rounds,
        "max_rounds": max_rounds,
        "unresolved_count": len(failed) if closed else 0,
        "unresolved_status": "UNRESOLVED_MISSING_PAGE" if closed else "",
        "unresolved_queries": sorted(page.get("query", "") for page in failed) if closed else [],
    }


def build_fallback_coverage(pages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Return qualified fallback rows while preserving the legacy function name."""

    return qualify_fallback_coverage(pages)


def write_retrieval_snapshot(output_dir: Path, pages: Iterable[dict[str, Any]]) -> dict[str, Path]:
    """Write raw pages, normalized records, and a deterministic query manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    page_rows = sorted(pages, key=lambda item: (item["index"], item["query"], item["page"]))
    manifest_pages: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for ordinal, page in enumerate(page_rows):
        filename = f"page-{ordinal:04d}-{page['raw_payload_sha256'][:12]}.json"
        raw_path = raw_dir / filename
        raw_path.write_text(json.dumps(page["raw_payload"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_pages.append(
            {
                "index": page["index"],
                "query": page["query"],
                "expected_doi": page.get("expected_doi", ""),
                "page": page["page"],
                "rows": page["rows"],
                "retrieval_stage": page["retrieval_stage"],
                "retry_round": int(page.get("retry_round", 0)),
                "status": page.get("status", "OK"),
                "error": page.get("error", ""),
                "url": page["url"],
                "retrieved_at": page.get("retrieved_at", ""),
                "hit_count": page["hit_count"],
                "raw_payload_sha256": page["raw_payload_sha256"],
                "raw_payload_path": str(Path("raw_pages") / filename),
            }
        )
        records.extend(page["records"])
    records = sorted(records, key=lambda item: (_normalize_title(item.get("title", "")), _as_int(item.get("year")) or 0, item.get("source_index", "")))
    manifest_path = output_dir / "retrieval_manifest.json"
    _write_json(manifest_path, {"protocol_id": "phase-x-x1-5-literature-audit-v1", "pages": manifest_pages})
    records_path = output_dir / "retrieval_records.json"
    _write_json(records_path, records)
    schedule_path = output_dir / "retry_schedule.json"
    retry_state_path = output_dir / RETRY_STATE_FILENAME
    if retry_state_path.is_file():
        retry_state = _refresh_retry_state_counts(json.loads(retry_state_path.read_text(encoding="utf-8")))
        _write_json(retry_state_path, retry_state)
        _write_json(schedule_path, _retry_schedule_from_state(retry_state))
        cutoff_value = _retry_cutoff_from_state(retry_state)
    else:
        retry_state = initialize_dblp_retry_state(page_rows)
        _write_json(retry_state_path, retry_state)
        _write_json(schedule_path, _retry_schedule_from_state(retry_state))
        cutoff_value = _retry_cutoff_from_state(retry_state)
    cutoff_path = output_dir / "retrieval_cutoff.json"
    _write_json(cutoff_path, cutoff_value)
    fallback_path = output_dir / "fallback_coverage.csv"
    _write_csv(fallback_path, CSV_SCHEMAS["fallback_coverage.csv"], build_fallback_coverage(page_rows))
    return {
        "retrieval_manifest.json": manifest_path,
        "retrieval_records.json": records_path,
        "retry_schedule.json": schedule_path,
        "retrieval_cutoff.json": cutoff_path,
        "fallback_coverage.csv": fallback_path,
    }


def _load_retrieval_pages(output_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((output_dir / "retrieval_manifest.json").read_text(encoding="utf-8"))
    records = json.loads((output_dir / "retrieval_records.json").read_text(encoding="utf-8"))
    pages: list[dict[str, Any]] = []
    for page in manifest["pages"]:
        raw_path = output_dir / page["raw_payload_path"]
        payload = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else {}
        page_records = [
            record
            for record in records
            if record.get("source_index") == page["index"]
            and record.get("query") == page["query"]
            and record.get("retrieval_stage") == page["retrieval_stage"]
        ]
        pages.append({**page, "raw_payload": payload, "records": page_records})
    return pages


def _retry_state_path(output_dir: Path) -> Path:
    return output_dir / RETRY_STATE_FILENAME


def _load_or_initialize_retry_state(output_dir: Path, pages: list[dict[str, Any]]) -> dict[str, Any]:
    path = _retry_state_path(output_dir)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("queries"), list):
            raise ValueError("dblp_retry_state.json has invalid shape")
        return _refresh_retry_state_counts(state)
    return initialize_dblp_retry_state(pages)


def _retry_batch_due(state: dict[str, Any], now: str) -> bool:
    last = _parse_utc(state.get("last_batch_completed_at"))
    current = _parse_utc(now)
    return not last or not current or current >= last + timedelta(seconds=int(state.get("min_batch_interval_seconds", RETRY_BATCH_INTERVAL_SECONDS)))


def _record_retry_batch(
    state: dict[str, Any], batch: dict[str, Any], results: dict[str, dict[str, Any]], completed_at: str
) -> dict[str, Any]:
    by_key = {entry.get("query_key"): entry for entry in state.get("queries", [])}
    for selected in batch.get("queries", []):
        key = selected.get("query_key")
        entry = by_key[key]
        result = results[key]
        round_number = int(batch["round_number"])
        entry.setdefault("attempts", []).append(
            {
                "round": round_number,
                "status": result.get("status", "FAILED"),
                "error": result.get("error", ""),
                "hit_count": int(result.get("hit_count", 0) or 0),
                "raw_payload_sha256": result.get("raw_payload_sha256", ""),
                "retrieved_at": result.get("retrieved_at", completed_at),
            }
        )
        if result.get("status") == "OK":
            entry["status"] = "RECOVERED"
            entry["next_round"] = None
        elif round_number >= int(state.get("max_rounds", 3)):
            entry["status"] = "UNRESOLVED_MISSING_PAGE"
            entry["next_round"] = None
        else:
            entry["status"] = "PENDING"
            entry["next_round"] = round_number + 1
    state.setdefault("batches", []).append(
        {
            "batch_sequence": batch.get("batch_sequence", len(state.get("batches", [])) + 1),
            "round_number": batch.get("round_number"),
            "query_keys": [entry.get("query_key") for entry in batch.get("queries", [])],
            "completed_at": completed_at,
        }
    )
    state["last_batch_completed_at"] = completed_at
    current = _parse_utc(completed_at)
    state["next_eligible_at"] = (
        (current + timedelta(seconds=int(state.get("min_batch_interval_seconds", RETRY_BATCH_INTERVAL_SECONDS)))).isoformat(timespec="seconds").replace("+00:00", "Z")
        if current else ""
    )
    return _refresh_retry_state_counts(state)


def _retry_cutoff_from_state(state: dict[str, Any]) -> dict[str, Any]:
    unresolved = [entry for entry in state.get("queries", []) if entry.get("status") == "UNRESOLVED_MISSING_PAGE"]
    cutoff = {
        "status": "COMPLETE" if not state.get("pending_query_count") and not unresolved else ("CLOSED" if unresolved and not state.get("pending_query_count") else "OPEN"),
        "completed_rounds": int(state.get("completed_rounds", 0)),
        "max_rounds": int(state.get("max_rounds", 3)),
        "unresolved_count": len(unresolved),
        "unresolved_status": "UNRESOLVED_MISSING_PAGE" if unresolved else "",
        "unresolved_queries": sorted(str(entry.get("query", "")) for entry in unresolved),
    }
    closure_status = str(state.get("closure", {}).get("status", ""))
    if closure_status in {"CLOSED_WITH_FALLBACK", "CLOSED_BLOCKED"}:
        cutoff["status"] = closure_status
    return cutoff


def _retry_schedule_from_state(state: dict[str, Any]) -> dict[str, Any]:
    batch = select_next_dblp_batch(state)
    return {
        "state_version": state.get("state_version", 1),
        "round_number": batch.get("round_number", state.get("current_round", 1)),
        "batch_size": state.get("batch_size", 3),
        "pending_count": state.get("pending_query_count", 0),
        "next_batch": [
            {
                "query_key": entry.get("query_key", ""),
                "query": entry.get("query", ""),
                "retrieval_stage": entry.get("retrieval_stage", ""),
                "retry_round": entry.get("next_round", ""),
            }
            for entry in batch.get("queries", [])
        ],
        "next_eligible_at": state.get("next_eligible_at", ""),
    }


def retry_dblp_next(
    output_dir: Path,
    transport: Callable[[str], dict[str, Any]] | None = None,
    now: str = "",
    sleep_fn: Callable[[float], None] | None = None,
    repo_root: Path | None = None,
    rows: int = 100,
) -> dict[str, Any]:
    """Execute exactly one eligible next DBLP batch from the persistent ledger."""

    pages = _load_retrieval_pages(output_dir)
    state = _load_or_initialize_retry_state(output_dir, pages)
    now = now or _utc_now_iso()
    state_path = _retry_state_path(output_dir)
    if not _retry_batch_due(state, now):
        return {"status": "NOT_DUE", "next_eligible_at": state.get("next_eligible_at", ""), "paths": {RETRY_STATE_FILENAME: state_path}}
    _write_json(state_path, state)
    batch = select_next_dblp_batch(state)
    if not batch.get("queries"):
        fallback_rows = qualify_fallback_coverage(pages)
        closure = assess_dblp_retry_closure(_retry_cutoff_from_state(state), fallback_rows)
        state["fallback_rows"] = fallback_rows
        state["closure"] = closure
        cutoff = _retry_cutoff_from_state(state)
        _write_json(output_dir / "retrieval_cutoff.json", cutoff)
        _write_csv(output_dir / "fallback_coverage.csv", CSV_SCHEMAS["fallback_coverage.csv"], fallback_rows)
        _write_json(state_path, state)
        _write_json(output_dir / "retry_schedule.json", _retry_schedule_from_state(state))
        return {"status": closure["status"], "closure": closure, "paths": {RETRY_STATE_FILENAME: state_path}}
    results: dict[str, dict[str, Any]] = {}
    page_by_key = {(page["index"], page["query"], page["retrieval_stage"]): page for page in pages}
    sleeper = sleep_fn or time.sleep
    for entry in batch["queries"]:
        jitter_fingerprint = _sha256_text(f"{batch['round_number']}|{entry['query']}|{entry['retrieval_stage']}")
        jitter = 3.0 + (int(jitter_fingerprint[:8], 16) % 2001) / 1000.0
        sleeper(jitter)
        key = ("DBLP", entry["query"], entry["retrieval_stage"])
        try:
            result = fetch_index_page(
                "DBLP", entry["query"], page=0, rows=rows, transport=transport,
                retrieval_stage=entry["retrieval_stage"], retrieved_at=now,
            )
        except Exception as exc:
            old = page_by_key[key]
            result = {
                **old,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "retry_round": batch["round_number"],
                "retrieved_at": now,
                "hit_count": 0,
                "raw_payload": {},
                "records": [],
                "raw_payload_sha256": _payload_hash({}),
            }
        result["retry_round"] = batch["round_number"]
        page_by_key[key] = result
        results[entry["query_key"]] = result
    pages = list(page_by_key.values())
    state = _record_retry_batch(state, batch, results, now)
    fallback_rows = qualify_fallback_coverage(pages)
    cutoff = _retry_cutoff_from_state(state)
    closure = assess_dblp_retry_closure(cutoff, fallback_rows)
    if closure["status"] in {"CLOSED_WITH_FALLBACK", "CLOSED_BLOCKED"}:
        cutoff["status"] = closure["status"]
    state["fallback_rows"] = fallback_rows
    state["closure"] = closure
    retrieval_paths = write_retrieval_snapshot(output_dir, pages)
    _write_json(output_dir / "retrieval_cutoff.json", cutoff)
    _write_csv(output_dir / "fallback_coverage.csv", CSV_SCHEMAS["fallback_coverage.csv"], fallback_rows)
    _write_json(state_path, state)
    # write_retrieval_snapshot may have observed the pre-batch ledger; refresh
    # the schedule after the finalized state is persisted so the next batch
    # cannot be skipped by a stale/reindexed list.
    _write_json(output_dir / "retry_schedule.json", _retry_schedule_from_state(state))
    raw_records = [record for page in pages for record in page["records"]]
    adjudication_errors: list[str] = []
    manual_path = output_dir / "manual_adjudications.json"
    if manual_path.is_file():
        adjudications = json.loads(manual_path.read_text(encoding="utf-8"))
        if isinstance(adjudications, list):
            raw_records, adjudication_errors = apply_adjudications(raw_records, adjudications)
        else:
            adjudication_errors.append("manual_adjudications.json:not_list")
    hard_blockers = closure["blocking_conditions"] or (["retrieval_channel_open"] if closure["status"] == "OPEN" else [])
    derived_paths = write_outputs(repo_root or output_dir.parents[3], output_dir, raw_records, hard_blockers, adjudication_errors)
    return {
        "status": "BATCH_COMPLETED",
        "round_number": batch["round_number"],
        "batch_sequence": batch["batch_sequence"],
        "updated_query_count": len(batch["queries"]),
        "closure": closure,
        "paths": {**retrieval_paths, RETRY_STATE_FILENAME: state_path, **derived_paths},
    }


def dblp_retry_status(output_dir: Path) -> dict[str, Any]:
    """Return current retry status without network access or file writes."""

    pages = _load_retrieval_pages(output_dir)
    state_path = _retry_state_path(output_dir)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else initialize_dblp_retry_state(pages)
    )
    state = _refresh_retry_state_counts(state)
    fallback_rows = qualify_fallback_coverage(pages)
    closure = assess_dblp_retry_closure(_retry_cutoff_from_state(state), fallback_rows)
    return {
        "state_initialized": state_path.is_file(),
        "state": state,
        "closure": closure,
        "next_batch": select_next_dblp_batch(state),
    }


def retry_dblp_batch(
    output_dir: Path,
    round_number: int,
    batch_index: int,
    batch_size: int = 3,
    rows: int = 100,
    transport: Callable[[str], dict[str, Any]] | None = None,
    retries: int = 0,
    retry_delay: float = 4.0,
    honor_wait: bool = True,
    retrieved_at: str = "",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Retry one scheduled DBLP batch and preserve all other pages unchanged."""

    pages = _load_retrieval_pages(output_dir)
    schedule = build_dblp_retry_schedule(pages, round_number=round_number, batch_size=batch_size)
    if batch_index < 0 or batch_index >= len(schedule["batches"]):
        raise ValueError(f"batch_index {batch_index} is not available")
    batch = schedule["batches"][batch_index]
    if honor_wait and batch["wait_before_batch_seconds"]:
        time.sleep(batch["wait_before_batch_seconds"])
    page_by_key = {(page["index"], page["query"], page["retrieval_stage"]): page for page in pages}
    for item in batch["queries"]:
        if honor_wait:
            time.sleep(item["delay_seconds"])
        key = ("DBLP", item["query"], item["retrieval_stage"])
        last_error: Exception | None = None
        result: dict[str, Any] | None = None
        for attempt in range(max(0, retries) + 1):
            try:
                result = fetch_index_page(
                    "DBLP",
                    item["query"],
                    page=0,
                    rows=rows,
                    transport=transport,
                    retrieval_stage=item["retrieval_stage"],
                    retrieved_at=retrieved_at,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < max(0, retries) and retry_delay > 0:
                    time.sleep(retry_delay)
        if result is None:
            old = page_by_key[key]
            result = {
                **old,
                "status": "FAILED",
                "error": f"{type(last_error).__name__}: {last_error}",
                "retry_round": round_number,
                "records": [],
                "raw_payload": {},
                "raw_payload_sha256": _payload_hash({}),
            }
        else:
            result["retry_round"] = round_number
        page_by_key[key] = result
    updated_pages = list(page_by_key.values())
    retrieval_paths = write_retrieval_snapshot(output_dir, updated_pages)
    raw_records = [record for page in updated_pages for record in page["records"]]
    cutoff = assess_retrieval_cutoff(updated_pages, completed_rounds=round_number)
    blockers = []
    if any(page.get("status") == "FAILED" and page.get("index") == "DBLP" for page in updated_pages):
        blockers.append("retrieval_failed")
    if cutoff["status"] == "CLOSED":
        blockers.append("UNRESOLVED_MISSING_PAGE")
    adjudication_errors: list[str] = []
    manual_path = output_dir / "manual_adjudications.json"
    if manual_path.is_file():
        adjudications = json.loads(manual_path.read_text(encoding="utf-8"))
        if not isinstance(adjudications, list):
            adjudication_errors.append("manual_adjudications.json:not_list")
        else:
            raw_records, adjudication_errors = apply_adjudications(raw_records, adjudications)
    derived_paths = write_outputs(
        repo_root or output_dir.parents[3],
        output_dir,
        raw_records,
        blockers,
        adjudication_errors,
    )
    return {
        "round_number": round_number,
        "batch_index": batch_index,
        "updated_query_count": len(batch["queries"]),
        "cutoff": cutoff,
        "paths": {**retrieval_paths, **derived_paths},
    }


def _derive(
    repo_root: Path,
    raw_records: list[dict[str, Any]],
    extra_blockers: Iterable[str] = (),
    adjudication_errors: Iterable[str] = (),
) -> dict[str, Any]:
    protocol = load_protocol(repo_root)
    normalized = [normalize_record(item) for item in raw_records]
    records, dedup_decisions = deduplicate_records(normalized)
    dual_screening = run_dual_screening(records)
    screening = dual_screening["decisions"]
    conflicts = dual_screening["conflicts"]
    adjudication = dual_screening["adjudication"]
    neutral_screening = [row for row in screening if row["channel"] == "NEUTRAL_ELIGIBILITY"]
    included_ids = {row["record_id"] for row in neutral_screening if row["decision"] == "INCLUDE"}
    included = [record for record in records if record["record_id"] in included_ids]
    fulltext_manifest = build_fulltext_manifest(repo_root, records)
    evidence_extraction, extraction_errors = build_evidence_extraction(records)
    candidate_ids = {
        row["record_id"]
        for row in neutral_screening
        if row["decision"] != "EXCLUDE"
    }
    candidate_records = [record for record in records if record["record_id"] in candidate_ids]
    blockers: list[str] = list(extra_blockers)
    adjudication_error_list = sorted(set(str(item) for item in adjudication_errors if item))
    if adjudication_error_list:
        blockers.append("adjudication_errors")
    if not raw_records and not blockers:
        blockers.append("search_not_executed")
    if raw_records and any(record.get("peer_reviewed") is not True for record in candidate_records):
        blockers.append("peer_review_status_unverified")
    if raw_records and not included:
        blockers.append("screening_incomplete")
    if adjudication:
        blockers.append("human_adjudication_pending")
    if extraction_errors:
        blockers.append("evidence_locator_missing")
    verdict = assess_c1_verdict(included, blockers)
    screening_summary = {
        "record_count": len(records),
        "neutral_include_count": sum(row["decision"] == "INCLUDE" for row in neutral_screening),
        "neutral_uncertain_count": sum(row["decision"] == "UNCERTAIN" for row in neutral_screening),
        "neutral_auto_exclude_count": sum(row["notes"] == "AUTO_EXCLUDE_OBVIOUSLY_IRRELEVANT" for row in neutral_screening),
        "potential_direct_count": sum(record.get("c1_potential") == "POTENTIAL_DIRECT" for record in records),
        "potential_component_count": sum(record.get("c1_potential") == "POTENTIAL_COMPONENT" for record in records),
        "human_adjudication_count": len(adjudication),
    }
    matrix = [
        {
            "record_id": record["record_id"],
            "overlap_class": record["overlap_class"],
            "c1_relevance": record["c1_relevance"],
            "mechanism_match": record["mechanism_match"],
            "integration_match": record["integration_match"],
            "evidence_match": record["evidence_match"],
            "full_text_status": record["full_text_status"],
            "notes": "",
        }
        for record in included
    ]
    return {
        "protocol": protocol,
        "records": records,
        "screening": screening,
        "conflicts": conflicts,
        "adjudication": adjudication,
        "fulltext_manifest": fulltext_manifest,
        "evidence_extraction": evidence_extraction,
        "extraction_errors": extraction_errors,
        "dedup_decisions": dedup_decisions,
        "included": included,
        "novelty_matrix": matrix,
        "novelty_decision": verdict,
        "screening_summary": screening_summary,
        "adjudication_errors": adjudication_error_list,
    }


def write_outputs(
    repo_root: Path,
    output_dir: Path,
    raw_records: list[dict[str, Any]] | None = None,
    extra_blockers: Iterable[str] = (),
    adjudication_errors: Iterable[str] = (),
) -> dict[str, Path]:
    """Derive and serialize the first-stage audit outputs deterministically."""

    derived = _derive(
        repo_root,
        [] if raw_records is None else raw_records,
        extra_blockers,
        adjudication_errors,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records_rows = [
        {field: record.get(field, "") for field in CSV_SCHEMAS["records.csv"]}
        for record in derived["records"]
    ]
    paths: dict[str, Path] = {}
    for name, rows in (
        ("records.csv", records_rows),
        ("screening_decisions.csv", derived["screening"]),
        ("human_adjudication.csv", derived["adjudication"]),
        ("fulltext_manifest.csv", derived["fulltext_manifest"]),
        ("evidence_extraction.csv", derived["evidence_extraction"]),
        ("novelty_matrix.csv", derived["novelty_matrix"]),
    ):
        path = output_dir / name
        _write_csv(path, CSV_SCHEMAS[name], rows)
        paths[name] = path
    novelty_path = output_dir / "novelty_decision.json"
    _write_json(novelty_path, derived["novelty_decision"])
    paths["novelty_decision.json"] = novelty_path
    screening_summary_path = output_dir / "screening_summary.json"
    _write_json(screening_summary_path, derived["screening_summary"])
    paths["screening_summary.json"] = screening_summary_path
    checks = {
        "protocol_valid": not validate_protocol(derived["protocol"]),
        "raw_snapshot_record_count": len(derived["records"]),
        "dedup_decision_count": len(derived["dedup_decisions"]),
        "human_adjudication_pending_count": len(derived["adjudication"]),
        "evidence_extraction_error_count": len(derived["extraction_errors"]),
        "c1_exactly_one_verdict": derived["novelty_decision"]["verdict"] in VERDICTS,
        "novelty_blockers": derived["novelty_decision"]["blockers"],
        "adjudication_errors": derived["adjudication_errors"],
        "c2_c4_contribution_verdicts_emitted": False,
        "dynamic_fields_emitted": False,
        "status": "PASS" if not validate_protocol(derived["protocol"]) else "FAIL",
    }
    retry_state_path = output_dir / RETRY_STATE_FILENAME
    if retry_state_path.is_file():
        retry_state = json.loads(retry_state_path.read_text(encoding="utf-8"))
        retry_pages = _load_retrieval_pages(output_dir) if (output_dir / "retrieval_manifest.json").is_file() else []
        state_keys = {entry.get("query_key") for entry in retry_state.get("queries", [])}
        manifest_keys = {
            _retry_query_key(page.get("query", ""), page.get("retrieval_stage", ""))
            for page in retry_pages
            if page.get("index") == "DBLP"
            and (page.get("status") == "FAILED" or int(page.get("retry_round", 0)) > 0)
        }
        fallback_path = output_dir / "fallback_coverage.csv"
        fallback_rows: list[dict[str, str]] = []
        if fallback_path.is_file():
            with fallback_path.open(encoding="utf-8", newline="") as handle:
                fallback_rows = list(csv.DictReader(handle))
        checks["retry_state_manifest_consistent"] = state_keys == manifest_keys
        checks["fallback_qualification_checked"] = all(row.get("qualification") in {"QUALIFIED", "MISSING"} for row in fallback_rows)
        manifest_by_key = {
            _retry_query_key(page.get("query", ""), page.get("retrieval_stage", "")): page
            for page in retry_pages if page.get("index") == "DBLP"
        }
        raw_paths_valid = True
        latest_hashes_match = True
        for entry in retry_state.get("queries", []):
            page = manifest_by_key.get(entry.get("query_key"))
            if page is None:
                raw_paths_valid = False
                continue
            raw_paths_valid = raw_paths_valid and (output_dir / str(page.get("raw_payload_path", ""))).is_file()
            attempts = sorted(entry.get("attempts", []), key=lambda attempt: int(attempt.get("round", 0)))
            if attempts:
                latest_hashes_match = latest_hashes_match and attempts[-1].get("raw_payload_sha256", "") == page.get("raw_payload_sha256", "")
        checks["retry_raw_paths_valid"] = raw_paths_valid
        checks["retry_latest_hashes_match"] = latest_hashes_match
        state_attempts_valid = all(
            len({int(attempt.get("round", 0)) for attempt in entry.get("attempts", [])}) == len(entry.get("attempts", []))
            and len(entry.get("attempts", [])) <= int(retry_state.get("max_rounds", 3))
            for entry in retry_state.get("queries", [])
        )
        checks["retry_attempts_at_most_one_per_round"] = state_attempts_valid
        checks["retry_batch_size_valid"] = 1 <= int(retry_state.get("batch_size", 0)) <= 3
        derived_retry_closure = assess_dblp_retry_closure(_retry_cutoff_from_state(retry_state), fallback_rows)
        checks["retry_closure_status"] = str(derived_retry_closure.get("status", "OPEN"))
        checks["retry_closure_status_valid"] = checks["retry_closure_status"] in {"OPEN", "COMPLETE", "CLOSED_WITH_FALLBACK", "CLOSED_BLOCKED"}
    checks_path = output_dir / "audit_checks.json"
    _write_json(checks_path, checks)
    paths["audit_checks.json"] = checks_path
    return paths


def run_self_test(repo_root: Path) -> dict[str, Any]:
    """Run deterministic CPU-only fixtures used by CI and reviewers."""

    checks: dict[str, bool] = {}
    protocol = load_protocol(repo_root)
    checks["protocol_valid"] = not validate_protocol(protocol)
    duplicate_rows = [
        normalize_record({"title": "A", "year": 2022, "doi": "10.1/X", "source_index": "DBLP", "peer_reviewed": True, "language": "English"}),
        normalize_record({"title": "A.", "year": "2022", "doi": "https://doi.org/10.1/x", "source_index": "Crossref", "peer_reviewed": True, "language": "English"}),
    ]
    unique, decisions = deduplicate_records(duplicate_rows)
    checks["doi_precedence"] = len(unique) == 1 and decisions[0]["reason"] == "DOI"
    checks["screening_reason_codes"] = "E02" in screen_record(normalize_record({"title": "preprint", "year": 2024, "source_index": "OpenAlex", "peer_reviewed": False, "language": "English"}))["reason_codes"]
    checks["missing_full_text_blocks"] = assess_c1_verdict([{"record_id": "r", "c1_relevance": "DIRECT", "overlap_class": "DIRECT-FUNCTIONAL", "full_text_status": "MISSING"}], [])["verdict"] == "UNRESOLVED"
    checks["strong_component_retain"] = assess_c1_verdict([{"record_id": "r", "c1_relevance": "STRONG-COMPONENT", "overlap_class": "STRONG-COMPONENT", "full_text_status": "LOCATED"}], [])["verdict"] == "RETAIN"
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        first_paths = write_outputs(repo_root, Path(first))
        second_paths = write_outputs(repo_root, Path(second))
        first_hashes = {name: _sha256_bytes(path.read_bytes()) for name, path in sorted(first_paths.items())}
        second_hashes = {name: _sha256_bytes(path.read_bytes()) for name, path in sorted(second_paths.items())}
        checks["byte_deterministic"] = first_hashes == second_hashes
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"passed": not failed, "checks_failed": failed, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--raw-snapshot", type=Path, default=None)
    parser.add_argument("--adjudication-file", type=Path, default=None)
    parser.add_argument("--fetch-fixed-queries", action="store_true")
    retry_mode = parser.add_mutually_exclusive_group()
    retry_mode.add_argument("--retry-dblp-status", action="store_true")
    retry_mode.add_argument("--retry-dblp-next", action="store_true")
    parser.add_argument("--retry-dblp-batch", action="store_true")
    parser.add_argument("--retry-round", type=int, default=None)
    parser.add_argument("--retry-batch-index", type=int, default=None)
    parser.add_argument("--retry-batch-size", type=int, default=None)
    parser.add_argument("--honor-retry-wait", action="store_true")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--retrieved-at", default="")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.self_test:
        result = run_self_test(repo_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    output_dir = args.output_dir or (repo_root / DEFAULT_OUTPUT_RELATIVE)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if args.retry_dblp_status:
        print(json.dumps(dblp_retry_status(output_dir), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.retry_dblp_next:
        result = retry_dblp_next(output_dir, repo_root=repo_root, rows=args.rows)
        printable = dict(result)
        if isinstance(printable.get("paths"), dict):
            printable["paths"] = {key: str(value) for key, value in printable["paths"].items()}
        print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.fetch_fixed_queries:
        protocol = load_protocol(repo_root)
        pages = fetch_protocol_queries(
            protocol,
            rows=args.rows,
            retrieved_at=args.retrieved_at,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        retrieval_paths = write_retrieval_snapshot(output_dir, pages)
        raw_records = [record for page in pages for record in page["records"]]
        adjudication_errors: list[str] = []
        if args.adjudication_file:
            adjudications = json.loads(args.adjudication_file.read_text(encoding="utf-8"))
            if not isinstance(adjudications, list):
                raise ValueError("adjudication file must be a JSON list")
            raw_records, adjudication_errors = apply_adjudications(raw_records, adjudications)
        retrieval_blockers = _retrieval_blockers_from_output(output_dir)
        derived_paths = write_outputs(repo_root, output_dir, raw_records, retrieval_blockers, adjudication_errors)
        paths = {**retrieval_paths, **derived_paths}
        print(json.dumps({key: str(value) for key, value in sorted(paths.items())}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.retry_dblp_batch or args.retry_round is not None or args.retry_batch_index is not None or args.retry_batch_size is not None:
        raise ValueError("legacy DBLP retry flags are deprecated; use --retry-dblp-status or --retry-dblp-next with dblp_retry_state.json")
    raw_records = _load_raw_snapshot(args.raw_snapshot)
    adjudication_errors = []
    if args.adjudication_file:
        adjudications = json.loads(args.adjudication_file.read_text(encoding="utf-8"))
        if not isinstance(adjudications, list):
            raise ValueError("adjudication file must be a JSON list")
        raw_records, adjudication_errors = apply_adjudications(raw_records, adjudications)
    paths = write_outputs(
        repo_root,
        output_dir,
        raw_records,
        extra_blockers=_retrieval_blockers_from_output(output_dir),
        adjudication_errors=adjudication_errors,
    )
    print(json.dumps({key: str(value) for key, value in sorted(paths.items())}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
