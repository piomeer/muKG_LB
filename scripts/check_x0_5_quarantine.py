from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


HEADER_MARKER = "<!-- LEGACY-NON-AUTHORITATIVE: Phase X X0.5 -->"
REGISTER_PATH = "docs/phase_x_x0_5_legacy_narrative_quarantine.md"
PART1_PATH = "docs/evidence_audit_part1_claim_inventory.md"
PART1_SHA256 = "93dc4b0b6c363bc98e266449010436528c701988caaf5b6e3437255a407cb7a6"

LEGACY_FILES = (
    "paper/draft/method.md",
    "docs/paper_outline.md",
    "docs/paper_story_freeze.md",
    "docs/runtime_framework_spec.md",
    "docs/phase8_architecture_freeze.md",
)

SAFE_SOURCE_PATHS = (
    "docs/phase_x_x0_research_freeze.md",
    "docs/evidence_audit_part2_c1_gpu_runtime.md",
    "docs/evidence_audit_part3_c2_framework.md",
    "docs/evidence_audit_part4_c3_cost_model.md",
    "docs/evidence_audit_part5_c4_cbp.md",
    "docs/unified_runtime_architecture_freeze.md",
    "output/results/evidence_audit_part2",
    "output/results/evidence_audit_part3",
    "output/results/c1_r1_combined_rerun",
)

DATA_SOURCE_POLICY = {
    "output/results/c1_r1_combined_rerun": (
        "CURRENT-EVIDENCE",
        "C1.2, C1.3",
        "New C1 manuscript analysis and audited C1-R1 observations",
        "Part 2 C1 audit",
    ),
    "output/results/evidence_audit_part2": (
        "CURRENT-EVIDENCE",
        "C1.1-C1.9 lineage",
        "Audited C1 derived artifacts",
        "Part 2 C1 audit",
    ),
    "output/results/evidence_audit_part3": (
        "CURRENT-EVIDENCE",
        "C2.1-C2.6 lineage",
        "Audited C2 derived artifacts",
        "Part 3 C2 audit",
    ),
    "output/results/evidence_audit_part4": (
        "CURRENT-EVIDENCE",
        "C3.1-C3.6",
        "Audited C3 wording and derived artifacts",
        "Part 4 C3 audit",
    ),
    "output/results/evidence_audit_part5": (
        "CURRENT-EVIDENCE",
        "C4.1-C4.7",
        "Audited C4 wording and derived artifacts",
        "Part 5 C4 audit",
    ),
    "output/results/unified_runtime": (
        "LEGACY-DATA / AUDIT-ONLY",
        "C1.1, C1.4, C2.3",
        "Historical auditing only",
        "Part 2 and this register",
    ),
    "output/results/phase9_step2": (
        "LEGACY-DATA / AUDIT-ONLY",
        "C1.2, C1.8, C2.2, C4.5, C4.6",
        "Historical auditing and lineage repair only",
        "Part 2 and Part 7 propagation",
    ),
    "output/results/phase9_step3": (
        "LEGACY-DATA / AUDIT-ONLY",
        "C1.3, C1.7, C4.4",
        "Historical auditing and lineage repair only",
        "Part 2 and Part 7 propagation",
    ),
    "output/results/runtime_attribution": (
        "LEGACY-DATA / AUDIT-ONLY",
        "C1.9, C3.2, C4.1",
        "Part 4/5 historical auditing only",
        "Part 4 and Part 5 audits",
    ),
    "output/results/phase9_step4_5": (
        "LEGACY-DATA / AUDIT-ONLY",
        "C4.3",
        "Part 5 historical auditing only; rounded single-process trace",
        "Part 5 audit",
    ),
    "output/results/integration_validation": (
        "LEGACY-DATA / AUDIT-ONLY",
        "C2.3, C2.5, C4.7",
        "Part 3/5 historical auditing only",
        "Part 3 and Part 5 audits",
    ),
}

EXPECTED_CLAIM_MAP = {
    "Q-01": {"C1.1"},
    "Q-02": {"C1.4"},
    "Q-03": {"C1.2"},
    "Q-04": {"C1.3"},
    "Q-05": {"C3.1"},
    "Q-06": {"C2.4", "C2.6"},
    "Q-07": {"C4.1", "C4.3", "C4.4", "C4.7"},
    "Q-08": {"C2.1"},
    "Q-09": {"C1.5", "C1.8", "C4.5", "C4.6"},
    "Q-10": {"C1.6", "C1.9", "C2.5"},
    "Q-11": {"C2.3"},
}

EXPECTED_QUARANTINE_ROWS = {
    "Q-01": (
        "Q-01",
        "C1.1",
        "Historical 596ms to 3.0ms component speedup",
        "Method; story freeze; unified runtime traces",
        "PERMANENTLY-EXCLUDED",
        "Part 2 C1 audit",
        "Never release the historical number",
        "Part 2 closed; legacy retained for audit",
        "A new matched component/step claim requires a new Claim ID, a new Part 2 audit, and Part 7 propagation.",
    ),
    "Q-02": (
        "Q-02",
        "C1.4",
        "Historical 674ms to 79.7ms step-time speedup",
        "Method; Phase 8 trace data",
        "PERMANENTLY-EXCLUDED",
        "Part 2 C1 audit",
        "Never release the historical number",
        "Part 2 closed; legacy retained for audit",
        "A new matched component/step claim requires a new Claim ID, a new Part 2 audit, and Part 7 propagation.",
    ),
    "Q-03": (
        "Q-03",
        "C1.2",
        "Historical 25.1s to 4.4s epoch-time result",
        "Outline; story freeze; Phase 9 Step 2 data",
        "SUPERSEDED-PENDING-PROPAGATION",
        "Part 2 C1 audit; C1.2-R1/E1",
        "Old number is never released",
        "Part 7 pending",
        "Part 7 must replace it with C1.2-R1/E1 wording.",
    ),
    "Q-04": (
        "Q-04",
        "C1.3",
        "Historical 28.5ms to 0.2ms dispersion result",
        "Outline; story freeze; Phase 9 Step 3 data",
        "SUPERSEDED-PENDING-PROPAGATION",
        "Part 2 C1 audit; C1.3-R1/E2",
        "Old number is never released",
        "Part 7 pending",
        "Part 7 must replace it with C1.3-R1/E2 wording.",
    ),
    "Q-05": (
        "Q-05",
        "C3.1",
        "Cost-model R²=0.9008 efficacy narrative",
        "Method; story freeze; cost-model materials",
        "PREDICTIVE-RETRACTED-IMPLEMENTATION-ONLY",
        "Part 4 C3 audit",
        "Do not release predictive R² wording; implementation facts only",
        "Part 4 closed; X5.5 triage pending",
        "Release any predictive wording only after a new X6.5-approved out-of-sample Claim and Part 7 propagation.",
    ),
    "Q-06": (
        "Q-06",
        "C2.4, C2.6",
        "Negligible scheduler or runtime-overhead narrative",
        "Method; runtime specification; legacy scheduler results",
        "PARTIALLY-PROHIBITED",
        "Part 3 C2 audit",
        "Deterministic construction/O(1) lookup may use C2.4 wording now; do not claim negligible end-to-end overhead",
        "C2.4 A; C2.6 D",
        "An end-to-end negligible-overhead statement requires a new measured Claim graded A/B and Part 7 propagation; C2.6 remains D.",
    ),
    "Q-07": (
        "Q-07",
        "C4.1, C4.3, C4.4, C4.7",
        "CPU scheduler treatment/variance benefit narrative",
        "Method; story freeze; Phase 9 and attribution data",
        "COMPOSITE-FAIL-SORTER-CANDIDATE",
        "Part 5 C4 audit",
        "Do not release FFD/packing/CBP contribution wording",
        "Part 5 closed; X5.5 pending",
        "Release only after X5.5 selects a sorter-only or new CBP Claim, X6.5 supplies the approved evidence, and Part 7 propagates the wording.",
    ),
    "Q-08": (
        "Q-08",
        "C2.1",
        "Legacy four- or five-layer unified-runtime architecture",
        "Method; story freeze; runtime specification; Phase 8 freeze",
        "SUPERSEDED-PENDING-PROPAGATION",
        "Part 3 C2 audit; C2.1-R1; unified architecture freeze",
        "Use only the canonical implemented-layer wording",
        "Part 7 pending",
        "Part 7 replaces the old layer narrative with C2.1-R1; RuntimePolicy/GPUExecution remain future extensions.",
    ),
    "Q-09": (
        "Q-09",
        "C1.5, C1.8, C4.5, C4.6",
        "Quality preservation, convergence, or comparability narrative",
        "Method; outline; story freeze; Phase 9 Step 2 data",
        "PROHIBITED",
        "X0 quality boundary; Part 2 C1 audit",
        "Do not release quality claims",
        "New quality audit required",
        "Release requires a new full-convergence, valid official-test quality Claim audited at A/B and Part 7 propagation.",
    ),
    "Q-10": (
        "Q-10",
        "C1.6, C1.9, C2.5",
        "Sampler VRAM, bottleneck shift, DDP-ready, general, or SOTA narrative",
        "Method; outline; runtime specification; legacy profiling data",
        "PROHIBITED/OUT-OF-SCOPE",
        "X0 scope and external-validity boundaries",
        "Do not release any branch without its own evidence",
        "Separate audits required",
        "Sampler-only VRAM requires an isolated A/B Claim; bottleneck shift requires unified timing boundaries and A/B audit; DDP-ready requires a real distributed execution Claim; general/SOTA wording requires separately registered generalization/comparator evidence. Each branch also requires Part 7 propagation.",
    ),
    "Q-11": (
        "Q-11",
        "C2.3",
        "Transparent, automatic, or drop-in CPU-to-GPU backend wording",
        "Method; runtime specification; Phase 8 design",
        "PROHIBITED",
        "Part 3 C2 audit",
        "Current Part 7 wording must say the training loop explicitly selects the backend",
        "Interface implementation pending",
        "Transparent/automatic/drop-in backend wording requires a newly implemented and audited automatic-selection interface.",
    ),
}

MATRIX_HEADING = "## Quarantine Matrix"
REVERSE_INDEX_HEADING = "## Claim Reverse Index"
STATUS_HEADING = "## 4. Data Source Status Register"
DOCUMENT_STATUS_HEADING = "## 3. Document Status Register"
SAFE_SOURCES_HEADING = "## 1. Safe Writing Sources"
CONTROLLED_HEADINGS = (
    SAFE_SOURCES_HEADING,
    DOCUMENT_STATUS_HEADING,
    STATUS_HEADING,
    MATRIX_HEADING,
    REVERSE_INDEX_HEADING,
)
TABLE_BACKED_HEADINGS = (
    DOCUMENT_STATUS_HEADING,
    STATUS_HEADING,
    MATRIX_HEADING,
    REVERSE_INDEX_HEADING,
)
EXPECTED_SAFE_SOURCE_LINES = (
    "- docs/phase_x_x0_research_freeze.md — X0 authority for research questions,",
    "  scope, contribution hierarchy, and frozen estimands.",
    "- docs/evidence_audit_part2_c1_gpu_runtime.md and",
    "  output/results/evidence_audit_part2/ — Part 2 C1 audit and its derived",
    "  artifacts.",
    "- docs/evidence_audit_part3_c2_framework.md and",
    "  output/results/evidence_audit_part3/ — Part 3 C2 audit and its derived",
    "  artifacts.",
    "- docs/evidence_audit_part4_c3_cost_model.md and",
    "  output/results/evidence_audit_part4/ — Part 4 C3 audit and its derived",
    "  artifacts.",
    "- docs/evidence_audit_part5_c4_cbp.md and",
    "  output/results/evidence_audit_part5/ — Part 5 C4 audit and its derived",
    "  artifacts.",
    "- docs/unified_runtime_architecture_freeze.md — canonical",
    "  figure/interface boundary for the implemented framework.",
    "- output/results/c1_r1_combined_rerun/ — C1-R1 source for unrounded",
    "  performance observations.",
)
DOCUMENT_STATUS_COLUMNS = ("Status", "Path", "Writing role")
EXPECTED_DOCUMENT_STATUS = {
    "docs/phase_x_x0_research_freeze.md": (
        "CANONICAL",
        "Scope, hierarchy, and estimands",
    ),
    "docs/evidence_audit_part2_c1_gpu_runtime.md": (
        "CANONICAL",
        "C1 audited wording",
    ),
    "docs/evidence_audit_part3_c2_framework.md": (
        "CANONICAL",
        "C2 audited wording",
    ),
    "docs/unified_runtime_architecture_freeze.md": (
        "CANONICAL",
        "Implemented architecture and figure boundary",
    ),
    "docs/evidence_audit_part1_claim_inventory.md": (
        "FROZEN-INVENTORY",
        "Claim lineage only",
    ),
    "paper/draft/method.md": (
        "LEGACY-NON-AUTHORITATIVE",
        "Historical draft; do not source claims",
    ),
    "docs/paper_outline.md": (
        "LEGACY-NON-AUTHORITATIVE",
        "Historical outline; do not source claims",
    ),
    "docs/paper_story_freeze.md": (
        "LEGACY-NON-AUTHORITATIVE",
        "Historical narrative; do not source claims",
    ),
    "docs/runtime_framework_spec.md": (
        "LEGACY-NON-AUTHORITATIVE",
        "Historical design narrative; do not source claims",
    ),
    "docs/phase8_architecture_freeze.md": (
        "LEGACY-NON-AUTHORITATIVE",
        "Historical interface plan; do not source claims",
    ),
    "docs/baseline_freeze.md": (
        "LEGACY-NON-AUTHORITATIVE",
        "Historical baseline record; do not source claims",
    ),
    "docs/validation_plan.md": (
        "LEGACY-NON-AUTHORITATIVE",
        "Historical validation plan; do not source claims",
    ),
    "docs/evidence_matrix.md": (
        "LEGACY-NON-AUTHORITATIVE",
        "Historical evidence matrix; do not source claims",
    ),
}
DATA_STATUS_COLUMNS = (
    "Path",
    "Status",
    "Affected Claim IDs",
    "Allowed Use",
    "Superseded/Controlled By",
)
MATRIX_COLUMNS = (
    "Quarantine ID",
    "Affected Claim IDs",
    "Legacy Expression",
    "Affected Documents/Data",
    "Disposition",
    "Canonical Source",
    "Paper-Use Rule",
    "Audit Owner / State",
    "Explicit Release Condition",
)


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise ValueError("table row must start and end with a pipe")
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _is_separator(cells: Iterable[str]) -> bool:
    return all(cell and set(cell) <= {"-", ":"} and "-" in cell for cell in cells)


def parse_markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    """Return rows from the first pipe table after heading; reject ragged rows."""
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration as error:
        raise ValueError(f"missing heading: {heading}") from error

    header_index = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip().startswith("|")),
        None,
    )
    if header_index is None or header_index + 1 >= len(lines):
        raise ValueError(f"missing table after heading: {heading}")
    headers = _cells(lines[header_index])
    separator = _cells(lines[header_index + 1])
    if len(headers) != len(separator) or not _is_separator(separator):
        raise ValueError(f"invalid table separator after heading: {heading}")
    if len(set(headers)) != len(headers) or any(not header for header in headers):
        raise ValueError(f"invalid table headers after heading: {heading}")

    rows = []
    for line in lines[header_index + 2 :]:
        if not line.strip():
            break
        if not line.strip().startswith("|"):
            break
        values = _cells(line)
        if len(values) != len(headers):
            raise ValueError(f"ragged row after heading: {heading}")
        rows.append(dict(zip(headers, values, strict=True)))
    return rows


def _result(check_id: str, failures: list[str]) -> dict[str, object]:
    return {
        "check_id": check_id,
        "status": "PASS" if not failures else "FAIL",
        "detail": "OK" if not failures else "; ".join(sorted(failures)),
    }


def check_headers(repo_root: Path) -> dict[str, object]:
    """Require one marker in the entire file and require it within the first eight lines."""
    failures = []
    for relative_path in LEGACY_FILES:
        path = repo_root / relative_path
        if not path.is_file():
            failures.append(f"missing legacy file: {relative_path}")
            continue
        text = path.read_text(encoding="utf-8")
        if text.count(HEADER_MARKER) != 1:
            failures.append(f"marker count is not one: {relative_path}")
        elif HEADER_MARKER not in text.splitlines()[:8]:
            failures.append(f"marker is not in first eight lines: {relative_path}")
    return _result("headers", failures)


def check_part1_inventory(repo_root: Path) -> dict[str, object]:
    """Require the frozen Part 1 inventory to retain its canonical bytes."""
    path = repo_root / PART1_PATH
    if not path.is_file():
        return _result("part1_inventory", [f"missing frozen inventory: {PART1_PATH}"])
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != PART1_SHA256:
        return _result(
            "part1_inventory",
            [f"SHA-256 mismatch: {PART1_PATH}"],
        )
    return _result("part1_inventory", [])


def _claim_ids(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def check_matrix(register_text: str) -> dict[str, object]:
    """Require every Q-01..Q-11 nine-column row to match its frozen value."""
    failures = []
    try:
        rows = parse_markdown_table(register_text, MATRIX_HEADING)
    except ValueError as error:
        return _result("matrix", [str(error)])
    if not rows or tuple(rows[0]) != MATRIX_COLUMNS:
        failures.append("matrix columns do not match required nine-column header")
        return _result("matrix", failures)

    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        if any(not row[column] for column in MATRIX_COLUMNS):
            failures.append(f"blank matrix field: {row.get('Quarantine ID', '<unknown>')}")
        quarantine_id = row["Quarantine ID"]
        if quarantine_id in by_id:
            failures.append(f"duplicate quarantine ID: {quarantine_id}")
        by_id[quarantine_id] = row
    if set(by_id) != set(EXPECTED_CLAIM_MAP):
        failures.append("quarantine IDs do not exactly match Q-01 through Q-11")
    for quarantine_id, expected_claims in EXPECTED_CLAIM_MAP.items():
        row = by_id.get(quarantine_id)
        if row is not None and _claim_ids(row["Affected Claim IDs"]) != expected_claims:
            failures.append(f"claim mapping mismatch: {quarantine_id}")
        if row is not None:
            actual_row = tuple(row[column] for column in MATRIX_COLUMNS)
            if actual_row != EXPECTED_QUARANTINE_ROWS[quarantine_id]:
                failures.append(f"quarantine row mismatch: {quarantine_id}")
    return _result("matrix", failures)


def check_reverse_index(register_text: str) -> dict[str, object]:
    """Require every Claim in EXPECTED_CLAIM_MAP to map back to exactly one Q row."""
    failures = []
    try:
        rows = parse_markdown_table(register_text, REVERSE_INDEX_HEADING)
    except ValueError as error:
        return _result("reverse_index", [str(error)])
    if not rows or tuple(rows[0]) != ("Part 1 Claim ID", "Quarantine ID"):
        return _result("reverse_index", ["reverse-index columns do not match required header"])

    reverse: dict[str, list[str]] = {}
    for row in rows:
        claim_id = row["Part 1 Claim ID"]
        quarantine_id = row["Quarantine ID"]
        if not claim_id or not quarantine_id:
            failures.append("blank reverse-index field")
            continue
        reverse.setdefault(claim_id, []).append(quarantine_id)
    expected_reverse = {
        claim_id: quarantine_id
        for quarantine_id, claim_ids in EXPECTED_CLAIM_MAP.items()
        for claim_id in claim_ids
    }
    if set(reverse) != set(expected_reverse):
        failures.append("reverse-index claim IDs do not exactly match expected claims")
    for claim_id, quarantine_id in expected_reverse.items():
        if reverse.get(claim_id) != [quarantine_id]:
            failures.append(f"reverse-index mapping mismatch: {claim_id}")
    return _result("reverse_index", failures)


def _section_text(text: str, heading: str) -> str:
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration as error:
        raise ValueError(f"missing heading: {heading}") from error
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end])


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def check_controlled_sections(register_text: str) -> dict[str, object]:
    """Require one instance of each controlled heading and table."""
    lines = register_text.splitlines()
    failures = []
    for heading in CONTROLLED_HEADINGS:
        if sum(line.strip() == heading for line in lines) != 1:
            failures.append(f"controlled heading must occur exactly once: {heading}")
    for heading in TABLE_BACKED_HEADINGS:
        try:
            section_lines = _section_text(register_text, heading).splitlines()
        except ValueError as error:
            failures.append(str(error))
            continue
        table_count = 0
        for header, separator in zip(section_lines, section_lines[1:]):
            header_cells = _table_cells(header)
            separator_cells = _table_cells(separator)
            if (
                len(header_cells) >= 2
                and len(header_cells) == len(separator_cells)
                and _is_separator(separator_cells)
            ):
                table_count += 1
        if table_count != 1:
            failures.append(f"controlled section must contain exactly one table: {heading}")
    return _result("controlled_sections", failures)


def check_safe_writing_sources(register_text: str) -> dict[str, object]:
    """Require every allowlist bullet to match the frozen grouped representation."""
    try:
        section = _section_text(register_text, SAFE_SOURCES_HEADING)
    except ValueError as error:
        return _result("safe_writing_sources", [str(error)])

    section_lines = section.splitlines()
    failures = []
    list_marker = re.compile(r"^([ \t]*)([-+*])[ \t]+")
    marker_lines = []
    for line in section_lines:
        marker = list_marker.match(line)
        if marker:
            marker_lines.append(line)
            if marker.group(1) or marker.group(2) != "-":
                failures.append("unexpected or nested safe-source list marker")

    expected_markers = tuple(
        line for line in EXPECTED_SAFE_SOURCE_LINES if line.startswith("- ")
    )
    if tuple(marker_lines) != expected_markers:
        failures.append("safe writing source list markers do not exactly match approved entries")

    first_bullet = next(
        (index for index, line in enumerate(section_lines) if line.startswith("- ")),
        None,
    )
    source_lines = []
    if first_bullet is not None:
        for line in section_lines[first_bullet:]:
            if not line:
                break
            source_lines.append(line)
    if tuple(source_lines) != EXPECTED_SAFE_SOURCE_LINES:
        failures.append("safe writing source bullets do not exactly match approved entries")
    return _result("safe_writing_sources", failures)


def check_document_status_register(register_text: str) -> dict[str, object]:
    """Require the exact document-status path, status, and writing-role mapping."""
    failures = []
    try:
        section = _section_text(register_text, DOCUMENT_STATUS_HEADING)
        rows = parse_markdown_table(register_text, DOCUMENT_STATUS_HEADING)
    except ValueError as error:
        return _result("document_status_register", [str(error)])
    if not rows or tuple(rows[0]) != DOCUMENT_STATUS_COLUMNS:
        return _result(
            "document_status_register",
            ["document-status columns do not match required header"],
        )

    actual: dict[str, tuple[str, str]] = {}
    for row in rows:
        path = row["Path"]
        if any(not row[column] for column in DOCUMENT_STATUS_COLUMNS):
            failures.append("blank document-status field")
        elif path in actual:
            failures.append(f"duplicate document-status path: {path}")
        else:
            actual[path] = (row["Status"], row["Writing role"])
    if actual != EXPECTED_DOCUMENT_STATUS:
        failures.append("document-status entries do not exactly match declared policies")

    section_lines = section.splitlines()
    first_table_row = next(
        (index for index, line in enumerate(section_lines) if line.strip().startswith("|")),
        None,
    )
    if first_table_row is not None:
        after_table = first_table_row
        while (
            after_table < len(section_lines)
            and section_lines[after_table].strip().startswith("|")
        ):
            after_table += 1
        trailing_lines = section_lines[after_table:]
        has_table_like_row = any(line.count("|") >= 2 for line in trailing_lines)
        has_table_start = any(
            "|" in header
            and "|" in separator
            and len(header.split("|")) == len(separator.split("|"))
            and _is_separator(cell.strip() for cell in separator.split("|"))
            for header, separator in zip(trailing_lines, trailing_lines[1:])
        )
        if has_table_like_row or has_table_start:
            failures.append("unexpected table-like row after document-status table")
    return _result("document_status_register", failures)


def check_paths_and_statuses(repo_root: Path, register_text: str) -> dict[str, object]:
    """Require declared evidence paths, or absence when status is NOT-PRESENT."""
    failures = []
    for relative_path in SAFE_SOURCE_PATHS:
        if not (repo_root / relative_path).exists():
            failures.append(f"missing required path: {relative_path}")
    for relative_path, policy in DATA_SOURCE_POLICY.items():
        status = policy[0]
        path = repo_root / relative_path
        if status == "NOT-PRESENT":
            if path.exists():
                failures.append(f"path must be absent: {relative_path}")
        elif not path.exists():
            failures.append(f"missing required path: {relative_path}")
    try:
        rows = parse_markdown_table(register_text, STATUS_HEADING)
    except ValueError as error:
        return _result("paths_and_statuses", failures + [str(error)])
    if not rows or tuple(rows[0]) != DATA_STATUS_COLUMNS:
        return _result("paths_and_statuses", failures + ["status columns do not match required header"])
    policies: dict[str, tuple[str, str, str, str]] = {}
    for row in rows:
        path = row["Path"]
        if any(not row[column] for column in DATA_STATUS_COLUMNS):
            failures.append("blank data-status field")
        elif path in policies:
            failures.append(f"duplicate data-status path: {path}")
        else:
            policies[path] = tuple(row[column] for column in DATA_STATUS_COLUMNS[1:])
    if policies != DATA_SOURCE_POLICY:
        failures.append("data-source policies do not exactly match declared policies")
    return _result("paths_and_statuses", failures)


def build_checks(repo_root: Path) -> dict[str, object]:
    """Return sorted {check_id, status, detail} rows and overall PASS/FAIL."""
    register_path = repo_root / REGISTER_PATH
    if register_path.is_file():
        register_text = register_path.read_text(encoding="utf-8")
        register_checks = [
            check_controlled_sections(register_text),
            check_matrix(register_text),
            check_reverse_index(register_text),
            check_safe_writing_sources(register_text),
            check_document_status_register(register_text),
            check_paths_and_statuses(repo_root, register_text),
        ]
    else:
        register_checks = [_result("register", [f"missing register: {REGISTER_PATH}"])]
    checks = sorted(
        [check_headers(repo_root), check_part1_inventory(repo_root), *register_checks],
        key=lambda row: str(row["check_id"]),
    )
    return {
        "checks": checks,
        "overall_status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
    }


def _json_payload(result: dict[str, object]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_output(result: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_payload(result), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_checks(args.repo_root)
    if args.output is not None:
        write_output(result, args.output)
    print(_json_payload(result), end="")
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
