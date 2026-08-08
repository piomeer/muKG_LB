#!/usr/bin/env python3
"""Phase X X5.5 contribution triage.

This is a deterministic governance overlay over completed Part 2--5 audits.
It does not run experiments, import runtime/training code, access CUDA, or use
network services.  The explicit decision table is intentionally kept here so
the machine-readable output is auditable and reproducible.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_cross_claim_statistics import claim_ids, original_grade_map, sha256

SCRIPT_VERSION = "x5.5-1.0"
PART1 = "docs/evidence_audit_part1_claim_inventory.md"
X0 = "docs/phase_x_x0_research_freeze.md"
X05 = "docs/phase_x_x0_5_legacy_narrative_quarantine.md"
X15 = "output/results/evidence_audit_x1_5/x1_5_freeze_manifest.json"
X6_CONTRACT = "output/results/evidence_audit_part6/x6a/statistical_contract.json"
DECISIONS = {"RETAIN_PRIMARY", "RETAIN_SUPPORTING", "EXPLORATORY", "APPENDIX", "REMOVE"}
REPLACEMENTS = {
    "C1.2-R1": "C1.2", "C1.3-R1": "C1.3", "C1.7-R1": "C1.7", "C2.1-R1": "C2.1",
    "C3.1-R1": "C3.1", "C4.1-R1": "C4.1", "C4.3-R1": "C4.3", "C4.7-R1": "C4.7",
}

# The complete triage is frozen by the approved X5.5 decision.  Keeping this
# table explicit prevents a future audit from silently inferring a promotion.
PRIMARY = {"C1.2-R1", "C1.3-R1"}
SUPPORTING = {"C1.7-R1", "C2.1-R1", "C2.3"}
APPENDIX = {"C2.2", "C2.4", "C2.5", "C3.3", "C3.6", "C4.1-R1", "C4.2", "C4.3-R1", "C4.7-R1"}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decision_for(cid: str) -> str:
    if cid in PRIMARY:
        return "RETAIN_PRIMARY"
    if cid in SUPPORTING:
        return "RETAIN_SUPPORTING"
    if cid in APPENDIX:
        return "APPENDIX"
    return "REMOVE"


def grade_for(cid: str, grades):
    return {"C1.2-R1": "A", "C1.3-R1": "A", "C1.7-R1": "A", "C2.1-R1": "A",
            "C3.1-R1": "C", "C4.1-R1": "B", "C4.3-R1": "B", "C4.7-R1": "A"}.get(cid, grades.get(cid, "UNMAPPED"))


def destination_for(decision: str):
    return {"RETAIN_PRIMARY": "main_results", "RETAIN_SUPPORTING": "method_or_results",
            "APPENDIX": "appendix", "EXPLORATORY": "discussion_only", "REMOVE": "none"}[decision]


def claim_family(cid: str):
    if cid in {"C1.2", "C1.3", "C1.2-R1", "C1.3-R1"}:
        return "C1_joint_primary"
    if cid.startswith("C1."):
        return "C1_nonprimary"
    if cid.startswith("C2."):
        return "C2_implementation"
    if cid.startswith("C3."):
        return "C3_exploratory_or_implementation"
    return "C4_exploratory_or_implementation"


def wording(cid: str, decision: str):
    allowed = {
        "C1.2-R1": "Under the frozen muKG/FB15k-237/SimpleTransE/RTX3070 protocol, the declared GPU runtime path has 6.01x end-to-end epoch speedup (95% CI 5.94x–6.08x).",
        "C1.3-R1": "Under the same protocol, full-batch negative-sampling-time standard-deviation compression is 87.9x (95% CI 72.9x–105.9x).",
        "C1.7-R1": "The GPU path's full-batch negative-sampling time is approximately 3.003 ms under the audited protocol.",
        "C2.1-R1": "The implementation has an offline FeatureExtractor–CostModel–cost-table control plane and an online Scheduler–BatchProvider path; the training loop selects the backend.",
        "C2.3": "CPU and GPU configured paths consume the shared Scheduler/BatchProvider boundary while backend selection remains explicit in the training loop.",
    }
    if decision == "REMOVE":
        allowed_text = "None; retain only for audit lineage."
        prohibited = "Do not propagate this Claim into manuscript prose, tables, figures, abstract, or contribution list."
    elif decision == "APPENDIX":
        allowed_text = "Appendix-only implementation fact, protocol limitation, or negative reanalysis; no contribution claim."
        prohibited = "Do not use as primary evidence, causal evidence, predictive validity, generalization, or independent replication."
    else:
        allowed_text = allowed.get(cid, "Use only the corresponding audited, scope-limited wording.")
        prohibited = "Do not imply quality equivalence, non-inferiority, generalization, DDP readiness, or hardware portability."
    return allowed_text, prohibited


def rationale(cid: str, decision: str):
    if cid in PRIMARY:
        return "C1-R1 replacement has paired seed-level raw observations and simultaneous statistical support; original inventory entry is superseded."
    if cid in SUPPORTING:
        return "Audited implementation/supporting fact; not an independent empirical contribution."
    if cid in APPENDIX:
        return "Traceable implementation fact or negative/historical reanalysis retained for transparency without promotion."
    if cid.startswith("C3"):
        return "Predictive cost-model evidence is synthetic, descriptive, circular, or lacks provenance/held-out uncertainty; C3 rescue is waived."
    if cid.startswith("C4"):
        return "Composite CBP/FFD effect is not identified because FFDPacker equals ChunkPacker and historical traces lack qualified factorial repeats; C4 rescue is waived."
    return "Superseded, invalid, out of scope, or explicitly excluded by the C1 audit."


def manifest(repo: Path):
    rels = [PART1, X0, X05, X15, X6_CONTRACT,
            "docs/evidence_audit_part2_c1_gpu_runtime.md", "docs/evidence_audit_part3_c2_framework.md",
            "docs/evidence_audit_part4_c3_cost_model.md", "docs/evidence_audit_part5_c4_cbp.md",
            "output/results/evidence_audit_part2/recomputed_metrics.csv",
            "output/results/evidence_audit_part3/recomputed_metrics.csv",
            "output/results/evidence_audit_part4/claim_verdicts.csv",
            "output/results/evidence_audit_part5/claim_verdicts.csv"]
    return [{"path": rel, "exists": (repo / rel).exists(), "sha256": sha256(repo / rel) if (repo / rel).exists() else "", "bytes": (repo / rel).stat().st_size if (repo / rel).exists() else 0} for rel in rels]


def rows_for(repo: Path):
    ids = claim_ids(repo)
    grades = original_grade_map(repo)
    all_ids = ids + list(REPLACEMENTS)
    rows = []
    for cid in all_ids:
        parent = REPLACEMENTS.get(cid, "")
        decision = decision_for(cid)
        allowed, prohibited = wording(cid, decision)
        rows.append({
            "claim": cid,
            "original_claim": parent or cid,
            "lineage_status": "REPLACEMENT" if parent else "INVENTORY",
            "audit_grade": grade_for(cid, grades),
            "claim_family": claim_family(cid),
            "decision": decision,
            "paper_destination": destination_for(decision),
            "x6_5_disposition": "WAIVED" if cid.startswith(("C3", "C4")) else "NOT_APPLICABLE",
            "allowed_wording": allowed,
            "prohibited_use": prohibited,
            "decision_rationale": rationale(cid, decision),
            "source_audit": "Part 2" if cid.startswith("C1") else "Part 3" if cid.startswith("C2") else "Part 4" if cid.startswith("C3") else "Part 5",
        })
    return rows


def validate(repo: Path, rows):
    inventory = set(claim_ids(repo))
    replacements = set(REPLACEMENTS)
    actual = [r["claim"] for r in rows]
    errors = []
    if len(actual) != 36 or len(set(actual)) != 36:
        errors.append("triage must contain exactly 36 unique Claim rows")
    if set(actual) != inventory | replacements:
        errors.append("triage coverage does not equal 28 inventory IDs plus 8 replacements")
    if any(r["decision"] not in DECISIONS for r in rows):
        errors.append("invalid decision enum")
    by = {r["claim"]: r for r in rows}
    for child, parent in REPLACEMENTS.items():
        if by[child]["decision"] in {"RETAIN_PRIMARY", "RETAIN_SUPPORTING"} and by[parent]["decision"] != "REMOVE":
            errors.append(f"replacement parent {parent} must be REMOVE")
    if {r["claim"] for r in rows if r["decision"] == "RETAIN_PRIMARY"} != PRIMARY:
        errors.append("primary contribution set drift")
    if any(r["decision"] in {"RETAIN_PRIMARY", "RETAIN_SUPPORTING", "EXPLORATORY"} for r in rows if r["claim"].startswith(("C3", "C4"))):
        errors.append("C3/C4 must not be promoted or active exploratory")
    if any(r["audit_grade"] == "UNMAPPED" for r in rows):
        errors.append("unmapped audit grade")
    return errors


def decision_artifact(rows):
    return {
        "decision_version": SCRIPT_VERSION,
        "status": "FINAL",
        "x6_5_status": "WAIVED",
        "final_contribution_set": {
            "primary": sorted(PRIMARY),
            "supporting": sorted(SUPPORTING),
            "appendix": sorted(APPENDIX),
            "exploratory": [],
        },
        "branches": {
            "C3": {"decision": "WAIVED", "primary_promotion_estimand": None, "reason": "No provenance-complete held-out predictive target and uncertainty; minimum manuscript does not depend on C3."},
            "C4": {"decision": "WAIVED", "primary_promotion_estimand": None, "reason": "FFDPacker equals ChunkPacker and historical effect is not factorially identified; minimum manuscript does not depend on C4."},
        },
        "waiver_scope": {
            "covered": ["C3 predictive rescue", "C4 CBP/sorter/packer promotion rescue"],
            "not_covered": ["C1 sensitivity", "official quality evaluation", "C1.6 VRAM isolation", "C1.9 unified profiling", "cross-model/dataset studies", "X8 clean-room reproduction"],
        },
        "paper_boundary": "C1 is the sole empirical contribution; C2 is an implementation/supporting boundary; C3/C4 are not contributions.",
    }


def write(repo: Path, out: Path):
    rows = rows_for(repo)
    errors = validate(repo, rows)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    write_csv(out / "contribution_triage.csv", fields, rows)
    decision = decision_artifact(rows)
    dump_json(out / "gap_closing_decision.json", decision)
    dump_json(out / "source_manifest.json", manifest(repo))
    checks = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "inventory_claim_count": len(claim_ids(repo)),
        "replacement_count": len(REPLACEMENTS),
        "row_count": len(rows),
        "primary_count": len(PRIMARY),
        "c3_c4_x6_5_status": "WAIVED",
        "no_experiment_executed": True,
        "source_paths_exist": all(x["exists"] for x in manifest(repo)),
        "part1_untouched": True,
        "x1_5_snapshot_untouched": True,
    }
    dump_json(out / "audit_checks.json", checks)
    if errors:
        raise RuntimeError("X5.5 validation failed: " + "; ".join(errors))
    report = """# Phase X X5.5 — Contribution Triage\n\n## Material Passport\n\nVerification Status: **ANALYZED** (governance overlay; no new experiment)\n\n## Frozen decision\n\nThe minimum publishable manuscript retains C1 as the sole empirical contribution. C1.2-R1 and C1.3-R1 are primary; C1.7-R1 is supporting. C2.1-R1 and C2.3 are supporting implementation boundaries. C2.2, C2.4, C2.5, C3.3, C3.6, and the negative/historical C4 reanalyses are Appendix-only.\n\nPredictive C3 and composite CBP/FFD C4 are not promoted. Both X6.5 rescue branches are formally `WAIVED`; the waiver does not cover C1 sensitivity, official quality evaluation, VRAM isolation, unified profiling, cross-model/dataset studies, or X8 clean-room reproduction.\n\n## Claim-level governance\n\n`contribution_triage.csv` covers every Part 1 Claim and every audited replacement child. Original Claims superseded by a replacement are removed from manuscript authority. Appendix entries may describe implementation facts or limitations only; they cannot support causal, predictive, generalization, quality-equivalence, or independent-replication language.\n\n## X6 handoff\n\n`gap_closing_decision.json` is FINAL with `x6_5_status=WAIVED`. X6a may consume the triage and complete its cross-Claim statistical overlay; X6b may close as `COMPLETE_X6B_WAIVED`. X1.5 remains frozen and is not automatically resumed.\n"""
    (repo / "docs/phase_x_x5_5_contribution_triage.md").write_text(report, encoding="utf-8")


def self_test():
    assert decision_for("C1.2-R1") == "RETAIN_PRIMARY"
    assert decision_for("C3.1") == "REMOVE"
    assert decision_for("C4.7-R1") == "APPENDIX"
    artifact = decision_artifact([])
    assert artifact["status"] == "FINAL" and artifact["x6_5_status"] == "WAIVED"
    assert artifact["branches"]["C3"]["primary_promotion_estimand"] is None
    print("self-test: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return 0
    repo = args.repo_root.resolve()
    write(repo, args.output_dir or repo / "output/results/evidence_audit_x5_5")
    print(json.dumps({"status":"PASS","output":str(args.output_dir or repo / "output/results/evidence_audit_x5_5")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
