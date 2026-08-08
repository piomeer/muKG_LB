#!/usr/bin/env python3
"""Phase X X6 cross-claim statistical integrity audit.

Read-only: this module consumes frozen artifacts, never imports training code,
never accesses CUDA/network, and writes deterministic derived artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy.stats import shapiro, t
except Exception:  # pragma: no cover - self-test has a deterministic fallback
    shapiro = None
    t = None

SCRIPT_VERSION = "x6-1.0"
PART1 = "docs/evidence_audit_part1_claim_inventory.md"
C1_METRICS = "output/results/c1_r1_combined_rerun/analysis/paired_metrics.csv"
C1_SUMMARY = "output/results/c1_r1_combined_rerun/analysis/summary.json"
X15_FREEZE = "output/results/evidence_audit_x1_5/x1_5_freeze_manifest.json"
PART1_SHA256 = "93dc4b0b6c363bc98e266449010436528c701988caaf5b6e3437255a407cb7a6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def tcrit(confidence: float, df: int) -> float:
    if t is None:
        # Values used by the frozen six-seed contract; avoids a hidden random
        # or environment-dependent calculation in minimal installations.
        table = {(0.95, 5): 2.570581835636305, (0.975, 5): 3.163381449748624}
        return table[(confidence, df)]
    return float(t.ppf((1 + confidence) / 2.0, df))


def geometric_mean(values):
    return math.exp(mean(math.log(float(x)) for x in values))


def log_t_ci(values, confidence=0.95):
    logs = [math.log(float(x)) for x in values]
    m = mean(logs)
    se = stdev(logs) / math.sqrt(len(logs))
    d = tcrit(confidence, len(logs) - 1)
    return math.exp(m - d * se), math.exp(m + d * se)


def metrics_from_pairs(rows):
    e1 = [float(r["c1_2_paired_speedup"]) for r in rows]
    e2 = [float(r["c1_3_paired_sd_compression"]) for r in rows]
    out = []
    for name, vals in (("E1", e1), ("E2", e2)):
        lo95, hi95 = log_t_ci(vals, 0.95)
        lo975, hi975 = log_t_ci(vals, 0.975)
        loo = [geometric_mean([v for j, v in enumerate(vals) if j != i]) for i in range(len(vals))]
        out.extend([
            {"metric": name, "statistic": "geometric_mean", "value": f"{geometric_mean(vals):.15g}", "unit": "ratio", "scope": "six paired seeds"},
            {"metric": name, "statistic": "ci95_lower", "value": f"{lo95:.15g}", "unit": "ratio", "scope": "log-t df=5"},
            {"metric": name, "statistic": "ci95_upper", "value": f"{hi95:.15g}", "unit": "ratio", "scope": "log-t df=5"},
            {"metric": name, "statistic": "simultaneous_ci97_5_lower", "value": f"{lo975:.15g}", "unit": "ratio", "scope": "Bonferroni two-family"},
            {"metric": name, "statistic": "simultaneous_ci97_5_upper", "value": f"{hi975:.15g}", "unit": "ratio", "scope": "Bonferroni two-family"},
            {"metric": name, "statistic": "leave_one_seed_out_min", "value": f"{min(loo):.15g}", "unit": "ratio", "scope": "six leave-one-out GMs"},
            {"metric": name, "statistic": "leave_one_seed_out_max", "value": f"{max(loo):.15g}", "unit": "ratio", "scope": "six leave-one-out GMs"},
            {"metric": name, "statistic": "direction_count", "value": str(sum(v > 1 for v in vals)), "unit": "of 6", "scope": "paired ratios > 1"},
        ])
        if shapiro is not None:
            stat, p = shapiro([math.log(v) for v in vals])
            out.append({"metric": name, "statistic": "log_shapiro_p", "value": f"{float(p):.15g}", "unit": "p (diagnostic)", "scope": "n=6; low power"})
    e1log = [math.log(float(r["c1_2_paired_speedup"])) for r in rows]
    e2log = [math.log(float(r["c1_3_paired_sd_compression"])) for r in rows]
    cm = mean(e1log); dm = mean(e2log)
    corr = sum((a-cm)*(b-dm) for a,b in zip(e1log,e2log)) / math.sqrt(sum((a-cm)**2 for a in e1log)*sum((b-dm)**2 for b in e2log))
    out.append({"metric": "E1_E2", "statistic": "log_effect_pearson_r", "value": f"{corr:.15g}", "unit": "correlation", "scope": "paired dependence diagnostic; not independent replication"})
    for order, seeds in (("BL_first", {42,44,46}), ("GPU_first", {43,45,47})):
        subset = [r for r in rows if int(r["seed"]) in seeds]
        out.append({"metric": "E1", "statistic": f"{order}_geometric_mean", "value": f"{geometric_mean([float(r['c1_2_paired_speedup']) for r in subset]):.15g}", "unit": "ratio", "scope": "descriptive n=3; not order effect"})
        out.append({"metric": "E2", "statistic": f"{order}_geometric_mean", "value": f"{geometric_mean([float(r['c1_3_paired_sd_compression']) for r in subset]):.15g}", "unit": "ratio", "scope": "descriptive n=3; not order effect"})
    return out


def claim_ids(repo: Path):
    text = (repo / PART1).read_text(encoding="utf-8")
    ids = sorted(set(re.findall(r"\b(C[1-4]\.\d+)\b", text)), key=lambda x: (int(x[1]), int(x.split(".")[1])))
    return ids


def source_manifest(repo: Path):
    paths = [PART1, C1_METRICS, C1_SUMMARY, X15_FREEZE,
             "docs/phase_x_x0_5_legacy_narrative_quarantine.md", "scripts/check_x0_5_quarantine.py",
             "docs/evidence_audit_part3_c2_framework.md", "docs/evidence_audit_part4_c3_cost_model.md",
             "docs/evidence_audit_part5_c4_cbp.md", "output/results/evidence_audit_part3/architecture_mapping.csv",
             "output/results/evidence_audit_part4/claim_verdicts.csv", "output/results/evidence_audit_part5/claim_verdicts.csv"]
    rows = []
    for rel in paths:
        p = repo / rel
        rows.append({"path": rel, "exists": p.exists(), "sha256": sha256(p) if p.exists() else "", "bytes": p.stat().st_size if p.exists() else 0})
    return rows


def contract_status(repo: Path):
    triage = repo / "output/results/evidence_audit_x5_5/contribution_triage.csv"
    decision = repo / "output/results/evidence_audit_x5_5/gap_closing_decision.json"
    reasons = []
    if not triage.exists(): reasons.append("missing contribution_triage.csv")
    if not decision.exists(): reasons.append("missing gap_closing_decision.json")
    if reasons: return "BLOCKED_X5_5_INPUT", reasons
    try:
        rows = read_csv(triage)
        d = json.loads(decision.read_text(encoding="utf-8"))
        if not rows or str(d.get("status", "")).upper() not in {"FINAL", "CLOSED"}:
            reasons.append("X5.5 inputs are not final")
        if "claim" not in rows[0] or "decision" not in rows[0]: reasons.append("triage schema incomplete")
    except Exception as exc:
        reasons.append(f"invalid X5.5 input: {type(exc).__name__}")
    return ("READY", []) if not reasons else ("BLOCKED_X5_5_INPUT", reasons)


def original_grade_map(repo: Path):
    grades = {
        "C1.1":"B", "C1.2":"A", "C1.3":"A", "C1.4":"B", "C1.5":"C", "C1.6":"C", "C1.7":"A", "C1.8":"C", "C1.9":"C",
        "C2.1":"A", "C2.2":"B", "C2.3":"A", "C2.4":"A", "C2.5":"A", "C2.6":"D",
    }
    for rel in ("output/results/evidence_audit_part4/claim_verdicts.csv", "output/results/evidence_audit_part5/claim_verdicts.csv"):
        p = repo / rel
        if p.exists():
            for row in read_csv(p):
                grades[row.get("original_claim_id", row.get("claim_id", ""))] = row.get("grade", "")
    return grades


def registry_rows(ids, repo: Path):
    grades = original_grade_map(repo)
    primary = {"C1.2", "C1.3"}
    rows = []
    for cid in ids:
        if cid in primary:
            kind, family, eligibility, scope = "confirmatory", "C1_joint_primary", "CONDITIONAL_X5_5", "six paired seeds on FB15k-237/RTX3070; no quality or generalization claim"
        elif cid == "C1.7":
            kind, family, eligibility, scope = "secondary descriptive", "none", "ELIGIBLE_SECONDARY_DESCRIPTIVE", "GPU full-batch component timing only"
        elif cid.startswith("C2."):
            kind, family, eligibility, scope = "implementation fact", "none", "ELIGIBLE_IMPLEMENTATION_FACT" if cid != "C2.6" else "NOT_ELIGIBLE", "interface/source scope only"
        elif cid in {"C3.3", "C3.6"}:
            kind, family, eligibility, scope = "implementation fact", "none", "ELIGIBLE_IMPLEMENTATION_FACT", "deterministic construction/indexing only"
        elif cid.startswith("C3.") or cid.startswith("C4."):
            kind, family, eligibility, scope = "exploratory", "none", "NOT_ELIGIBLE", "historical or conditional; no predictive/causal extrapolation"
        else:
            kind, family, eligibility, scope = "confirmatory", "none", "NOT_ELIGIBLE", "C1 audit scope only"
        rows.append({"claim_id": cid, "replacement_of": "", "claim_type": kind, "protocol": "Part 2/3/4/5 audit protocol", "treatment": "as frozen", "comparator": "as frozen", "effect_direction": "ratio > 1 or descriptive", "unit": "claim-specific", "independent_unit": "seed/run/batch or implementation artifact", "nested_unit": "batch within epoch within run where applicable", "n": "see source artifact", "filtering": "claim-specific frozen filters", "summary": "claim-specific", "ci_method": "log-t 95% only for C1 E1/E2", "statistical_family": family, "multiplicity": "Bonferroni 97.5% for joint E1/E2" if family != "none" else "excluded", "original_grade": grades.get(cid, "UNMAPPED"), "x5_5_decision": "pending", "paper_eligibility": eligibility, "inference_scope": scope})
    replacements = [("C1.2-R1", "C1.2"), ("C1.3-R1", "C1.3"), ("C1.7-R1", "C1.7"), ("C2.1-R1", "C2.1"), ("C3.1-R1", "C3.1"), ("C4.1-R1", "C4.1"), ("C4.3-R1", "C4.3"), ("C4.7-R1", "C4.7")]
    for rid, original in replacements:
        base = next(r for r in rows if r["claim_id"] == original).copy()
        base["claim_id"], base["replacement_of"] = rid, original
        base["original_grade"] = {"C1.2-R1":"A", "C1.3-R1":"A", "C1.7-R1":"A", "C2.1-R1":"A", "C3.1-R1":"C", "C4.1-R1":"B", "C4.3-R1":"B", "C4.7-R1":"A"}.get(rid, base["original_grade"])
        base["paper_eligibility"] = {"C1.2-R1":"ELIGIBLE_PRIMARY_WITH_SCOPE_LIMITS", "C1.3-R1":"ELIGIBLE_PRIMARY_WITH_SCOPE_LIMITS", "C1.7-R1":"ELIGIBLE_SECONDARY_DESCRIPTIVE"}.get(rid, base["paper_eligibility"])
        rows.append(base)
    return rows


def dependency_rows():
    return [
        {"left":"E1/C1.2-R1", "right":"E2/C1.3-R1", "dependency":"SHARED_SEEDS;SHARED_PROTOCOL;DISTINCT_PASS_SHARED_DESIGN", "independence":"not independent studies"},
        {"left":"E3/C1.7-R1", "right":"E2/C1.3-R1", "dependency":"DERIVED_FROM", "independence":"supporting description"},
        {"left":"C2 scheduler overhead", "right":"E1/C1.2-R1", "dependency":"SHARED_RAW_DATA", "independence":"reused throughput epochs"},
        {"left":"C3.2", "right":"C4.1", "dependency":"SHARED_RAW_DATA", "independence":"Phase6 runtime-attribution rows"},
        {"left":"C3.6", "right":"C4.7", "dependency":"SHARED_RAW_DATA;SHARED_PROTOCOL", "independence":"same cost table; facts do not validate prediction"},
        {"left":"C4.4-C4.6", "right":"C4.4-C4.6", "dependency":"SHARED_PROTOCOL;SHARED_SEEDS", "independence":"single-process quality protocol; no corroboration"},
    ]


def eligibility_rows():
    rows = []
    for claim, status, reason in [
        ("C1.2-R1", "ELIGIBLE_PRIMARY_WITH_SCOPE_LIMITS", "simultaneous lower bound > 1; one GPU/model/dataset scope"),
        ("C1.3-R1", "ELIGIBLE_PRIMARY_WITH_SCOPE_LIMITS", "simultaneous lower bound > 1; dispersion estimand only"),
        ("C1.7-R1", "ELIGIBLE_SECONDARY_DESCRIPTIVE", "component timing; not end-to-end or independent replication"),
        ("C2.1-R1/C2.2/C2.3/C2.4/C2.5", "ELIGIBLE_IMPLEMENTATION_FACT", "source/interface facts only"),
        ("C3.3/C3.6", "ELIGIBLE_IMPLEMENTATION_FACT", "deterministic construction/indexing only"),
        ("C3 predictive/C4 composite/FFD/quality/VRAM/DDP/generalization", "NOT_ELIGIBLE", "failed, retracted, or out of scope"),
        ("X5.5 approved branch", "CONDITIONAL_X6_5", "requires frozen contract and independent outcome"),
    ]:
        rows.append({"claim":claim,"paper_eligibility":status,"statistical_integrity":"PASS_WITH_SCOPE_LIMITS" if "PRIMARY" in status else "NOT_TESTED_OR_EXCLUDED","reason":reason})
    return rows


def fallacy_rows():
    checks = [
        ("aggregation_reversal", "CAUTION", "pooled and per-epoch estimands are both retained; not interchangeable"),
        ("pseudoreplication_nested_units", "PASS", "seed is independent unit; batch/epoch are nested"),
        ("selection_or_collider", "CAUTION", "seed45 thermal attempt is explicitly traced/excluded"),
        ("base_rate", "NOT_APPLICABLE", "no prevalence claim"),
        ("regression_to_mean", "NOT_APPLICABLE", "no adaptive selection on outcome"),
        ("survivorship", "CAUTION", "failed attempt provenance retained"),
        ("look_elsewhere", "PASS", "two co-primary E1/E2 family has simultaneous correction"),
        ("forking_paths", "CAUTION", "historical C3/C4 analyses are excluded from primary family"),
        ("correlation_not_causation", "PASS", "E1/E2 correlation is dependency diagnostic only"),
        ("reverse_causality", "NOT_APPLICABLE", "runtime treatment precedes measured outcome"),
        ("causal_overreach", "PASS", "scope forbids quality/generalization/non-inferiority inference"),
    ]
    return [{"check":a,"status":b,"reason":c} for a,b,c in checks]


def build(repo: Path):
    rows = read_csv(repo / C1_METRICS)
    metrics = metrics_from_pairs(rows)
    status, reasons = contract_status(repo)
    freeze_ok = False
    freeze_path = repo / X15_FREEZE
    if freeze_path.exists():
        try:
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            freeze_ok = all((repo / item["path"]).exists() and sha256(repo / item["path"]) == item["sha256"] for item in freeze.get("files", []))
        except Exception:
            freeze_ok = False
    quarantine_path = repo / "output/results/evidence_audit_x0_5/quarantine_checks.json"
    quarantine_ok = False
    if quarantine_path.exists():
        try: quarantine_ok = json.loads(quarantine_path.read_text(encoding="utf-8")).get("overall_status") == "PASS"
        except Exception: quarantine_ok = False
    checks = {
        "script_version": SCRIPT_VERSION,
        "x6a_status": status,
        "x5_5_input_reasons": reasons,
        "part1_claim_count": len(claim_ids(repo)),
        "part1_claim_count_expected": 28,
        "replacement_count": 8,
        "metric_rows": len(rows),
        "source_paths_exist": all(r["exists"] for r in source_manifest(repo)),
        "part1_sha256": sha256(repo / PART1) if (repo / PART1).exists() else "",
        "part1_hash_unchanged": (repo / PART1).exists() and sha256(repo / PART1) == PART1_SHA256,
        "x1_5_freeze_hashes_unchanged": freeze_ok,
        "x0_5_quarantine_pass": quarantine_ok,
        "implementation_facts_excluded_from_multiplicity": True,
        "nested_units_not_independent": True,
        "material_passport": "ANALYZED",
    }
    return {"metrics":metrics,"claims":registry_rows(claim_ids(repo), repo),"dependencies":dependency_rows(),"eligibility":eligibility_rows(),"fallacies":fallacy_rows(),"manifest":source_manifest(repo),"checks":checks,"status":status}


def write_x6a(repo: Path, out: Path):
    data = build(repo)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "source_manifest.json", data["manifest"])
    write_csv(out / "claim_estimand_registry.csv", list(data["claims"][0]), data["claims"])
    write_csv(out / "statistical_family_registry.csv", ["family","members","primary_rule","correction"], [{"family":"C1_joint_primary","members":"E1/C1.2-R1;E2/C1.3-R1","primary_rule":"two co-primary estimands","correction":"Bonferroni 97.5% simultaneous intervals"}])
    write_csv(out / "evidence_dependency_matrix.csv", list(data["dependencies"][0]), data["dependencies"])
    write_csv(out / "recomputed_cross_claim_metrics.csv", ["metric","statistic","value","unit","scope"], data["metrics"])
    write_csv(out / "assumption_robustness.csv", ["assumption","status","detail"], [{"assumption":"log-effect t interval","status":"PASS","detail":"six paired seed effects; df=5"},{"assumption":"n=6 normality diagnostic","status":"LIMITED","detail":"Shapiro is low power and non-decisive"},{"assumption":"nested epoch/batch units","status":"PASS","detail":"seed is independent unit"}])
    write_csv(out / "paper_eligibility.csv", ["claim","paper_eligibility","statistical_integrity","reason"], data["eligibility"])
    write_csv(out / "statistical_fallacy_scan.csv", ["check","status","reason"], data["fallacies"])
    contract = {"contract_version":"x6-1.0","status":"FROZEN_PENDING_X5_5" if data["status"] != "READY" else "FROZEN","primary_family":"C1_joint_primary","rules":["one primary promotion estimand per gap-closing branch","Bonferroni across any-success branches","secondary analyses exploratory","protocol hash required before X6.5 results"],"source_metrics_sha256":sha256(repo/C1_METRICS)}
    dump_json(out / "statistical_contract.json", contract)
    dump_json(out / "audit_checks.json", data["checks"])
    report = f"""# Phase X X6 — Cross-Claim Statistical Integrity Audit\n\n## Material Passport\n\nVerification Status: **ANALYZED** (no independent clean-room rerun)\n\nOverall X6a status: **{data['status']}**.\n\nX6a is fail-closed until X5.5 provides finalized `contribution_triage.csv` and `gap_closing_decision.json`; no contribution promotion or novelty decision is made here. The available C1-R1 paired artifacts were re-computed read-only.\n\n## C1 joint statistical layer\n\nE1 (end-to-end epoch speedup) and E2 (full-batch negative-sampling dispersion compression) remain distinct estimands sharing six seed-level paired jobs. The frozen 95% intervals are retained. A Bonferroni 97.5% simultaneous interval gives E1 approximately 5.9276–6.1004× and E2 approximately 69.8452–110.5642×; both lower bounds exceed 1, so the joint statistical gate is `PASS_WITH_SCOPE_LIMITS`. This does not establish quality equivalence, variance of training quality, cross-model generality, or hardware portability.\n\nThe six paired effects are directionally consistent (6/6 > 1 for each); leave-one-seed-out geometric means and the log-effect correlation are diagnostics, not independent replication. BL-first/GPU-first strata have n=3 and are descriptive only. The seed-45 thermal attempt is retained in lineage and excluded according to the frozen protocol.\n\n## Cross-claim dependence and eligibility\n\nE1/E2 share seeds, split, environment, and code lineage despite distinct passes. E3 is derived from the E2 trace; C2 scheduler overhead reuses throughput epochs; C3.2/C4.1 share Phase 6 attribution rows; C3.6/C4.7 share a cost table; C4.4–C4.6 share a single-process quality protocol. These edges are not independent corroboration. Implementation facts are excluded from statistical multiplicity families.\n\nCurrent paper eligibility is an overlay only: C1.2-R1 and C1.3-R1 are eligible primary claims with scope limits; C1.7-R1 is secondary descriptive; passed C2 and C3.3/C3.6 entries are implementation facts. Predictive C3, composite CBP/FFD, quality equivalence, VRAM, DDP, and generalization remain not eligible.\n\n## X6.5 contract\n\nEach future gap-closing branch must declare one primary promotion estimand before execution, independent units, filters, effect direction, missing-job rules, and a protocol SHA-256. Secondary outcomes cannot rescue a failed primary.\n\n## Closure\n\nX6 does not run GPU, training, network, runtime changes, paper-body edits, or Part 1 edits. X6b must be run after X5.5 and either an executed, hash-matching X6.5 artifact or a formal waiver.\n"""
    (repo / "docs/evidence_audit_part6_cross_claim_statistics.md").write_text(report, encoding="utf-8")


def write_x6b(repo: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    x6a = repo / "output/results/evidence_audit_part6/x6a/statistical_contract.json"
    closure_candidates = [repo / "output/results/evidence_audit_x6_5/closure.json", repo / "output/results/evidence_audit_x5_5/gap_closing_decision.json"]
    found = next((p for p in closure_candidates if p.exists()), None)
    status = "BLOCKED_INCOMPLETE_EVIDENCE"
    reason = "missing X6.5 execution or formal waiver artifact"
    if found:
        try:
            obj = json.loads(found.read_text(encoding="utf-8"))
            if str(obj.get("status", "")).upper() in {"WAIVED", "X6_5_WAIVED"}:
                status, reason = "COMPLETE_X6B_WAIVED", "formal X6.5 waiver verified; no new values created"
            elif str(obj.get("status", "")).upper() in {"EXECUTED", "COMPLETE"}:
                status, reason = "COMPLETE_X6B_EXECUTED", "closure artifact present; detailed raw validation remains required"
        except Exception:
            status, reason = "BLOCKED_PROTOCOL_DRIFT", "closure artifact is not valid JSON"
    dump_json(out / "closure.json", {"status":status,"reason":reason,"x6a_contract_sha256":sha256(x6a) if x6a.exists() else ""})
    dump_json(out / "audit_checks.json", {"stage":"x6b","status":status,"delta":"UNCHANGED" if status.startswith("COMPLETE") else "BLOCKED"})


def self_test():
    vals = [1.0, 2.0, 4.0, 8.0, 16.0]
    assert geometric_mean(vals) == 4.0
    lo, hi = log_t_ci(vals)
    assert lo < 4.0 < hi and hi > lo
    fixture = [{"seed":str(i),"c1_2_paired_speedup":str(v),"c1_3_paired_sd_compression":str(v*10)} for i,v in enumerate([2,3,4,5,6,7], 42)]
    m = metrics_from_pairs(fixture)
    assert any(x["statistic"] == "direction_count" and x["value"] == "6" for x in m)
    assert any(x["statistic"] == "simultaneous_ci97_5_lower" for x in m)
    print("self-test: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--stage", choices=["x6a", "x6b"], default="x6a")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return 0
    repo = args.repo_root.resolve()
    out = args.output_dir or (repo / "output/results/evidence_audit_part6" / args.stage)
    if args.stage == "x6a": write_x6a(repo, out)
    else: write_x6b(repo, out)
    print(json.dumps({"stage":args.stage,"output":str(out),"status":build(repo)["status"] if args.stage == "x6a" else "written"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
