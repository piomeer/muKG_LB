#!/usr/bin/env python3
"""Run the one protocol-allowed infrastructure retry for a C1-R1 pair."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "src/py/experiments/c1_r1_combined_rerun.py"
ROOT = REPO / "output/results/c1_r1_combined_rerun"
RETRY_ROOT = ROOT / "reruns/throughput_seed45_attempt2"
FIELDS = [
    "label", "config", "seed", "attempt", "start_time_ns", "end_time_ns",
    "elapsed_ns", "returncode", "stdout_path", "stderr_path",
    "stdout_sha256", "stderr_sha256",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def artifact_rows() -> list[dict[str, object]]:
    rows = []
    for path in sorted(item for item in RETRY_ROOT.rglob("*") if item.is_file()):
        if path.name == "artifact_hashes.csv":
            continue
        rows.append({
            "path": str(path.relative_to(RETRY_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def main() -> int:
    if RETRY_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite retry evidence: {RETRY_ROOT}")
    (RETRY_ROOT / "logs").mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    for config in ("GPU", "BL"):
        label = f"throughput_{config}_seed45_attempt2"
        stdout_path = RETRY_ROOT / "logs" / f"{label}.stdout.log"
        stderr_path = RETRY_ROOT / "logs" / f"{label}.stderr.log"
        command = [
            sys.executable,
            str(RUNNER),
            "job",
            "--root",
            str(RETRY_ROOT),
            "--config",
            config,
            "--pass-name",
            "throughput",
            "--seed",
            "45",
        ]
        start_ns = time.time_ns()
        print(f"START {label}", flush=True)
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, \
                stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
            result = subprocess.run(command, cwd=REPO, stdout=stdout, stderr=stderr)
        end_ns = time.time_ns()
        row = {
            "label": label,
            "config": config,
            "seed": 45,
            "attempt": 2,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "elapsed_ns": end_ns - start_ns,
            "returncode": result.returncode,
            "stdout_path": str(stdout_path.relative_to(RETRY_ROOT)),
            "stderr_path": str(stderr_path.relative_to(RETRY_ROOT)),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
        }
        manifest.append(row)
        write_csv(RETRY_ROOT / "rerun_manifest.csv", FIELDS, manifest)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {stderr_path}")
        print(f"DONE {label} elapsed={(end_ns - start_ns) / 1e9:.1f}s", flush=True)
    write_csv(
        RETRY_ROOT / "artifact_hashes.csv",
        ["path", "bytes", "sha256"],
        artifact_rows(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
