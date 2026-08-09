import csv
import json
from pathlib import Path

import numpy as np

from scripts import audit_c4_cbp as audit


def test_population_sd_and_complete_batch_filter():
    rows = [
        {"batch_idx": "0", "epoch": "0", "config": "BL", "neg_time_ms": "1.0"},
        {"batch_idx": "1", "epoch": "0", "config": "BL", "neg_time_ms": "3.0"},
        {"batch_idx": "2", "epoch": "0", "config": "BL", "neg_time_ms": "5.0"},
        {"batch_idx": "3", "epoch": "0", "config": "BL", "neg_time_ms": "99.0"},
    ]
    assert audit.population_sd([1.0, 3.0, 5.0]) == np.std([1.0, 3.0, 5.0], ddof=0)
    assert audit.filter_step_rows(rows, exclude_partial=True, exclude_first=True)[0]["batch_idx"] == "1"


def test_chunk_and_legacy_ffd_are_equivalent_for_fixtures():
    fixtures = [[], list(range(3)), list(range(10)), list(range(11))]
    for values in fixtures:
        assert audit.chunk_pack(values, 5) == audit.legacy_ffd_pack(values, 5)


def test_greedy_layout_is_distinct_and_covers_exactly():
    values = list(range(23))
    scores = {value: float((value * 7) % 11 + 1) for value in values}
    layout = audit.greedy_least_load_pack(values, scores, [5, 5, 5, 5, 3])
    assert [len(batch) for batch in layout] == [5, 5, 5, 5, 3]
    assert sorted(item for batch in layout for item in batch) == values
    assert layout != audit.chunk_pack(values, 5)


def test_phase9_epoch_sd_summary_excludes_partial():
    rows = [
        {"config": "BL", "epoch": "0", "batch_idx": str(i), "neg_time_ms": str(i + 1)}
        for i in range(3)
    ] + [{"config": "BL", "epoch": "0", "batch_idx": "3", "neg_time_ms": "100"}]
    metrics = audit.epoch_sd_metrics(rows, config="BL", exclude_partial=True)
    assert metrics[0]["n"] == 3
    assert metrics[0]["sd"] == np.std([1.0, 2.0, 3.0], ddof=0)


def test_self_test_payload_is_deterministic():
    first = audit.run_cpu_fixtures()
    second = audit.run_cpu_fixtures()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_repository_audit_has_expected_sources_and_claims():
    root = Path(__file__).resolve().parents[1]
    result = audit.build_audit(root)
    assert len(result["claim_verdicts"]) == 10
    assert result["facts"]["phase6_rows"] == 546
    assert result["facts"]["phase9_step45_rows"] == 324
    assert result["facts"]["integration_rows"] == 220
    assert result["facts"]["legacy_ffd_equals_chunk"] is True
