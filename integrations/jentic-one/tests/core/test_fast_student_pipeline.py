import json
from pathlib import Path

import numpy as np
import pytest

from scripts.training.build_jentic_student_data import render_rows
from scripts.training.train_fast_student import calibrate_thresholds, load_student_rows

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]


def test_synthetic_builder_never_creates_test_rows_or_cross_split_families():
    rows = render_rows(seed=42, variants=8)
    family_splits = {}
    for row in rows:
        assert row["split"] in {"train", "validation"}
        family_splits.setdefault(row["family"], row["split"])
        assert family_splits[row["family"]] == row["split"]
    assert {(row["split"], row["label"]) for row in rows} == {
        ("train", 0),
        ("train", 1),
        ("validation", 0),
        ("validation", 1),
    }


def test_training_loader_rejects_test_split(tmp_path):
    path = tmp_path / "student.jsonl"
    row = {
        "id": "case-1",
        "text": "Unique text",
        "label": 0,
        "surface": "runtime_response",
        "family": "family-1",
        "split": "test",
        "provenance": "synthetic",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test data is forbidden"):
        load_student_rows(path)


def test_bounded_calibration_finds_separated_thresholds():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])

    low, high, rates = calibrate_thresholds(labels, scores)

    assert 0.03 < low < high <= 0.97
    assert rates["attack_recall"] == 1.0
    assert rates["benign_false_block_rate"] == 0.0
    assert rates["benign_flag_rate"] == 0.0
    assert rates["validation_gates_passed"] is True


def test_public_training_source_is_revision_pinned_and_excludes_test():
    manifest = json.loads(
        (INTEGRATION_ROOT / "evaluation" / "sources" / "jentic-training-sources.json").read_text()
    )
    source = manifest["sources"][0]

    assert len(source["revision"]) == 40
    assert source["license"] == "MIT"
    assert source["allowed_splits"] == ["train", "validation"]
    assert source["excluded_splits"] == ["test"]
