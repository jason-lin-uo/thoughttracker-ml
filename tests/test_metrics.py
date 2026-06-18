"""Tests for src/training/metrics.py — pure metric helpers + PNG writer."""

import json
from pathlib import Path

from src.data.label_schema import LABELS
from src.training.metrics import (
    build_classification_report,
    build_confusion_matrix,
    compute_metrics,
    save_confusion_matrix_png,
    save_metrics,
)


def test_compute_metrics_perfect_prediction():
    """Behavioral test: compute metrics perfect prediction."""
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0


def test_compute_metrics_complete_mismatch():
    """Behavioral test: compute metrics complete mismatch."""
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.0


def test_build_classification_report_includes_each_label():
    """Behavioral test: build classification report includes each label."""
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    report = build_classification_report(y_true, y_pred)
    for label in LABELS:
        assert label in report


def test_build_confusion_matrix_dims():
    """Behavioral test: build confusion matrix dims."""
    y_true = [0, 1]
    y_pred = [0, 0]
    cm = build_confusion_matrix(y_true, y_pred)
    assert len(cm) == len(LABELS)
    assert len(cm[0]) == len(LABELS)


def test_save_metrics_writes_json(tmp_path: Path):
    """Behavioral test: save metrics writes json."""
    out = tmp_path / "metrics" / "out.json"
    payload = {"accuracy": 0.9, "extra": [1, 2, 3]}
    save_metrics(payload, out)
    assert out.exists()
    assert json.loads(out.read_text()) == payload


def test_save_confusion_matrix_png_writes_image(tmp_path: Path):
    """Behavioral test: save confusion matrix png writes image."""
    cm = [[1 if i == j else 0 for j in range(len(LABELS))] for i in range(len(LABELS))]
    out = tmp_path / "figures" / "cm.png"
    save_confusion_matrix_png(cm, out, title="Test CM")
    assert out.exists()
    # PNG header bytes
    assert out.read_bytes()[:4] == b"\x89PNG"
