"""Tests for the canonical label schema."""

import pytest

from src.data.label_schema import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    all_labels,
    id_to_label,
    label_to_id,
    num_labels,
)


def test_labels_are_unique():
    """Behavioral test: labels are unique."""
    assert len(LABELS) == len(set(LABELS))


def test_label_count():
    """Behavioral test: label count."""
    assert num_labels() == 5
    assert set(LABELS) == {"supportive", "opposed", "neutral", "mixed", "unclear"}


def test_label_to_id_roundtrip():
    """Behavioral test: label to id roundtrip."""
    for label in LABELS:
        idx = label_to_id(label)
        assert id_to_label(idx) == label


def test_id2label_matches_label2id():
    """Behavioral test: id2label matches label2id."""
    assert {v: k for k, v in LABEL2ID.items()} == ID2LABEL


def test_unknown_label_raises():
    """Behavioral test: unknown label raises."""
    with pytest.raises(ValueError):
        label_to_id("definitely_not_a_label")


def test_unknown_id_raises():
    """Behavioral test: unknown id raises."""
    with pytest.raises(ValueError):
        id_to_label(99)


def test_all_labels_returns_list_copy():
    """Behavioral test: all labels returns list copy."""
    out = all_labels()
    assert out == list(LABELS)
    out.append("hacked")
    assert "hacked" not in LABELS  # mutating return value must not mutate source
