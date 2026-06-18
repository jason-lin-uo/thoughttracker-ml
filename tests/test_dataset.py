"""Tests for dataset loading and validation."""

import pandas as pd
import pytest

from src.data.load_dataset import (
    DatasetValidationError,
    build_input_text,
    load_stance_dataset,
    validate_dataset,
)


def _write_tiny_dataset(path):
    """Test helper: write tiny dataset."""
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "topic": ["ai", "health", "policy", "media", "business"],
            "text": ["support", "oppose", "neutral", "mixed", "unclear"],
            "label": ["supportive", "opposed", "neutral", "mixed", "unclear"],
        }
    ).to_csv(path, index=False)


def test_dataset_loads_from_explicit_path(tmp_path):
    """Behavioral test: dataset loads from explicit path."""
    csv_path = tmp_path / "stance.csv"
    _write_tiny_dataset(csv_path)
    df = load_stance_dataset(csv_path)
    assert len(df) > 0
    assert set(["id", "topic", "text", "label"]).issubset(df.columns)


def test_dataset_only_uses_known_labels(tmp_path):
    """Behavioral test: dataset only uses known labels."""
    csv_path = tmp_path / "stance.csv"
    _write_tiny_dataset(csv_path)
    df = load_stance_dataset(csv_path)
    allowed = {"supportive", "opposed", "neutral", "mixed", "unclear"}
    assert set(df["label"].str.lower()) <= allowed


def test_validate_rejects_missing_columns():
    """Behavioral test: validate rejects missing columns."""
    bad = pd.DataFrame({"id": [1], "label": ["supportive"]})
    with pytest.raises(DatasetValidationError):
        validate_dataset(bad)


def test_validate_rejects_unknown_labels():
    """Behavioral test: validate rejects unknown labels."""
    bad = pd.DataFrame(
        {
            "id": [1],
            "topic": ["x"],
            "text": ["y"],
            "label": ["definitely_unknown"],
        }
    )
    with pytest.raises(DatasetValidationError):
        validate_dataset(bad)


def test_build_input_text_format():
    """Behavioral test: build input text format."""
    encoded = build_input_text("foreign policy", "I disagree with this approach.")
    assert "[TOPIC] foreign policy" in encoded
    assert "[TEXT] I disagree with this approach." in encoded


def test_build_input_text_handles_whitespace():
    """Behavioral test: build input text handles whitespace."""
    assert build_input_text("  ai  ", "  hello  ") == "[TOPIC] ai [TEXT] hello"
