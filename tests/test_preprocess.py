"""Tests for src/data/preprocess.py — splitting + model-input building."""

import pandas as pd
import pytest

from src.data.label_schema import LABELS, label_to_id
from src.data.preprocess import (
    Splits,
    _can_stratify,
    split_dataset,
    to_model_inputs,
)


def _balanced_df(n_per_label: int = 4) -> pd.DataFrame:
    """Test helper: balanced df."""
    rows = []
    rid = 0
    for label in LABELS:
        for i in range(n_per_label):
            rows.append(
                {
                    "id": f"r{rid}",
                    "topic": "ai",
                    "text": f"{label} example {i}",
                    "label": label,
                }
            )
            rid += 1
    return pd.DataFrame(rows)


def test_split_dataset_returns_three_splits_with_balanced_data():
    """Behavioral test: split dataset returns three splits with balanced data."""
    df = _balanced_df(4)  # 5 labels * 4 rows = 20 rows
    splits = split_dataset(df, test_size=0.25, val_size=0.1, seed=7)
    assert isinstance(splits, Splits)
    assert len(splits.train) + len(splits.val) + len(splits.test) == 20
    # Each split is reindexed from 0.
    assert splits.train.index[0] == 0
    assert splits.val.index[0] == 0
    assert splits.test.index[0] == 0


def test_split_dataset_rejects_invalid_split_fractions():
    """test_size + val_size must be a fraction in (0, 1) — reject sums ≥ 1
    (which would leave no train data / crash sklearn) up front."""
    df = _balanced_df(4)
    with pytest.raises(ValueError, match="must sum to a fraction"):
        split_dataset(df, test_size=0.6, val_size=0.5)


def test_split_dataset_falls_back_for_small_data():
    """Behavioral test: split dataset falls back for small data."""
    df = pd.DataFrame(
        [{"topic": "t", "text": "x", "label": "supportive"} for _ in range(5)]
    )
    splits = split_dataset(df)
    assert len(splits.train) >= 1
    assert len(splits.val) >= 0
    assert len(splits.test) >= 0
    assert len(splits.train) + len(splits.val) + len(splits.test) == 5


def test_split_dataset_skips_stratify_when_too_few_per_label():
    # A label with just one row -> _can_stratify returns False.
    """Behavioral test: split dataset skips stratify when too few per label."""
    rows = []
    for label in LABELS:
        rows.append({"topic": "t", "text": "x", "label": label})
    # Pad enough so we go past the small-dataset branch.
    for _ in range(20):
        rows.append({"topic": "t", "text": "x", "label": "supportive"})
    df = pd.DataFrame(rows)
    # Should not raise even though some labels have a count of 1.
    splits = split_dataset(df, seed=1)
    assert len(splits.train) > 0


def test_to_model_inputs_returns_texts_and_label_ids():
    """Behavioral test: to model inputs returns texts and label ids."""
    df = pd.DataFrame(
        [
            {"topic": "ai", "text": "I support it.", "label": "supportive"},
            {"topic": "econ", "text": "I disagree.", "label": "opposed"},
        ]
    )
    texts, label_ids = to_model_inputs(df)
    assert len(texts) == 2
    assert len(label_ids) == 2
    # Encoded input format: topic appears in the encoded text.
    assert "ai" in texts[0].lower()
    assert label_ids[0] == label_to_id("supportive")
    assert label_ids[1] == label_to_id("opposed")


def test_can_stratify_helper():
    """Behavioral test: can stratify helper."""
    assert _can_stratify(pd.Series(["a", "a", "b", "b"])) is True
    # Only one label -> not stratifiable.
    assert _can_stratify(pd.Series(["a", "a"])) is False
    # Label with min count <2 -> not stratifiable.
    assert _can_stratify(pd.Series(["a", "a", "b"])) is False
