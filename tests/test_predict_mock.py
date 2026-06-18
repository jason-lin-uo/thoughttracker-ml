"""Tests for the deterministic mock predictor.

These tests do not require torch or transformers. They cover the response
shape and the mock fallback path used when no trained model exists.
"""

import os

# Force mock mode and a non-existent model directory before importing.
os.environ["ENABLE_MOCK_INFERENCE"] = "true"
os.environ["MODEL_DIR"] = "models/__nonexistent__"

import pytest  # noqa: E402

from src.data.label_schema import LABELS  # noqa: E402
from src.inference.predict import predict  # noqa: E402


def test_predict_returns_full_response_shape():
    """Behavioral test: predict returns full response shape."""
    out = predict(topic="economics", text="I disagree with this approach.")
    assert out["topic"] == "economics"
    assert out["text"] == "I disagree with this approach."
    assert out["predictedLabel"] in LABELS
    assert 0.0 <= out["confidence"] <= 1.0
    assert set(out["labelScores"].keys()) == set(LABELS)
    total = sum(out["labelScores"].values())
    assert 0.95 < total < 1.05  # softmax → ~1
    assert out["modelVersion"].endswith("-mock")


def test_predict_supportive_cue():
    """Behavioral test: predict supportive cue."""
    out = predict(topic="ai", text="I support this and I'm in favor of it.")
    assert out["predictedLabel"] == "supportive"


def test_predict_opposed_cue():
    """Behavioral test: predict opposed cue."""
    out = predict(topic="ai", text="I disagree with this. We shouldn't be doing this.")
    assert out["predictedLabel"] == "opposed"


def test_predict_neutral_cue():
    """Behavioral test: predict neutral cue."""
    out = predict(
        topic="ai",
        text="According to the data, there are tradeoffs in both directions.",
    )
    assert out["predictedLabel"] == "neutral"


def test_predict_requires_text():
    """Behavioral test: predict requires text."""
    with pytest.raises(ValueError):
        predict(topic="ai", text="   ")


def test_predict_requires_topic():
    """Behavioral test: predict requires topic."""
    with pytest.raises(ValueError):
        predict(topic="", text="hello")


def test_predict_deterministic():
    """Behavioral test: predict deterministic."""
    a = predict(topic="ai", text="this is a sample excerpt with no strong cue")
    b = predict(topic="ai", text="this is a sample excerpt with no strong cue")
    assert a == b
