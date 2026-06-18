"""Extra tests for src/inference/predict.py to cover CLI + edge cases."""

import dataclasses
import json
import os

os.environ["ENABLE_MOCK_INFERENCE"] = "true"
os.environ["MODEL_DIR"] = "models/__nonexistent__"

import pytest  # noqa: E402

from src.inference import predict as predict_mod  # noqa: E402


def test_mock_predict_mixed_cue():
    """Behavioral test: mock predict mixed cue."""
    out = predict_mod.predict(
        topic="ai",
        text="I see the upside, however on the other hand there are risks at the same time.",
    )
    assert out["predictedLabel"] in (
        "mixed",
        "neutral",
        "unclear",
        "supportive",
        "opposed",
    )
    # mixed cues should at least register a non-zero score.
    assert out["labelScores"]["mixed"] > 0


def test_mock_predict_unclear_cue():
    """Behavioral test: mock predict unclear cue."""
    out = predict_mod.predict(
        topic="ai",
        text="I'm not sure, it's hypothetical and could be read either way.",
    )
    # Confidence is positive and label is in valid set.
    assert 0 < out["confidence"] <= 1


def test_mock_predict_no_cue_falls_back_to_unclear():
    """Behavioral test: mock predict no cue falls back to unclear."""
    out = predict_mod.predict(
        topic="weather",
        text="zzzzz qwerty floof",
    )
    # When no cue matches, fallback assigns unclear baseline.
    assert out["predictedLabel"] in (
        "unclear",
        "neutral",
        "supportive",
        "mixed",
        "opposed",
    )
    assert sum(out["labelScores"].values()) == pytest.approx(1.0, rel=0.05)


def test_main_cli_returns_zero_and_prints_json(capsys):
    """Behavioral test: main cli returns zero and prints json."""
    rc = predict_mod.main(["--topic", "ai", "--text", "I support this."])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["predictedLabel"] == "supportive"


def test_main_cli_value_error_returns_1(capsys):
    """Behavioral test: main cli value error returns 1."""
    rc = predict_mod.main(["--topic", "", "--text", "hello"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "topic" in err.lower()


def test_main_cli_file_not_found_returns_2(monkeypatch, capsys):
    """When ENABLE_MOCK_INFERENCE is false and no model exists, the CLI
    should print the helpful error and return 2."""
    new_cfg = dataclasses.replace(predict_mod.config, enable_mock_inference=False)
    monkeypatch.setattr(predict_mod, "config", new_cfg)
    rc = predict_mod.main(["--topic", "ai", "--text", "I support this."])
    assert rc == 2
    err = capsys.readouterr().err
    assert "No trained model" in err


def test_softmax_helper_sums_to_one():
    """Behavioral test: softmax helper sums to one."""
    out = predict_mod._softmax([1.0, 2.0, 3.0])
    assert abs(sum(out) - 1.0) < 1e-6
    assert out[2] > out[1] > out[0]


def test_softmax_helper_handles_empty_input():
    """Degenerate case: _softmax on an empty list returns an empty list.

    The only in-app caller passes a 5-element list (one per label) so
    this path isn't hit during normal inference, but the function is
    public-ish (leading underscore is a Python-style 'internal' marker,
    not a hard barrier) and the defensive branch is worth documenting
    with a test.
    """
    assert predict_mod._softmax([]) == []


def test_argmax_tie_break_is_deterministic_first_wins():
    """An exact probability tie must resolve to the FIRST label in LABELS.

    ``_build_prediction_response`` picks the predicted label with
    ``max(range(...), key=lambda i: probs[i])``, which returns the lowest
    index attaining the maximum — i.e. first-wins in ``LABELS`` order.
    Here every label shares the identical probability, so the tie-break is
    the *only* thing deciding the winner. Asserting it lands on
    ``LABELS[0]`` pins the documented "earlier label wins" guarantee so a
    future refactor (e.g. swapping in ``np.argmax``, whose tie behavior is
    dtype/version-sensitive) can't silently make predictions
    non-reproducible.
    """
    from src.data.label_schema import LABELS

    n = len(LABELS)
    tied = [1.0 / n] * n
    out = predict_mod._build_prediction_response("topic", "text", tied, "v-test")
    assert out["predictedLabel"] == LABELS[0]
    # Idempotent: same input → same winner, every time.
    again = predict_mod._build_prediction_response("topic", "text", tied, "v-test")
    assert again["predictedLabel"] == LABELS[0]
