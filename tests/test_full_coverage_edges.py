"""Targeted edge-path coverage for topic API and inference modules."""

from __future__ import annotations

import dataclasses
import pickle
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.config import config as base_config


class _CliRelevanceModel:
    """Test double / fixture: CliRelevanceModel."""
    classes_ = np.array(["irrelevant", "relevant"])

    def predict_proba(self, _rows):
        """Test helper: predict proba."""
        return np.array([[0.4, 0.6]])


class _CliVectorizer:
    """Test double / fixture: CliVectorizer."""
    def transform(self, rows):
        """Test helper: transform."""
        return rows


class _CliClassifier:
    """Test double / fixture: CliClassifier."""
    def predict_proba(self, _matrix):
        """Test helper: predict proba."""
        return [np.array([0.7, 0.1])]


class _CliMultiLabelBinarizer:
    """Test double / fixture: CliMultiLabelBinarizer."""
    classes_ = np.array(["ai_policy", "economics"])


client = TestClient(api_main.app)


def test_startup_warmup_swallows_topic_model_load_errors(monkeypatch):
    """Behavioral test: startup warmup swallows topic model load errors."""
    monkeypatch.setattr(api_main, "is_model_available", lambda: True)
    monkeypatch.setattr(api_main, "load_model", lambda: None)
    monkeypatch.setattr(api_main, "is_topic_relevance_model_available", lambda: True)
    monkeypatch.setattr(api_main, "is_topic_reranker_model_available", lambda: True)

    def relevance_boom():
        """Test helper: relevance boom."""
        raise RuntimeError("relevance warmup failed")

    def reranker_boom():
        """Test helper: reranker boom."""
        raise RuntimeError("reranker warmup failed")

    monkeypatch.setattr(api_main, "load_topic_relevance_model", relevance_boom)
    monkeypatch.setattr(api_main, "load_topic_reranker_model", reranker_boom)

    api_main._warmup_model()


def test_predict_topic_relevance_success(monkeypatch):
    """Behavioral test: predict topic relevance success."""
    monkeypatch.setattr(
        api_main,
        "predict_topic_relevance",
        lambda topic, text: {
            "topic": topic,
            "text": text,
            "predictedLabel": "relevant",
            "confidence": 0.91,
            "labelScores": {"irrelevant": 0.09, "relevant": 0.91},
            "modelVersion": "topic-rel-test",
        },
    )

    resp = client.post(
        "/predict-topic-relevance",
        json={"topic": "ai", "text": "This is about AI policy."},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "predictedLabel": "relevant",
        "confidence": 0.91,
        "labelScores": {"irrelevant": 0.09, "relevant": 0.91},
        "modelVersion": "topic-rel-test",
    }


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (FileNotFoundError("missing topic relevance model"), 503, "MODEL_NOT_LOADED"),
        (ValueError("bad topic relevance input"), 400, "INVALID_INPUT"),
        (RuntimeError("sensitive topic relevance failure"), 500, "INTERNAL_ERROR"),
    ],
)
def test_predict_topic_relevance_error_branches(
    monkeypatch, error, status_code, error_code
):
    """Behavioral test: predict topic relevance error branches."""
    def fail(*_args, **_kwargs):
        """Test helper: fail."""
        raise error

    monkeypatch.setattr(api_main, "predict_topic_relevance", fail)
    resp = client.post(
        "/predict-topic-relevance",
        json={"topic": "ai", "text": "This is about AI policy."},
    )

    assert resp.status_code == status_code
    body = resp.json()
    assert body["error"] == error_code
    if status_code == 500:
        assert "sensitive topic relevance failure" not in body["message"]


def test_predict_topics_success(monkeypatch):
    """Behavioral test: predict topics success."""
    monkeypatch.setattr(
        api_main,
        "predict_topic_candidates",
        lambda text, limit, min_score: {
            "topics": [
                {"topicSlug": "ai_policy", "confidence": 0.88},
                {"topicSlug": "economics", "confidence": 0.33},
            ],
            "modelVersion": f"topics-{limit}-{min_score}-{text[:4]}",
        },
    )

    resp = client.post(
        "/predict-topics",
        json={"text": "AI policy and jobs", "limit": 2, "minScore": 0.3},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "topics": [
            {"topicSlug": "ai_policy", "confidence": 0.88},
            {"topicSlug": "economics", "confidence": 0.33},
        ],
        "modelVersion": "topics-2-0.3-AI p",
    }


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (FileNotFoundError("missing topic reranker model"), 503, "MODEL_NOT_LOADED"),
        (ValueError("bad topic reranker input"), 400, "INVALID_INPUT"),
        (RuntimeError("sensitive topic reranker failure"), 500, "INTERNAL_ERROR"),
    ],
)
def test_predict_topics_error_branches(monkeypatch, error, status_code, error_code):
    """Behavioral test: predict topics error branches."""
    def fail(*_args, **_kwargs):
        """Test helper: fail."""
        raise error

    monkeypatch.setattr(api_main, "predict_topic_candidates", fail)
    resp = client.post("/predict-topics", json={"text": "AI policy and jobs"})

    assert resp.status_code == status_code
    body = resp.json()
    assert body["error"] == error_code
    if status_code == 500:
        assert "sensitive topic reranker failure" not in body["message"]


def test_topic_relevance_predict_dispatches_transformer_branch(monkeypatch):
    """Behavioral test: topic relevance predict dispatches transformer branch."""
    from src.inference import topic_relevance

    monkeypatch.setattr(
        topic_relevance,
        "load_topic_relevance_model",
        lambda: {"kind": "transformer", "model": object(), "tokenizer": object()},
    )
    monkeypatch.setattr(
        topic_relevance,
        "_predict_transformer",
        lambda loaded, encoded: [0.31, 0.69],
    )
    monkeypatch.setattr(
        topic_relevance,
        "config",
        dataclasses.replace(
            topic_relevance.config, topic_relevance_model_version="transformer-test"
        ),
    )

    out = topic_relevance.predict_topic_relevance("AI Policy", "A chunk about AI.")

    assert out["predictedLabel"] == "relevant"
    assert out["labelScores"] == {"irrelevant": 0.31, "relevant": 0.69}
    assert out["modelVersion"] == "transformer-test"


def test_topic_reranker_filters_scores_below_threshold(monkeypatch):
    """Behavioral test: topic reranker filters scores below threshold."""
    from src.inference import topic_reranker
    from tests.test_topic_inference_modules import (
        FakeClassifier,
        FakeMultiLabelBinarizer,
        FakeVectorizer,
    )

    monkeypatch.setattr(
        topic_reranker,
        "load_topic_reranker_model",
        lambda: {
            "vectorizer": FakeVectorizer(),
            "classifier": FakeClassifier(),
            "multiLabelBinarizer": FakeMultiLabelBinarizer(),
            "modelVersion": "threshold-test",
        },
    )

    out = topic_reranker.predict_topic_candidates(
        "AI policy and economics", limit=5, min_score=0.5
    )

    assert out == {
        "topics": [{"topicSlug": "topic_a", "confidence": 0.92}],
        "modelVersion": "threshold-test",
    }


def test_topic_relevance_module_entrypoint_runs_main(
    monkeypatch, tmp_path: Path, capsys
):
    """Behavioral test: topic relevance module entrypoint runs main."""
    import src.config as config_mod
    from src.inference import topic_relevance

    model_path = tmp_path / topic_relevance.MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(_CliRelevanceModel(), handle)

    monkeypatch.setattr(
        config_mod,
        "config",
        dataclasses.replace(
            base_config,
            topic_relevance_model_dir=tmp_path,
            topic_relevance_model_version="cli-relevance-test",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["topic_relevance.py", "--topic", "AI", "--text", "AI policy chunk"],
    )

    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("src.inference.topic_relevance", run_name="__main__")

    assert excinfo.value.code == 0
    assert '"modelVersion": "cli-relevance-test"' in capsys.readouterr().out


def test_topic_reranker_module_entrypoint_runs_main(
    monkeypatch, tmp_path: Path, capsys
):
    """Behavioral test: topic reranker module entrypoint runs main."""
    import src.config as config_mod
    from src.inference import topic_reranker

    model_path = tmp_path / topic_reranker.MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "vectorizer": _CliVectorizer(),
                "classifier": _CliClassifier(),
                "multiLabelBinarizer": _CliMultiLabelBinarizer(),
                "modelVersion": "cli-reranker-test",
            },
            handle,
        )

    monkeypatch.setattr(
        config_mod,
        "config",
        dataclasses.replace(
            base_config,
            topic_reranker_model_dir=tmp_path,
            topic_reranker_model_version="fallback-reranker-test",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["topic_reranker.py", "--text", "AI policy chunk", "--min-score", "0.2"],
    )

    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("src.inference.topic_reranker", run_name="__main__")

    assert excinfo.value.code == 0
    assert '"modelVersion": "cli-reranker-test"' in capsys.readouterr().out
