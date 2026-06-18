"""Extra API tests covering startup behavior + error branches."""

import dataclasses
import os

os.environ["ENABLE_MOCK_INFERENCE"] = "true"
os.environ["MODEL_DIR"] = "models/__nonexistent__"

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main as api_main  # noqa: E402


client = TestClient(api_main.app)


def test_predict_500_when_underlying_predict_raises(monkeypatch):
    """If predict() raises something other than ValueError/FileNotFoundError,
    the endpoint should return 500 with the structured error shape.
    The message is intentionally generic — the original exception text
    is logged server-side but not echoed to the client to avoid PII or
    internal-path leakage."""

    def boom(*_args, **_kwargs):
        """Test helper: boom."""
        raise RuntimeError("kapow")

    monkeypatch.setattr(api_main, "predict", boom)
    resp = client.post("/predict", json={"topic": "ai", "text": "hello"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "INTERNAL_ERROR"
    # Generic message — should NOT leak the original "kapow" text.
    assert "kapow" not in body["message"]
    assert "Internal server error" in body["message"]


def test_predict_503_when_model_file_not_found(monkeypatch):
    """Behavioral test: predict 503 when model file not found."""
    def missing(*_args, **_kwargs):
        """Test helper: missing."""
        raise FileNotFoundError("no model on disk")

    monkeypatch.setattr(api_main, "predict", missing)
    resp = client.post("/predict", json={"topic": "ai", "text": "hello"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "MODEL_NOT_LOADED"


def test_predict_400_when_predict_raises_value_error(monkeypatch):
    """Behavioral test: predict 400 when predict raises value error."""
    def bad(*_args, **_kwargs):
        """Test helper: bad."""
        raise ValueError("bad input")

    monkeypatch.setattr(api_main, "predict", bad)
    resp = client.post("/predict", json={"topic": "ai", "text": "hello"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "INVALID_INPUT"


def test_startup_warmup_skips_when_no_model(monkeypatch, tmp_path):
    """warmup_model logs + returns early when no model is on disk."""
    new_cfg = dataclasses.replace(
        api_main.config, model_dir=tmp_path / "nope", enable_mock_inference=True
    )
    monkeypatch.setattr(api_main, "config", new_cfg)
    # Should not raise even though no model exists.
    api_main._warmup_model()


def test_startup_warmup_warns_when_no_model_and_not_mock(monkeypatch, tmp_path):
    """Behavioral test: startup warmup warns when no model and not mock."""
    new_cfg = dataclasses.replace(
        api_main.config, model_dir=tmp_path / "nope", enable_mock_inference=False
    )
    monkeypatch.setattr(api_main, "config", new_cfg)
    api_main._warmup_model()  # Just exercises the warning branch.


def test_startup_warmup_calls_load_model_when_available(monkeypatch):
    """Behavioral test: startup warmup calls load model when available."""
    monkeypatch.setattr(api_main, "is_model_available", lambda: True)
    called = {"count": 0}

    def fake_load():
        """Test helper: fake load."""
        called["count"] += 1

    monkeypatch.setattr(api_main, "load_model", fake_load)
    api_main._warmup_model()
    assert called["count"] == 1


def test_startup_warmup_loads_topic_models_when_available(monkeypatch):
    """warmup logs success for the topic relevance + reranker models when both
    are present. Stubs availability + load so the success branches are covered
    WITHOUT the real (LFS) model files — the CI environment has none, so these
    lines were the only ones missing CI coverage (model-present-only locally)."""
    monkeypatch.setattr(api_main, "is_model_available", lambda: True)
    monkeypatch.setattr(api_main, "load_model", lambda: None)
    monkeypatch.setattr(api_main, "is_topic_relevance_model_available", lambda: True)
    monkeypatch.setattr(api_main, "load_topic_relevance_model", lambda: {"kind": "stub"})
    monkeypatch.setattr(api_main, "is_topic_reranker_model_available", lambda: True)
    monkeypatch.setattr(api_main, "load_topic_reranker_model", lambda: {"kind": "stub"})
    api_main._warmup_model()  # exercises the relevance + reranker success logs


def test_startup_warmup_swallows_load_errors(monkeypatch):
    """Behavioral test: startup warmup swallows load errors."""
    monkeypatch.setattr(api_main, "is_model_available", lambda: True)

    def boom():
        """Test helper: boom."""
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(api_main, "load_model", boom)
    # Should not raise — warmup_model logs and continues.
    api_main._warmup_model()
