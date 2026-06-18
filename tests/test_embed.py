"""Tests for src/inference/embed.py + the POST /embed endpoint.

Runs in mock-inference mode (no torch model in hermetic CI). The real
encoder/inference helpers (`_build_encoder`, `_encode_with_model`) are pragma'd
and exercised by the smoke test / real runs; everything else — the mock vector,
the load bookkeeping, the fallback, and the endpoint — is covered here.
"""

import dataclasses
import os

os.environ["ENABLE_MOCK_INFERENCE"] = "true"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402
from src.inference import embed  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_embed_state(monkeypatch):
    """Each test starts with a clean embed-module cache."""
    monkeypatch.setattr(embed, "_loaded", None)
    monkeypatch.setattr(embed, "_load_error", None)
    monkeypatch.setattr(embed, "_load_attempted", False)


def _mock_off(monkeypatch):
    monkeypatch.setattr(embed, "config", dataclasses.replace(embed.config, enable_mock_inference=False))


def test_mock_vector_is_unit_length_768():
    v = embed._mock_vector("foldable phones are durable and worth the price")
    assert len(v) == embed.EMBED_DIM == 768
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_mock_vector_empty_text_is_all_zero():
    v = embed._mock_vector("")
    assert len(v) == 768 and all(x == 0.0 for x in v)


def test_embed_texts_mock_mode_returns_768d():
    out = embed.embed_texts(["hello world", "second text"])
    assert len(out) == 2 and all(len(v) == 768 for v in out)


def test_embed_texts_falls_back_to_mock_when_model_unavailable(monkeypatch):
    _mock_off(monkeypatch)
    monkeypatch.setattr(embed, "_load_attempted", True)  # skip the load; _loaded stays None
    out = embed.embed_texts(["x y z foo bar"])
    assert len(out) == 1 and len(out[0]) == 768
    assert embed.is_embed_model_available() is False


def test_embed_texts_uses_encoder_when_loaded(monkeypatch):
    _mock_off(monkeypatch)
    monkeypatch.setattr(embed, "_loaded", ("tok", "model", "torch"))
    monkeypatch.setattr(embed, "_encode_with_model", lambda texts: [[0.5] * 768 for _ in texts])
    out = embed.embed_texts(["a", "b"])
    assert out == [[0.5] * 768, [0.5] * 768]
    assert embed.is_embed_model_available() is True


def test_load_success_via_build_encoder(monkeypatch):
    monkeypatch.setattr(embed, "_build_encoder", lambda: ("tok", "model", "torch"))
    embed._load()
    assert embed.is_embed_model_available() is True
    assert embed.get_embed_load_error() is None


def test_load_failure_records_error(monkeypatch):
    def boom():
        raise RuntimeError("no torch here")

    monkeypatch.setattr(embed, "_build_encoder", boom)
    embed._load()
    assert embed.is_embed_model_available() is False
    assert embed.get_embed_load_error() == "no torch here"


def test_load_embed_model_is_idempotent(monkeypatch):
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        embed._load_attempted = True

    monkeypatch.setattr(embed, "_load", fake_load)
    embed.load_embed_model()  # _load_attempted False → loads
    embed.load_embed_model()  # now True → no-op
    assert calls["n"] == 1


def test_embed_endpoint_returns_vectors():
    res = client.post("/embed", json={"texts": ["hello", "world"]})
    assert res.status_code == 200
    body = res.json()
    assert body["dim"] == 768
    assert len(body["vectors"]) == 2 and len(body["vectors"][0]) == 768
    assert isinstance(body["mockInference"], bool)


def test_embed_endpoint_handles_internal_error(monkeypatch):
    import src.api.main as main

    def boom(_texts):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "embed_texts", boom)
    res = client.post("/embed", json={"texts": ["x"]})
    assert res.status_code == 500
    assert res.json()["error"] == "INTERNAL_ERROR"
