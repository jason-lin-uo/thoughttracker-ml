"""
Behavioral hardening tests (audit §9 follow-ups).

Unlike the ``coverage-*`` / ``*_lift`` touch-tests (which exist to pin
specific lines), every test here asserts a *behavior* a careful reviewer
would probe:

  - group-aware split / leakage guard (no source row straddles splits)
  - loader thread-safety (one load under concurrent first-requests)
  - the exact error code per the integration contract (no ``400 or 422``)
  - generic, path-free 503 messages (privacy stance)
  - the device-aware eval selector (cuda > mps > cpu)
  - env-var validation failing loudly with a clear message
  - the ``/health`` load-error surface and the lifespan warmup hook
  - the ``len(probs) == len(LABELS)`` schema-drift guard
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
import types
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

os.environ["ENABLE_MOCK_INFERENCE"] = "true"
os.environ["MODEL_DIR"] = "models/__nonexistent__"


# ---------------------------------------------------------------------------
# Group-aware split / leakage guard (preprocess.py)
# ---------------------------------------------------------------------------


def _grouped_df(n_groups: int = 6, chunks_per_group: int = 4) -> pd.DataFrame:
    """Build a df where each ``id`` group has several near-duplicate chunks.

    Mimics the real failure mode: many chunks share one source id, so a
    naive row split would scatter near-duplicates across train/test.
    """
    from src.data.label_schema import LABELS

    rows = []
    for g in range(n_groups):
        label = LABELS[g % len(LABELS)]
        for c in range(chunks_per_group):
            rows.append(
                {
                    "id": f"video-{g}",
                    "topic": "ai",
                    "text": f"group {g} chunk {c} about ai",
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_split_is_group_aware_no_id_straddles_splits():
    """No source ``id`` may appear in more than one split."""
    from src.data.preprocess import assert_no_group_leakage, split_dataset

    df = _grouped_df(n_groups=8, chunks_per_group=5)
    splits = split_dataset(df, test_size=0.25, val_size=0.125, seed=7)

    # The public guard raises AssertionError on any straddling group.
    checked = assert_no_group_leakage(splits)
    assert "id" in checked

    train_ids = set(splits.train["id"])
    val_ids = set(splits.val["id"])
    test_ids = set(splits.test["id"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(val_ids)
    assert val_ids.isdisjoint(test_ids)
    # Every row is accounted for exactly once.
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)


def _label_balanced_df(n_per_label: int = 8) -> pd.DataFrame:
    """A row-per-example df with every label well-represented (stratifiable)."""
    from src.data.label_schema import LABELS

    rows = []
    rid = 0
    for label in LABELS:
        for i in range(n_per_label):
            rows.append({"id": f"r{rid}", "topic": "ai", "text": f"{label} {i}", "label": label})
            rid += 1
    return pd.DataFrame(rows)


def test_split_falls_back_to_stratified_when_no_group_column():
    """With no group key present, splitting still works (stratified path).

    Uses a label-balanced df with split fractions that leave each of the
    second-level (val/test) splits with at least one row per class, so
    sklearn's stratifier is satisfied — the point of the test is the
    *fallback path*, not stratification's row-count preconditions.
    """
    from src.data.preprocess import split_dataset

    df = _label_balanced_df(10).drop(columns=["id"])  # 50 rows, 10 per label
    splits = split_dataset(df, test_size=0.2, val_size=0.2, seed=1)
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)


def test_split_explicit_empty_group_columns_disables_grouping():
    """Passing an empty ``group_columns`` forces the stratified fallback."""
    from src.data.preprocess import split_dataset

    # Unique ids per row so the group path (if enabled) would be a row
    # split; the empty group_columns forces the stratified branch instead.
    df = _label_balanced_df(10)
    splits = split_dataset(df, test_size=0.2, val_size=0.2, seed=1, group_columns=[])
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)


def test_resolve_group_key_declines_with_too_few_groups():
    """A column with <3 distinct groups can't fill 3 splits → declines."""
    from src.data.preprocess import _resolve_group_key

    df = pd.DataFrame({"id": ["a", "a", "b", "b"], "label": ["x"] * 4})
    assert _resolve_group_key(df, None) is None


def test_assert_no_group_leakage_detects_a_planted_leak():
    """The guard must FAIL when a group id is planted in two splits."""
    from src.data.preprocess import Splits, assert_no_group_leakage

    shared = pd.DataFrame({"id": ["dup"], "topic": ["t"], "text": ["x"], "label": ["unclear"]})
    splits = Splits(train=shared.copy(), val=shared.iloc[:0].copy(), test=shared.copy())
    with pytest.raises(AssertionError, match="leaks across splits"):
        assert_no_group_leakage(splits)


def test_assert_no_group_leakage_returns_empty_when_no_group_column():
    """When no group column is present in all splits, nothing is checked."""
    from src.data.preprocess import Splits, assert_no_group_leakage

    frame = pd.DataFrame({"topic": ["t"], "text": ["x"], "label": ["unclear"]})
    splits = Splits(train=frame, val=frame, test=frame)
    assert assert_no_group_leakage(splits) == []


def test_assert_no_group_leakage_accepts_explicit_group_columns():
    """An explicit ``group_columns`` list is honored (not just auto-detect)."""
    from src.data.preprocess import Splits, assert_no_group_leakage

    train = pd.DataFrame({"video_id": ["a", "b"], "label": ["x", "y"]})
    val = pd.DataFrame({"video_id": ["c"], "label": ["x"]})
    test = pd.DataFrame({"video_id": ["d"], "label": ["y"]})
    splits = Splits(train=train, val=val, test=test)
    assert assert_no_group_leakage(splits, group_columns=["video_id"]) == ["video_id"]


# ---------------------------------------------------------------------------
# Loader thread-safety (topic_relevance.py + topic_reranker.py)
# ---------------------------------------------------------------------------


def test_topic_relevance_loader_loads_once_under_concurrency(monkeypatch):
    """Concurrent first-requests must trigger exactly ONE underlying load."""
    from src.inference import topic_relevance

    topic_relevance.load_topic_relevance_model.cache_clear()
    calls = {"n": 0}

    def slow_load():
        """Test helper: slow load."""
        calls["n"] += 1
        time.sleep(0.05)  # widen the race window
        return {"kind": "sklearn", "model": object()}

    monkeypatch.setattr(topic_relevance, "_load_topic_relevance_model_uncached", slow_load)

    results = []
    barrier = threading.Barrier(8)

    def worker():
        """Test helper: worker."""
        barrier.wait()
        results.append(topic_relevance.load_topic_relevance_model())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1
    # Every thread observed the same cached instance.
    assert all(r is results[0] for r in results)
    topic_relevance.load_topic_relevance_model.cache_clear()


def test_topic_reranker_loader_loads_once_under_concurrency(monkeypatch):
    """Same one-load-only guarantee for the reranker loader."""
    from src.inference import topic_reranker

    topic_reranker.load_topic_reranker_model.cache_clear()
    calls = {"n": 0}

    def slow_load():
        """Test helper: slow load."""
        calls["n"] += 1
        time.sleep(0.05)
        return {"vectorizer": object(), "classifier": object(), "multiLabelBinarizer": object()}

    monkeypatch.setattr(topic_reranker, "_load_topic_reranker_model_uncached", slow_load)

    results = []
    barrier = threading.Barrier(8)

    def worker():
        """Test helper: worker."""
        barrier.wait()
        results.append(topic_reranker.load_topic_reranker_model())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1
    assert all(r is results[0] for r in results)
    topic_reranker.load_topic_reranker_model.cache_clear()


def test_topic_loaders_use_joblib_not_pickle(monkeypatch, tmp_path: Path):
    """The sklearn fallback must deserialize via joblib (hardening), not pickle."""
    import joblib

    from src.inference import topic_reranker

    topic_reranker.load_topic_reranker_model.cache_clear()
    bundle = {"vectorizer": object(), "classifier": object(), "multiLabelBinarizer": object()}
    joblib.dump(bundle, tmp_path / topic_reranker.MODEL_FILE)

    cfg = dataclasses.replace(topic_reranker.config, topic_reranker_model_dir=tmp_path)
    monkeypatch.setattr(topic_reranker, "config", cfg)

    sentinel = object()
    monkeypatch.setattr(topic_reranker.joblib, "load", lambda _p: sentinel)
    assert topic_reranker._load_topic_reranker_model_uncached() is sentinel
    topic_reranker.load_topic_reranker_model.cache_clear()


# ---------------------------------------------------------------------------
# Exact error-code-per-contract + generic (path-free) messages
# ---------------------------------------------------------------------------


def _client():
    """Fresh TestClient over the FastAPI app (mock-inference mode)."""
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    return TestClient(api_main.app), api_main


def test_pydantic_failure_is_exactly_400_not_422():
    """Per the contract there is NO 422 — a schema failure is 400 INVALID_INPUT."""
    client, _ = _client()
    resp = client.post("/predict", json={"topic": "ai"})  # missing `text`
    assert resp.status_code == 400
    assert resp.json()["error"] == "INVALID_INPUT"


def test_predict_503_message_does_not_leak_model_path(monkeypatch):
    """A 503 must NOT echo the filesystem path from the FileNotFoundError."""
    client, api_main = _client()

    def missing(*_a, **_kw):
        """Test helper: missing."""
        raise FileNotFoundError("No trained model found at /app/models/secret-path")

    monkeypatch.setattr(api_main, "predict", missing)
    resp = client.post("/predict", json={"topic": "ai", "text": "hello"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "MODEL_NOT_LOADED"
    assert "/app/models" not in body["message"]
    assert "secret-path" not in body["message"]


def test_predict_400_message_does_not_leak_value_error_text(monkeypatch):
    """A 400 must NOT echo arbitrary ValueError text either."""
    client, api_main = _client()

    def bad(*_a, **_kw):
        """Test helper: bad."""
        raise ValueError("internal detail /tmp/leak")

    monkeypatch.setattr(api_main, "predict", bad)
    resp = client.post("/predict", json={"topic": "ai", "text": "hello"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "INVALID_INPUT"
    assert "/tmp/leak" not in resp.json()["message"]


def test_topic_relevance_503_message_is_generic(monkeypatch):
    """The relevance endpoint's 503 is also path-free."""
    client, api_main = _client()

    def missing(*_a, **_kw):
        """Test helper: missing."""
        raise FileNotFoundError("No topic relevance model found at /srv/models/rel")

    monkeypatch.setattr(api_main, "predict_topic_relevance", missing)
    resp = client.post("/predict-topic-relevance", json={"topic": "ai", "text": "hi"})
    assert resp.status_code == 503
    assert "/srv/models" not in resp.json()["message"]


def test_topic_relevance_400_message_is_generic(monkeypatch):
    """The relevance endpoint's 400 is also generic."""
    client, api_main = _client()

    def bad(*_a, **_kw):
        """Test helper: bad."""
        raise ValueError("leak /etc/x")

    monkeypatch.setattr(api_main, "predict_topic_relevance", bad)
    resp = client.post("/predict-topic-relevance", json={"topic": "ai", "text": "hi"})
    assert resp.status_code == 400
    assert "/etc/x" not in resp.json()["message"]


def test_topics_503_and_400_messages_are_generic(monkeypatch):
    """The topics endpoint's 503 + 400 are also path-free."""
    client, api_main = _client()

    monkeypatch.setattr(
        api_main,
        "predict_topic_candidates",
        mock.Mock(side_effect=FileNotFoundError("No topic reranker model found at /m/r")),
    )
    resp = client.post("/predict-topics", json={"text": "ai policy"})
    assert resp.status_code == 503
    assert "/m/r" not in resp.json()["message"]

    monkeypatch.setattr(
        api_main,
        "predict_topic_candidates",
        mock.Mock(side_effect=ValueError("leak /v")),
    )
    resp = client.post("/predict-topics", json={"text": "ai policy"})
    assert resp.status_code == 400
    assert "/v" not in resp.json()["message"]


def test_no_handler_message_contains_mojibake():
    """Guard the (previously-corrupted) em-dash: no mojibake on any 500."""
    client, api_main = _client()
    for endpoint, payload, target in (
        ("/predict", {"topic": "a", "text": "b"}, "predict"),
        ("/predict-topic-relevance", {"topic": "a", "text": "b"}, "predict_topic_relevance"),
        ("/predict-topics", {"text": "b"}, "predict_topic_candidates"),
    ):
        with mock.patch.object(api_main, target, mock.Mock(side_effect=RuntimeError("x"))):
            resp = client.post(endpoint, json=payload)
        assert resp.status_code == 500
        assert "\u00e2\u20ac" not in resp.json()["message"]  # mojibake byte-pair
        assert "Internal server error" in resp.json()["message"]


# ---------------------------------------------------------------------------
# /health surfaces get_load_error() + lifespan warmup runs
# ---------------------------------------------------------------------------


def test_health_reports_load_error_as_degraded(monkeypatch):
    """When a prior STANCE load failed, /health is 'degraded' and labels it."""
    client, api_main = _client()
    monkeypatch.setattr(api_main, "get_load_error", lambda: "boom: model corrupt")
    monkeypatch.setattr(api_main, "get_topic_relevance_load_error", lambda: None)
    monkeypatch.setattr(api_main, "get_topic_reranker_load_error", lambda: None)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["loadError"] == "stance: boom: model corrupt"


def test_health_ok_when_no_load_error(monkeypatch):
    """No load error on ANY model → status 'ok' and loadError None."""
    client, api_main = _client()
    monkeypatch.setattr(api_main, "get_load_error", lambda: None)
    monkeypatch.setattr(api_main, "get_topic_relevance_load_error", lambda: None)
    monkeypatch.setattr(api_main, "get_topic_reranker_load_error", lambda: None)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["loadError"] is None


def test_health_surfaces_topic_relevance_load_error(monkeypatch):
    """A topic-relevance warmup failure must degrade /health, not pass as ok.

    Before this fix the warmup ``except`` block only logged the failure, so
    a broken relevance model left /health falsely reporting "ok". Now its
    ``get_load_error()`` is aggregated into the response.
    """
    client, api_main = _client()
    monkeypatch.setattr(api_main, "get_load_error", lambda: None)
    monkeypatch.setattr(
        api_main, "get_topic_relevance_load_error", lambda: "relevance broke"
    )
    monkeypatch.setattr(api_main, "get_topic_reranker_load_error", lambda: None)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["loadError"] == "topicRelevance: relevance broke"


def test_health_aggregates_multiple_load_errors(monkeypatch):
    """All failing models are joined into one ``loadError`` string."""
    client, api_main = _client()
    monkeypatch.setattr(api_main, "get_load_error", lambda: "stance dead")
    monkeypatch.setattr(
        api_main, "get_topic_relevance_load_error", lambda: "relevance dead"
    )
    monkeypatch.setattr(
        api_main, "get_topic_reranker_load_error", lambda: "reranker dead"
    )
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["loadError"] == (
        "stance: stance dead; topicRelevance: relevance dead; "
        "topicReranker: reranker dead"
    )


def test_lifespan_warmup_runs_on_context_enter(monkeypatch):
    """Entering the TestClient context must invoke the warmup hook (lifespan)."""
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    called = {"n": 0}
    monkeypatch.setattr(api_main, "_warmup_model", lambda: called.__setitem__("n", called["n"] + 1))
    with TestClient(api_main.app):
        pass
    assert called["n"] == 1


# ---------------------------------------------------------------------------
# config env-var validation
# ---------------------------------------------------------------------------


def test_env_int_rejects_non_integer():
    """A non-integer env value raises a ConfigError naming the variable."""
    from src.config import ConfigError, _env_int

    with mock.patch.dict(os.environ, {"TT_TEST_INT": "not-a-number"}):
        with pytest.raises(ConfigError, match="TT_TEST_INT"):
            _env_int("TT_TEST_INT", 5)


def test_env_int_uses_default_when_blank_or_missing():
    """Blank/unset env value falls back to the default."""
    from src.config import _env_int

    with mock.patch.dict(os.environ, {"TT_TEST_INT": "  "}):
        assert _env_int("TT_TEST_INT", 7) == 7
    os.environ.pop("TT_TEST_INT", None)
    assert _env_int("TT_TEST_INT", 9) == 9
    with mock.patch.dict(os.environ, {"TT_TEST_INT": "11"}):
        assert _env_int("TT_TEST_INT", 0) == 11


def test_env_float_rejects_non_number():
    """A non-numeric env value raises a ConfigError naming the variable."""
    from src.config import ConfigError, _env_float

    with mock.patch.dict(os.environ, {"TT_TEST_FLOAT": "abc"}):
        with pytest.raises(ConfigError, match="TT_TEST_FLOAT"):
            _env_float("TT_TEST_FLOAT", 0.5)


def test_env_float_uses_default_and_parses_valid():
    """Blank/unset → default; a valid value parses."""
    from src.config import _env_float

    os.environ.pop("TT_TEST_FLOAT", None)
    assert _env_float("TT_TEST_FLOAT", 0.25) == 0.25
    with mock.patch.dict(os.environ, {"TT_TEST_FLOAT": "0.75"}):
        assert _env_float("TT_TEST_FLOAT", 0.0) == 0.75


# ---------------------------------------------------------------------------
# Device-aware eval selector (evaluate.py)
# ---------------------------------------------------------------------------


def test_select_device_prefers_cuda():
    """CUDA available → 'cuda' device."""
    from src.training.evaluate import _select_device

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        device=lambda name: name,
    )
    assert _select_device(fake_torch) == "cuda"


def test_select_device_uses_mps_when_no_cuda():
    """No CUDA but MPS available → 'mps' device."""
    from src.training.evaluate import _select_device

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        device=lambda name: name,
    )
    assert _select_device(fake_torch) == "mps"


def test_select_device_falls_back_to_cpu():
    """No CUDA, no MPS namespace → 'cpu' device."""
    from src.training.evaluate import _select_device

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name: name,
    )  # note: no `backends` attribute at all
    assert _select_device(fake_torch) == "cpu"


def test_select_device_cpu_when_mps_present_but_unavailable():
    """MPS namespace present but unavailable → 'cpu'."""
    from src.training.evaluate import _select_device

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        device=lambda name: name,
    )
    assert _select_device(fake_torch) == "cpu"


# ---------------------------------------------------------------------------
# predict.py schema-drift guard
# ---------------------------------------------------------------------------


def test_build_prediction_response_rejects_wrong_probs_length():
    """A probs vector shorter/longer than LABELS must raise (no silent mis-map)."""
    from src.inference.predict import _build_prediction_response

    with pytest.raises(ValueError, match="label schema"):
        _build_prediction_response("topic", "text", [0.5, 0.5], "v1")  # 2 != 5 labels


def test_build_prediction_response_accepts_correct_length():
    """A correctly-sized probs vector maps cleanly to all labels."""
    from src.data.label_schema import LABELS
    from src.inference.predict import _build_prediction_response

    probs = [1.0 / len(LABELS)] * len(LABELS)
    out = _build_prediction_response("t", "x", probs, "v1")
    assert set(out["labelScores"]) == set(LABELS)
