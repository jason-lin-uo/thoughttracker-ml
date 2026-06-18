import dataclasses
import os
import pickle
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


class FakeRelevanceModel:
    """Test double / fixture: FakeRelevanceModel."""
    classes_ = np.array(["irrelevant", "relevant"])

    def predict_proba(self, _rows):
        """Test helper: predict proba."""
        return np.array([[0.2, 0.8]])


class FakeVectorizer:
    """Test double / fixture: FakeVectorizer."""
    def transform(self, rows):
        """Test helper: transform."""
        return rows


class FakeClassifier:
    """Test double / fixture: FakeClassifier."""
    def predict_proba(self, _matrix):
        """Test helper: predict proba."""
        return [np.array([0.92, 0.48, 0.12])]


class FakeMultiLabelBinarizer:
    """Test double / fixture: FakeMultiLabelBinarizer."""
    classes_ = np.array(["topic_a", "topic_b", "topic_c"])


class TiedClassifier:
    """Classifier whose every class is equally probable — forces ties.

    Module-level (not nested in the test) so the bundle can be pickled to
    the on-disk reranker artifact the loader reads back.
    """

    def predict_proba(self, _matrix):
        """Return identical probabilities so the tie-break path is exercised."""
        return [np.array([0.5, 0.5, 0.5])]


@pytest.fixture(autouse=True)
def clear_model_caches():
    """Test helper: clear model caches."""
    from src.inference import topic_relevance, topic_reranker

    topic_relevance.load_topic_relevance_model.cache_clear()
    topic_reranker.load_topic_reranker_model.cache_clear()
    yield
    topic_relevance.load_topic_relevance_model.cache_clear()
    topic_reranker.load_topic_reranker_model.cache_clear()


def test_topic_relevance_availability_and_missing_model(monkeypatch, tmp_path: Path):
    """Behavioral test: topic relevance availability and missing model."""
    from src.inference import topic_relevance

    cfg = dataclasses.replace(topic_relevance.config, topic_relevance_model_dir=tmp_path)
    monkeypatch.setattr(topic_relevance, "config", cfg)

    assert topic_relevance.is_topic_relevance_model_available() is False
    with pytest.raises(FileNotFoundError):
        topic_relevance.load_topic_relevance_model()

    (tmp_path / topic_relevance.TRANSFORMER_CONFIG_FILE).write_text("{}", encoding="utf-8")
    assert topic_relevance.is_topic_relevance_model_available() is True


def test_topic_relevance_sklearn_prediction(monkeypatch, tmp_path: Path):
    """Behavioral test: topic relevance sklearn prediction."""
    from src.inference import topic_relevance

    model_path = tmp_path / topic_relevance.MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(FakeRelevanceModel(), handle)

    cfg = dataclasses.replace(
        topic_relevance.config,
        topic_relevance_model_dir=tmp_path,
        topic_relevance_model_version="rel-test-v1",
    )
    monkeypatch.setattr(topic_relevance, "config", cfg)

    out = topic_relevance.predict_topic_relevance("AI Policy", "This chunk is about AI policy.")
    assert out["predictedLabel"] == "relevant"
    assert out["confidence"] == 0.8
    assert out["labelScores"] == {"irrelevant": 0.2, "relevant": 0.8}
    assert out["modelVersion"] == "rel-test-v1"


def test_topic_relevance_validation_and_cli(monkeypatch, capsys):
    """Behavioral test: topic relevance validation and cli."""
    from src.inference import topic_relevance

    with pytest.raises(ValueError):
        topic_relevance.predict_topic_relevance("", "body")
    with pytest.raises(ValueError):
        topic_relevance.predict_topic_relevance("topic", " ")

    monkeypatch.setattr(
        topic_relevance,
        "predict_topic_relevance",
        lambda topic, text: {"topic": topic, "text": text, "predictedLabel": "relevant"},
    )
    assert topic_relevance.main(["--topic", "AI", "--text", "body"]) == 0
    assert '"predictedLabel": "relevant"' in capsys.readouterr().out

    monkeypatch.setattr(
        topic_relevance,
        "predict_topic_relevance",
        mock.Mock(side_effect=FileNotFoundError("missing rel model")),
    )
    assert topic_relevance.main(["--topic", "AI", "--text", "body"]) == 2
    assert "missing rel model" in capsys.readouterr().err

    monkeypatch.setattr(
        topic_relevance,
        "predict_topic_relevance",
        mock.Mock(side_effect=ValueError("bad rel request")),
    )
    assert topic_relevance.main(["--topic", "AI", "--text", "body"]) == 1
    assert "bad rel request" in capsys.readouterr().err


def test_topic_relevance_transformer_load_and_predict(monkeypatch, tmp_path: Path):
    """Behavioral test: topic relevance transformer load and predict."""
    from src.inference import topic_relevance

    (tmp_path / topic_relevance.TRANSFORMER_CONFIG_FILE).write_text("{}", encoding="utf-8")
    cfg = dataclasses.replace(topic_relevance.config, topic_relevance_model_dir=tmp_path)
    monkeypatch.setattr(topic_relevance, "config", cfg)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: name

    fake_tokenizer_cls = mock.MagicMock()
    fake_model_cls = mock.MagicMock()
    fake_model = mock.MagicMock()
    fake_model_cls.from_pretrained.return_value = fake_model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = fake_tokenizer_cls
    fake_transformers.AutoModelForSequenceClassification = fake_model_cls

    with mock.patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
        loaded = topic_relevance.load_topic_relevance_model()

    assert loaded["kind"] == "transformer"
    fake_model.to.assert_called_once_with("cpu")
    fake_model.eval.assert_called_once()

    fake_input = mock.MagicMock()
    fake_input.to.return_value = fake_input
    loaded["tokenizer"].return_value = {"input_ids": fake_input}
    loaded["model"].parameters.return_value = iter([types.SimpleNamespace(device="cpu")])
    loaded["model"].return_value = types.SimpleNamespace(logits=["fake-logits"])

    class NoGrad:
        """Test double / fixture: NoGrad."""
        def __enter__(self):
            """Test helper: enter."""
            return self

        def __exit__(self, *_exc):
            """Test helper: exit."""
            return False

    fake_torch.no_grad = NoGrad
    fake_torch.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(
            softmax=mock.MagicMock(return_value=types.SimpleNamespace(tolist=lambda: [0.25, 0.75]))
        )
    )

    cfg = dataclasses.replace(topic_relevance.config, topic_relevance_max_length=384)
    monkeypatch.setattr(topic_relevance, "config", cfg)

    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        assert topic_relevance._predict_transformer(loaded, "encoded text") == [0.25, 0.75]
    loaded["tokenizer"].assert_called_once_with(
        "encoded text",
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )


def test_topic_relevance_falls_back_to_pkl_when_transformer_config_is_a_stub(
    monkeypatch, tmp_path: Path
):
    """A stub config.json next to a valid .pkl must load the .pkl, not crash.

    This is the probe/load-branch disagreement the audit called out:
    ``is_topic_relevance_model_available`` returns True whenever EITHER a
    config.json OR a .pkl exists, but the load branch preferred the
    transformer whenever config.json merely existed. A half-written /
    stubbed config.json sitting beside a loadable sklearn fallback would
    therefore explode in ``from_pretrained`` even though the probe (and a
    perfectly good .pkl) said "available". The load branch now falls back
    to the .pkl when the transformer artifact fails to load, so the two
    agree.
    """
    from src.inference import topic_relevance

    # Stub transformer config (present but not a real HF artifact)…
    (tmp_path / topic_relevance.TRANSFORMER_CONFIG_FILE).write_text("{}", encoding="utf-8")
    # …plus a genuinely loadable sklearn fallback.
    model_path = tmp_path / topic_relevance.MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(FakeRelevanceModel(), handle)

    cfg = dataclasses.replace(topic_relevance.config, topic_relevance_model_dir=tmp_path)
    monkeypatch.setattr(topic_relevance, "config", cfg)

    # Probe says available (it counts the .pkl)…
    assert topic_relevance.is_topic_relevance_model_available() is True

    # …and the transformer load is forced to fail (mimicking a stub config),
    # so the loader must fall back to the sklearn .pkl rather than raise.
    monkeypatch.setattr(
        topic_relevance,
        "_load_transformer_relevance_model",
        mock.Mock(side_effect=RuntimeError("stub config.json is not a real HF artifact")),
    )
    loaded = topic_relevance.load_topic_relevance_model()
    assert loaded["kind"] == "sklearn"


def test_topic_relevance_transformer_failure_reraises_without_pkl(monkeypatch, tmp_path: Path):
    """A broken transformer config with NO .pkl fallback re-raises the real error.

    The fallback only kicks in when there's actually a .pkl to fall back
    to; otherwise the transformer load error is the genuine failure and
    must propagate (so /health surfaces it) rather than being swallowed
    into a generic FileNotFoundError.
    """
    from src.inference import topic_relevance

    (tmp_path / topic_relevance.TRANSFORMER_CONFIG_FILE).write_text("{}", encoding="utf-8")
    cfg = dataclasses.replace(topic_relevance.config, topic_relevance_model_dir=tmp_path)
    monkeypatch.setattr(topic_relevance, "config", cfg)

    monkeypatch.setattr(
        topic_relevance,
        "_load_transformer_relevance_model",
        mock.Mock(side_effect=RuntimeError("transformer artifact corrupt")),
    )
    with pytest.raises(RuntimeError, match="transformer artifact corrupt"):
        topic_relevance.load_topic_relevance_model()


def test_topic_reranker_availability_missing_and_prediction(monkeypatch, tmp_path: Path):
    """Behavioral test: topic reranker availability missing and prediction."""
    from src.inference import topic_reranker

    cfg = dataclasses.replace(topic_reranker.config, topic_reranker_model_dir=tmp_path)
    monkeypatch.setattr(topic_reranker, "config", cfg)

    assert topic_reranker.is_topic_reranker_model_available() is False
    with pytest.raises(FileNotFoundError):
        topic_reranker.load_topic_reranker_model()

    model_path = tmp_path / topic_reranker.MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "vectorizer": FakeVectorizer(),
                "classifier": FakeClassifier(),
                "multiLabelBinarizer": FakeMultiLabelBinarizer(),
                "modelVersion": "rerank-test-v1",
            },
            handle,
        )

    assert topic_reranker.is_topic_reranker_model_available() is True
    out = topic_reranker.predict_topic_candidates("AI policy and economics", limit=2, min_score=0.2)
    assert out == {
        "topics": [
            {"topicSlug": "topic_a", "confidence": 0.92},
            {"topicSlug": "topic_b", "confidence": 0.48},
        ],
        "modelVersion": "rerank-test-v1",
    }


def test_topic_reranker_validation_and_cli(monkeypatch, capsys):
    """Behavioral test: topic reranker validation and cli."""
    from src.inference import topic_reranker

    with pytest.raises(ValueError):
        topic_reranker.predict_topic_candidates("")
    with pytest.raises(ValueError):
        topic_reranker.predict_topic_candidates("body", limit=0)
    with pytest.raises(ValueError):
        topic_reranker.predict_topic_candidates("body", min_score=1.5)

    monkeypatch.setattr(
        topic_reranker,
        "predict_topic_candidates",
        lambda text, limit=12, min_score=0.2: {
            "topics": [{"topicSlug": "topic_a", "confidence": 0.9}],
            "modelVersion": f"{limit}:{min_score}:{text}",
        },
    )
    assert topic_reranker.main(["--text", "body", "--limit", "3", "--min-score", "0.4"]) == 0
    assert '"topicSlug": "topic_a"' in capsys.readouterr().out

    monkeypatch.setattr(
        topic_reranker,
        "predict_topic_candidates",
        mock.Mock(side_effect=FileNotFoundError("missing reranker model")),
    )
    assert topic_reranker.main(["--text", "body"]) == 2
    assert "missing reranker model" in capsys.readouterr().err

    monkeypatch.setattr(
        topic_reranker,
        "predict_topic_candidates",
        mock.Mock(side_effect=ValueError("bad reranker request")),
    )
    assert topic_reranker.main(["--text", "body"]) == 1
    assert "bad reranker request" in capsys.readouterr().err


def test_topic_reranker_tie_break_is_deterministic_by_label_order(monkeypatch, tmp_path: Path):
    """Equal probabilities must resolve to a stable, label-ordered ranking.

    Three topics here share the exact same probability (0.5). With the old
    unstable ``np.argsort(...)[::-1]`` the order on a tie was arbitrary; the
    stable ``argsort(-probs)`` fix guarantees the earlier label in
    ``mlb.classes_`` (topic_a, then topic_b, then topic_c) always comes
    first, so the candidate list is reproducible.
    """
    from src.inference import topic_reranker

    cfg = dataclasses.replace(topic_reranker.config, topic_reranker_model_dir=tmp_path)
    monkeypatch.setattr(topic_reranker, "config", cfg)
    model_path = tmp_path / topic_reranker.MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "vectorizer": FakeVectorizer(),
                "classifier": TiedClassifier(),
                "multiLabelBinarizer": FakeMultiLabelBinarizer(),
                "modelVersion": "rerank-tie-v1",
            },
            handle,
        )

    out = topic_reranker.predict_topic_candidates("anything", limit=3, min_score=0.2)
    assert [topic["topicSlug"] for topic in out["topics"]] == [
        "topic_a",
        "topic_b",
        "topic_c",
    ]


def test_topic_relevance_records_and_clears_load_error(monkeypatch, tmp_path: Path):
    """A failed relevance load is recorded; a later success clears it.

    This is the signal ``/health`` reads so a broken topic-relevance model
    reports as ``degraded`` instead of silently logging-and-passing.
    """
    from src.inference import topic_relevance

    cfg = dataclasses.replace(topic_relevance.config, topic_relevance_model_dir=tmp_path)
    monkeypatch.setattr(topic_relevance, "config", cfg)

    # No artifact on disk → load raises FileNotFoundError and the error is
    # captured for /health to read.
    assert topic_relevance.get_load_error() is None
    with pytest.raises(FileNotFoundError):
        topic_relevance.load_topic_relevance_model()
    assert topic_relevance.get_load_error() is not None
    assert "topic relevance model" in topic_relevance.get_load_error()

    # Now drop in a loadable sklearn fallback artifact; a fresh load should
    # succeed and clear the recorded error back to None.
    topic_relevance.load_topic_relevance_model.cache_clear()
    model_path = tmp_path / topic_relevance.MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(FakeRelevanceModel(), handle)
    loaded = topic_relevance.load_topic_relevance_model()
    assert loaded["kind"] == "sklearn"
    assert topic_relevance.get_load_error() is None


def _write_reranker_bundle(model_path: Path, *, model_version: str, classifier=None) -> None:
    """Pickle a minimal reranker bundle to ``model_path`` for the loader to read."""
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "vectorizer": FakeVectorizer(),
                "classifier": classifier or FakeClassifier(),
                "multiLabelBinarizer": FakeMultiLabelBinarizer(),
                "modelVersion": model_version,
            },
            handle,
        )


def test_topic_reranker_cache_invalidates_when_model_file_changes(monkeypatch, tmp_path: Path):
    """Swapping the .pkl on disk must invalidate the cached bundle.

    The old ``functools.lru_cache`` pinned the first-loaded bundle for the
    life of the process, so a retrained / hot-swapped model was silently
    ignored. The path+mtime cache key now reloads when the artifact's
    mtime changes, so the second load returns the NEW model version.
    """
    from src.inference import topic_reranker

    cfg = dataclasses.replace(topic_reranker.config, topic_reranker_model_dir=tmp_path)
    monkeypatch.setattr(topic_reranker, "config", cfg)
    model_path = tmp_path / topic_reranker.MODEL_FILE

    _write_reranker_bundle(model_path, model_version="rerank-v1")
    first = topic_reranker.load_topic_reranker_model()
    assert first["modelVersion"] == "rerank-v1"

    # Rewrite the artifact with a new version and bump mtime explicitly so
    # the change is detectable even within the same wall-clock second.
    _write_reranker_bundle(model_path, model_version="rerank-v2")
    stat = model_path.stat()
    os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    second = topic_reranker.load_topic_reranker_model()
    assert second["modelVersion"] == "rerank-v2", "stale bundle served after model swap"


def test_topic_reranker_cache_reused_when_file_unchanged(monkeypatch, tmp_path: Path):
    """An unchanged artifact must be served from cache (no redundant reload).

    Guards the hot path: once loaded, a stable file should NOT trigger a
    second ``joblib.load``. We spy on the uncached loader to prove the
    second call is a pure cache hit.
    """
    from src.inference import topic_reranker

    cfg = dataclasses.replace(topic_reranker.config, topic_reranker_model_dir=tmp_path)
    monkeypatch.setattr(topic_reranker, "config", cfg)
    model_path = tmp_path / topic_reranker.MODEL_FILE
    _write_reranker_bundle(model_path, model_version="rerank-stable")

    real_loader = topic_reranker._load_topic_reranker_model_uncached
    spy = mock.Mock(side_effect=real_loader)
    monkeypatch.setattr(topic_reranker, "_load_topic_reranker_model_uncached", spy)

    first = topic_reranker.load_topic_reranker_model()
    second = topic_reranker.load_topic_reranker_model()
    assert first is second
    assert spy.call_count == 1, "unchanged artifact should not reload"


def test_topic_reranker_records_and_clears_load_error(monkeypatch, tmp_path: Path):
    """A failed reranker load is recorded; a later success clears it."""
    from src.inference import topic_reranker

    cfg = dataclasses.replace(topic_reranker.config, topic_reranker_model_dir=tmp_path)
    monkeypatch.setattr(topic_reranker, "config", cfg)

    assert topic_reranker.get_load_error() is None
    with pytest.raises(FileNotFoundError):
        topic_reranker.load_topic_reranker_model()
    assert topic_reranker.get_load_error() is not None
    assert "topic reranker model" in topic_reranker.get_load_error()

    topic_reranker.load_topic_reranker_model.cache_clear()
    model_path = tmp_path / topic_reranker.MODEL_FILE
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "vectorizer": FakeVectorizer(),
                "classifier": FakeClassifier(),
                "multiLabelBinarizer": FakeMultiLabelBinarizer(),
                "modelVersion": "rerank-test-v1",
            },
            handle,
        )
    loaded = topic_reranker.load_topic_reranker_model()
    assert loaded["modelVersion"] == "rerank-test-v1"
    assert topic_reranker.get_load_error() is None


def test_load_topic_relevance_model_returns_cached_instance(monkeypatch):
    """A warm call returns the cached model via the double-checked-locking fast
    path. Covers the cache-hit early return without loading a real model — that
    line is otherwise only reached when an actual model loads (absent in CI)."""
    from src.inference import topic_relevance

    sentinel = {"kind": "transformer", "_cached_sentinel": True}
    monkeypatch.setattr(topic_relevance, "_cache", sentinel)
    assert topic_relevance.load_topic_relevance_model() is sentinel
