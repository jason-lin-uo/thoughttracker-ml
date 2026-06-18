"""Tests for src/inference/model_loader.py.

We don't have a saved transformer in CI, so most tests cover the
"not available" and "import-error" branches via monkeypatching the
frozen config dataclass via `dataclasses.replace`.
"""

import dataclasses
import json
import os
from pathlib import Path
from unittest import mock

os.environ["ENABLE_MOCK_INFERENCE"] = "true"
os.environ["MODEL_DIR"] = "models/__nonexistent__"

import pytest  # noqa: E402

from src.inference import model_loader  # noqa: E402


def _replace_model_dir(monkeypatch, model_dir: Path) -> None:
    """Test helper: replace model dir."""
    new_cfg = dataclasses.replace(model_loader.config, model_dir=model_dir)
    monkeypatch.setattr(model_loader, "config", new_cfg)


def test_is_model_available_false_for_missing_dir(monkeypatch, tmp_path: Path):
    """Behavioral test: is model available false for missing dir."""
    _replace_model_dir(monkeypatch, tmp_path / "nope")
    assert model_loader.is_model_available() is False


def test_is_model_available_true_when_config_json_exists(monkeypatch, tmp_path: Path):
    """Behavioral test: is model available true when config json exists."""
    (tmp_path / "config.json").write_text("{}")
    _replace_model_dir(monkeypatch, tmp_path)
    assert model_loader.is_model_available() is True


def test_load_model_raises_file_not_found_when_missing(monkeypatch, tmp_path: Path):
    """Behavioral test: load model raises file not found when missing."""
    _replace_model_dir(monkeypatch, tmp_path / "nope")
    monkeypatch.setattr(model_loader, "_loaded", None)
    monkeypatch.setattr(model_loader, "_load_error", None)
    with pytest.raises(FileNotFoundError):
        model_loader.load_model()
    assert model_loader.get_load_error() is not None


def test_load_model_caches_loaded_instance(monkeypatch):
    """Behavioral test: load model caches loaded instance."""
    sentinel = model_loader.LoadedModel(
        tokenizer="t", model="m", model_version="v1", base_model="base"
    )
    monkeypatch.setattr(model_loader, "_loaded", sentinel)
    out = model_loader.load_model(force=False)
    assert out is sentinel


def test_load_model_reads_model_card_for_base_model(monkeypatch, tmp_path: Path):
    """Behavioral test: load model reads model card for base model."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model_card.json").write_text(json.dumps({"baseModel": "my-base"}))
    _replace_model_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(model_loader, "_loaded", None)

    fake_tokenizer = mock.MagicMock()
    fake_model = mock.MagicMock()

    fake_at = mock.MagicMock()
    fake_at.from_pretrained = mock.MagicMock(return_value=fake_tokenizer)
    fake_am = mock.MagicMock()
    fake_am.from_pretrained = mock.MagicMock(return_value=fake_model)

    fake_transformers = mock.MagicMock(
        AutoTokenizer=fake_at, AutoModelForSequenceClassification=fake_am
    )

    with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
        loaded = model_loader.load_model(force=True)
    assert loaded.base_model == "my-base"
    assert loaded.tokenizer is fake_tokenizer
    assert loaded.model is fake_model


def test_load_model_falls_back_when_card_unparseable(monkeypatch, tmp_path: Path):
    """Behavioral test: load model falls back when card unparseable."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model_card.json").write_text("not json {")
    _replace_model_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(model_loader, "_loaded", None)

    fake_at = mock.MagicMock()
    fake_at.from_pretrained = mock.MagicMock(return_value=mock.MagicMock())
    fake_am = mock.MagicMock()
    fake_am.from_pretrained = mock.MagicMock(return_value=mock.MagicMock())

    fake_transformers = mock.MagicMock(
        AutoTokenizer=fake_at, AutoModelForSequenceClassification=fake_am
    )

    with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
        loaded = model_loader.load_model(force=True)
    # When the card is unreadable, base_model falls back to config.base_model.
    assert loaded.base_model == model_loader.config.base_model


def test_load_model_records_load_error_when_from_pretrained_raises(
    monkeypatch, tmp_path: Path
):
    """A corrupt / incompatible saved model raises inside from_pretrained.
    The loader must record `_load_error` (so /health reports the real reason
    instead of a stale/None error) AND re-raise so the caller fails loudly."""
    (tmp_path / "config.json").write_text("{}")
    _replace_model_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(model_loader, "_loaded", None)
    monkeypatch.setattr(model_loader, "_load_error", None)

    fake_at = mock.MagicMock()
    fake_at.from_pretrained = mock.MagicMock(side_effect=RuntimeError("corrupt model"))
    fake_am = mock.MagicMock()
    fake_am.from_pretrained = mock.MagicMock(return_value=mock.MagicMock())
    fake_transformers = mock.MagicMock(
        AutoTokenizer=fake_at, AutoModelForSequenceClassification=fake_am
    )

    with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
        with pytest.raises(RuntimeError, match="corrupt model"):
            model_loader.load_model(force=True)
    err = model_loader.get_load_error()
    assert err is not None and "Failed to load model" in err
