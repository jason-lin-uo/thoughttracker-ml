"""
coverage_lift.py — targeted tests that mop up the last uncovered lines
across the ML package. Each test pins ONE specific uncovered branch
identified by ``pytest --cov-report=term-missing`` so a future change
that breaks the branch shows up immediately.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest


# ----------------------------------------------------------------------------
# src/data/load_dataset.py — FileNotFoundError + DatasetValidationError paths
# ----------------------------------------------------------------------------


def test_load_stance_dataset_raises_for_missing_file(tmp_path: Path):
    """Exercise the explicit FileNotFoundError branch."""
    from src.data.load_dataset import load_stance_dataset

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_stance_dataset(tmp_path / "nope.csv")


def test_validate_dataset_rejects_empty_frame():
    """Exercise the `len(df) == 0` branch in validate_dataset."""
    from src.data.load_dataset import DatasetValidationError, validate_dataset

    df = pd.DataFrame({"id": [], "topic": [], "text": [], "label": []})
    with pytest.raises(DatasetValidationError, match="empty"):
        validate_dataset(df)


def test_validate_dataset_rejects_unknown_label():
    """Exercise the unknown-label branch."""
    from src.data.load_dataset import DatasetValidationError, validate_dataset

    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "topic": ["x", "x"],
            "text": ["t1", "t2"],
            "label": ["supportive", "bogus_label"],
        }
    )
    with pytest.raises(DatasetValidationError, match="bogus_label"):
        validate_dataset(df)


# ----------------------------------------------------------------------------
# src/api/main.py — ValidationError handler (the second one)
# ----------------------------------------------------------------------------


def test_validation_handler_returns_400_for_pydantic_validation_error():
    """Force a Pydantic ValidationError inside a handler so the
    second exception handler (the one for ``pydantic.ValidationError``,
    not ``RequestValidationError``) actually fires."""
    os.environ["ENABLE_MOCK_INFERENCE"] = "true"

    from fastapi.testclient import TestClient
    from pydantic import BaseModel, ValidationError

    from src.api import main as api_main

    class _Probe(BaseModel):
        """Test double / fixture: Probe."""
        x: int

    @api_main.app.get("/__cov_probe__")
    def _probe():
        # Constructing the model with the wrong type raises a
        # ``pydantic.ValidationError`` (not ``RequestValidationError``),
        # which the second handler intercepts.
        """Test helper: probe."""
        try:
            _Probe(x="not an int")  # type: ignore[arg-type]
        except ValidationError as exc:
            raise exc
        return {"ok": True}

    client = TestClient(api_main.app)
    resp = client.get("/__cov_probe__")
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "INVALID_INPUT"


# ----------------------------------------------------------------------------
# src/inference/predict.py — _predict_real path
# ----------------------------------------------------------------------------


def test_predict_real_path_via_mocked_transformers(monkeypatch):
    """Exercise the real-model inference branch with a mock model so
    we don't need an actual saved DistilBERT."""
    import sys

    from src.inference import model_loader, predict as predict_mod

    # Fake LoadedModel — load_model returns this without hitting disk.
    fake_loaded = model_loader.LoadedModel(
        tokenizer=mock.MagicMock(),
        model=mock.MagicMock(),
        model_version="test-v1",
        base_model="test-base",
    )
    monkeypatch.setattr(predict_mod, "load_model", lambda: fake_loaded)
    monkeypatch.setattr(predict_mod, "is_model_available", lambda: True)

    # Fake torch — softmax of a fixed logit vector resolves to "supportive".
    fake_torch = mock.MagicMock()

    class _NoGrad:
        """Test double / fixture: NoGrad."""
        def __enter__(self):
            """Test helper: enter."""
            return self

        def __exit__(self, *exc):
            """Test helper: exit."""
            return False

    fake_torch.no_grad = _NoGrad

    # The inference path does:
    #   logits = model(**inputs).logits[0]
    #   probs = torch.nn.functional.softmax(logits, dim=-1).tolist()
    fake_input = mock.MagicMock()
    fake_input.to.return_value = fake_input
    fake_loaded.tokenizer.return_value = {"input_ids": fake_input}
    fake_loaded.model.parameters.return_value = iter([mock.MagicMock(device="cpu")])
    logits_tensor = mock.MagicMock()
    fake_loaded.model.return_value.logits.__getitem__ = lambda _self, _i: logits_tensor
    fake_torch.nn.functional.softmax.return_value.tolist.return_value = [
        0.6,  # supportive
        0.1,  # opposed
        0.1,  # neutral
        0.1,  # mixed
        0.1,  # unclear
    ]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    out = predict_mod.predict(topic="x", text="some text")
    assert out["predictedLabel"] == "supportive"
    assert out["modelVersion"] == "test-v1"


# ----------------------------------------------------------------------------
# src/inference/predict.py — _softmax helper degenerate case
# ----------------------------------------------------------------------------


def test_softmax_with_all_zero_logits_still_sums_to_one():
    """Behavioral test: softmax with all zero logits still sums to one."""
    from src.inference.predict import _softmax

    out = _softmax([0.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(sum(out) - 1.0) < 1e-9
    # Should be uniform.
    assert all(abs(p - 0.2) < 1e-9 for p in out)


# ----------------------------------------------------------------------------
# src/inference/model_loader.py — RuntimeError when transformers is missing
# ----------------------------------------------------------------------------


def test_load_model_raises_runtime_error_when_transformers_unimportable(
    monkeypatch, tmp_path: Path
):
    """Trigger the ImportError-to-RuntimeError translation by hiding
    ``transformers`` from the import system."""
    import builtins
    import sys
    from src.inference import model_loader

    # Set up a model dir with config.json so the FileNotFoundError
    # branch doesn't short-circuit us.
    (tmp_path / "config.json").write_text("{}")
    new_cfg = dataclasses.replace(model_loader.config, model_dir=tmp_path)
    monkeypatch.setattr(model_loader, "config", new_cfg)
    monkeypatch.setattr(model_loader, "_loaded", None)
    monkeypatch.setattr(model_loader, "_load_error", None)

    # Remove any cached transformers + intercept the import.
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    real_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        """Test helper: stub import."""
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("simulated missing transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)

    with pytest.raises(RuntimeError, match="transformers is not installed"):
        model_loader.load_model(force=True)
