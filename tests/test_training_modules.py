"""Tests for src/training/evaluate.py + src/training/train.py.

These modules drive the full training/eval pipeline. We mock transformers /
torch / datasets so the loops run end-to-end on a tiny synthetic dataframe
without downloading model weights or doing real backprop.
"""

import dataclasses
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.data.label_schema import LABELS


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _balanced_df(n_per_label: int = 5) -> pd.DataFrame:
    """Test helper: balanced df."""
    rows = []
    rid = 0
    for label in LABELS:
        for i in range(n_per_label):
            rows.append(
                {
                    "id": f"r{rid}",
                    "topic": "topic-x",
                    "text": f"{label} ex {i}",
                    "label": label,
                }
            )
            rid += 1
    return pd.DataFrame(rows)


def _build_fake_torch():
    """Test helper: build fake torch."""
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

    return fake_torch


def _build_fake_logits(label_idx: int):
    """Return a (1, num_labels) array where argmax falls on label_idx."""
    vec = np.full(len(LABELS), -1.0)
    vec[label_idx] = 5.0
    return vec.reshape(1, -1)


def _build_fake_tokenizer():
    """Test helper: build fake tokenizer."""
    tok = mock.MagicMock()
    # When called returns a dict-like inputs blob; downstream model accepts **inputs.
    tok.return_value = mock.MagicMock(__iter__=lambda self: iter([]))
    return tok


def _build_fake_model(logits: np.ndarray):
    """Test helper: build fake model."""
    model = mock.MagicMock()
    model.eval = mock.MagicMock()
    model.return_value.logits.numpy.return_value = logits
    # `model(**inputs).logits.numpy()[0]` — make numpy() return a 2D array.
    return model


# ----------------------------------------------------------------------------
# Evaluate.main()
# ----------------------------------------------------------------------------


def test_evaluate_main_runs_against_fake_model(monkeypatch, tmp_path: Path):
    """Behavioral test: evaluate main runs against fake model."""
    from src.training import evaluate as eval_mod

    # Set up a dataset csv on disk.
    csv_path = tmp_path / "stance.csv"
    _balanced_df(5).to_csv(csv_path, index=False)

    # Create the model_dir with a config.json so eval doesn't bail.
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"

    # Patch the config + path constants
    new_cfg = dataclasses.replace(
        eval_mod.config,
        model_dir=model_dir,
        dataset_path=csv_path,
        test_size=0.2,
        val_size=0.1,
        seed=42,
    )
    monkeypatch.setattr(eval_mod, "config", new_cfg)
    monkeypatch.setattr(eval_mod, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(eval_mod, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(eval_mod, "ensure_dirs", lambda: None)

    # Fake transformers + torch via sys.modules.
    fake_tokenizer_cls = mock.MagicMock()
    # The tokenizer output is a dict of tensors; each value must support
    # ``.to(device)`` (the device-aware eval moves inputs onto the model's
    # device). A MagicMock value returns itself from `.to(...)`.
    fake_tensor = mock.MagicMock()
    fake_tensor.to.return_value = fake_tensor
    fake_tokenizer = mock.MagicMock(return_value={"input_ids": fake_tensor})
    fake_tokenizer_cls.from_pretrained.return_value = fake_tokenizer

    fake_model_cls = mock.MagicMock()
    fake_model = mock.MagicMock()
    fake_model.eval = mock.MagicMock()
    # Each call returns logits that argmax to label index 0 (supportive).
    # Device-aware eval reads `.logits.detach().cpu().numpy()`, so pin the
    # value at the end of that chain.
    fake_model.return_value.logits.detach.return_value.cpu.return_value.numpy.return_value = _build_fake_logits(
        0
    )
    fake_model_cls.from_pretrained.return_value = fake_model

    fake_transformers = mock.MagicMock(
        AutoTokenizer=fake_tokenizer_cls,
        AutoModelForSequenceClassification=fake_model_cls,
    )

    # A fake torch whose device selection lands on CPU so we don't depend
    # on the host having CUDA/MPS during the test.
    fake_torch = _build_fake_torch()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.device = lambda name: name

    with mock.patch.dict(
        sys.modules,
        {"transformers": fake_transformers, "torch": fake_torch},
    ):
        eval_mod.main()

    # Output artifacts written.
    eval_json = metrics_dir / "eval_metrics.json"
    cm_png = figures_dir / "confusion_matrix_eval.png"
    assert eval_json.exists()
    assert cm_png.exists()
    payload = json.loads(eval_json.read_text())
    assert "testMetrics" in payload
    assert "confusionMatrix" in payload


def test_evaluate_main_raises_when_no_model(monkeypatch, tmp_path: Path):
    """Behavioral test: evaluate main raises when no model."""
    from src.training import evaluate as eval_mod

    csv_path = tmp_path / "stance.csv"
    _balanced_df(2).to_csv(csv_path, index=False)

    new_cfg = dataclasses.replace(
        eval_mod.config,
        model_dir=tmp_path / "nope",
        dataset_path=csv_path,
    )
    monkeypatch.setattr(eval_mod, "config", new_cfg)
    monkeypatch.setattr(eval_mod, "ensure_dirs", lambda: None)

    with pytest.raises(FileNotFoundError):
        eval_mod.main()


# ----------------------------------------------------------------------------
# Train.main()
# ----------------------------------------------------------------------------


def test_train_main_runs_with_fake_transformers(monkeypatch, tmp_path: Path):
    """Behavioral test: train main runs with fake transformers."""
    from src.training import train as train_mod

    monkeypatch.setenv("USE_CLASS_WEIGHTS", "true")

    csv_path = tmp_path / "stance.csv"
    _balanced_df(12).to_csv(csv_path, index=False)

    model_dir = tmp_path / "model_out"
    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"

    new_cfg = dataclasses.replace(
        train_mod.config,
        model_dir=model_dir,
        dataset_path=csv_path,
        test_size=0.2,
        val_size=0.1,
        seed=42,
        num_train_epochs=1,
    )
    monkeypatch.setattr(train_mod, "config", new_cfg)
    monkeypatch.setattr(train_mod, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(train_mod, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(train_mod, "ensure_dirs", lambda: None)

    # Fake every transformer / dataset / torch import.
    fake_tokenizer_cls = mock.MagicMock()
    fake_tokenizer = mock.MagicMock(return_value={"input_ids": "x"})
    fake_tokenizer.save_pretrained = mock.MagicMock()
    fake_tokenizer_cls.from_pretrained.return_value = fake_tokenizer

    fake_model_cls = mock.MagicMock()
    fake_model = mock.MagicMock()
    fake_model.save_pretrained = mock.MagicMock()
    fake_model_cls.from_pretrained.return_value = fake_model

    # Trainer behavior: provide a .train() and a .predict() that yields one
    # logit per example with consistent label.
    class _FakePredictionOutput:
        """Test double / fixture: FakePredictionOutput."""
        def __init__(self, n: int):
            # Use the first label for every row.
            """Test helper: init."""
            self.predictions = np.tile(_build_fake_logits(0), (n, 1))
            self.label_ids = np.zeros(n, dtype=int)
            self.metrics = {"test_accuracy": 1.0}

    def _predict(ds):
        # ds is a Dataset; len() works on it.
        """Test helper: predict."""
        try:
            n = len(ds)
        except TypeError:
            n = 4
        return _FakePredictionOutput(n)

    fake_trainers = []

    class _FakeTrainer:
        """Test double / fixture: FakeTrainer."""
        def __init__(self, *args, **kwargs):
            """Test helper: init."""
            self.args = args
            self.kwargs = kwargs
            self.train = mock.MagicMock()
            self.save_model = mock.MagicMock()
            self.predict = mock.MagicMock(side_effect=_predict)
            fake_trainers.append(self)

    fake_training_args = mock.MagicMock()

    fake_collator = mock.MagicMock()

    fake_set_seed = mock.MagicMock()

    fake_transformers = mock.MagicMock(
            AutoTokenizer=fake_tokenizer_cls,
            AutoModelForSequenceClassification=fake_model_cls,
            DataCollatorWithPadding=mock.MagicMock(return_value=fake_collator),
            Trainer=_FakeTrainer,
            TrainingArguments=mock.MagicMock(return_value=fake_training_args),
            set_seed=fake_set_seed,
        )

    # datasets.Dataset.from_dict — return a thin object with __len__ + map().
    class _FakeDataset:
        """Test double / fixture: FakeDataset."""
        def __init__(self, payload):
            """Test helper: init."""
            self.payload = payload

        def __len__(self):
            """Test helper: len."""
            return len(self.payload.get("text", []))

        def map(self, fn, batched=True, num_proc=None):
            # num_proc was added to encode() so multi-core tokenization
            # speeds up real training. The fake Dataset ignores it.
            """Test helper: map."""
            del num_proc
            return self

    fake_datasets = mock.MagicMock()
    fake_datasets.Dataset.from_dict.side_effect = lambda payload: _FakeDataset(payload)

    fake_torch = _build_fake_torch()

    with mock.patch.dict(
        sys.modules,
        {
            "transformers": fake_transformers,
            "datasets": fake_datasets,
            "torch": fake_torch,
        },
    ):
        train_mod.main()

    # Artifacts saved.
    assert fake_trainers
    assert fake_trainers[0].class_weights is not None
    fake_trainers[0].save_model.assert_called()
    fake_tokenizer.save_pretrained.assert_called()
    assert (metrics_dir / "test_metrics.json").exists()
    assert (figures_dir / "confusion_matrix.png").exists()
    # model_card.json is written into config.model_dir
    assert (model_dir / "model_card.json").exists()


def test_train_main_raises_when_transformers_missing(monkeypatch, tmp_path: Path):
    """Behavioral test: train main raises when transformers missing."""
    from src.training import train as train_mod

    csv_path = tmp_path / "stance.csv"
    _balanced_df(12).to_csv(csv_path, index=False)

    new_cfg = dataclasses.replace(
        train_mod.config,
        dataset_path=csv_path,
    )
    monkeypatch.setattr(train_mod, "config", new_cfg)
    monkeypatch.setattr(train_mod, "ensure_dirs", lambda: None)

    # Force _train_with_transformers to raise ImportError so we exercise the
    # try/except in main() that logs + re-raises.
    def _raise(*_a, **_kw):
        """Test helper: raise."""
        raise ImportError("simulated transformers missing")

    monkeypatch.setattr(train_mod, "_train_with_transformers", _raise)

    with pytest.raises(ImportError):
        train_mod.main()
