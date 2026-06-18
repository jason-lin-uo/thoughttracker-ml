"""
Re-evaluate a previously-trained stance model against the test split.

Why a separate evaluate.py
--------------------------
``train.py`` already evaluates on the test split as its final step.
This script is for the "I want to re-evaluate after some change"
cases:

  - I tweaked the test-split seed and want fresh numbers without
    retraining.
  - I swapped in a different held-out dataset (override
    ``dataset_path``).
  - I want to spot-check that a saved model still performs as the
    metrics report claimed.

Loading is intentionally per-example (no batching) so the code is
easy to read end-to-end. Performance-critical re-eval would batch
the tokenizer calls, but we typically run this against a small
test split (a few hundred rows) where per-example is fine.

Output artifacts go to ``reports/metrics/eval_metrics.json`` and
``reports/figures/confusion_matrix_eval.png`` — distinct from the
training-time files so the two don't overwrite each other.

Usage::

    python -m src.training.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import config
from ..data.load_dataset import load_stance_dataset
from ..data.preprocess import split_dataset, to_model_inputs
from ..utils.logging import get_logger
from ..utils.paths import FIGURES_DIR, METRICS_DIR, ensure_dirs
from .metrics import (
    build_classification_report,
    build_confusion_matrix,
    compute_metrics,
    save_confusion_matrix_png,
    save_metrics,
)

logger = get_logger("evaluate")


def _select_device(torch):
    """Pick the best available torch device: CUDA > MPS > CPU.

    ``torch`` is passed in rather than imported at module top so this
    module stays cheap to import (and so the test suite can hand in a
    fake torch). Apple-Silicon ``mps`` is checked via
    ``torch.backends.mps.is_available`` behind a ``getattr`` guard,
    because older torch builds don't expose the ``backends.mps``
    namespace at all and a bare attribute access would ``AttributeError``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(dataset_path: Optional[Path] = None) -> None:
    """Re-evaluate the saved model against the test split.

    Steps:
      1. Verify a saved model exists at ``config.model_dir``.
         Fail loudly if not — re-evaluating a missing model isn't
         a recoverable condition; the operator needs to run training
         first.
      2. Load + split the dataset using the same seed as train.py so
         the test split is byte-identical to the one the model was
         originally trained against (modulo the optional
         ``dataset_path`` override).
      3. Load model + tokenizer, moving the model to the best available
         device (CUDA > Apple-Silicon MPS > CPU). Earlier this ran
         CPU-only regardless of hardware; on a GPU box that made re-eval
         needlessly slow.
      4. Per-example forward pass with ``torch.no_grad()``. Inputs are
         moved to the same device as the model; logits are pulled back
         to CPU before converting to numpy.
      5. Compute metrics + classification report + confusion matrix.
      6. Persist to ``reports/metrics/eval_metrics.json`` and
         ``reports/figures/confusion_matrix_eval.png``.

    Parameters
    ----------
    dataset_path
        Override ``config.dataset_path``. Useful for re-evaluating
        the same model against a different held-out corpus.

    Raises
    ------
    FileNotFoundError
        If no trained model exists at ``config.model_dir``.
    """
    ensure_dirs()

    if not (config.model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"No trained model found at {config.model_dir}. "
            f"Run `python -m src.training.train` first."
        )

    df = load_stance_dataset(dataset_path or config.dataset_path)
    splits = split_dataset(
        df, test_size=config.test_size, val_size=config.val_size, seed=config.seed
    )
    test_texts, test_labels = to_model_inputs(splits.test)

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(config.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(config.model_dir))
    device = _select_device(torch)
    model.to(device)
    model.eval()
    logger.info("Evaluating on device: %s", device)

    pred_ids = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            # `.detach().cpu()` pulls the tensor off GPU/MPS before numpy()
            # — numpy can't read a CUDA/MPS tensor directly.
            logits = model(**inputs).logits.detach().cpu().numpy()[0]
            pred_ids.append(int(np.argmax(logits)))

    test_metrics = compute_metrics(test_labels, pred_ids)
    test_report = build_classification_report(test_labels, pred_ids)
    confusion_matrix_values = build_confusion_matrix(test_labels, pred_ids)

    save_metrics(
        {
            "modelVersion": config.model_version,
            "testMetrics": test_metrics,
            "classificationReport": test_report,
            "confusionMatrix": confusion_matrix_values,
        },
        METRICS_DIR / "eval_metrics.json",
    )
    save_confusion_matrix_png(
        confusion_matrix_values,
        FIGURES_DIR / "confusion_matrix_eval.png",
        title="Eval confusion matrix",
    )

    logger.info("Evaluation metrics:\n%s", json.dumps(test_metrics, indent=2))
    logger.info("Saved metrics to %s", METRICS_DIR / "eval_metrics.json")
    logger.info("Saved figure to %s", FIGURES_DIR / "confusion_matrix_eval.png")


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover
