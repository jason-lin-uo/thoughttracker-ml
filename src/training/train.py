"""
Fine-tune a pretrained transformer for transcript stance classification.

What this script does, top to bottom
------------------------------------
1. Read the labeled stance dataset from ``DATASET_PATH``.
2. Split into train / val / test reproducibly (seeded).
3. Tokenize via Hugging Face's ``AutoTokenizer`` for the base model
   (default: ``distilbert-base-uncased``).
4. Fine-tune for ``NUM_TRAIN_EPOCHS`` (default 3) using the HF Trainer
   API — handles batching, eval-on-epoch, best-model saving.
5. Predict on the held-out test split, compute final metrics.
6. Persist:
   - The full model + tokenizer to ``MODEL_DIR``
     (``save_pretrained``).
   - A ``model_card.json`` sidecar with version + base model + label
     schema (consumed by the FastAPI ``/health`` endpoint).
   - ``reports/metrics/test_metrics.json`` with the eval numbers.
   - ``reports/figures/confusion_matrix.png`` for the README.

Usage::

    python -m src.training.train
    # or override hyperparameters:
    DATASET_PATH=data/processed/stance_training.csv \\
      NUM_TRAIN_EPOCHS=5 TRAIN_BATCH_SIZE=32 \\
      python -m src.training.train

Why heavy imports are LAZY
--------------------------
torch, transformers, and datasets together take ~2-3 seconds to
import and pull in ~500 MB of memory. Doing those imports at module
top would slow every test module that imports this file. By
importing inside ``_train_with_transformers``, the rest of the
package (config, label_schema, load_dataset, FastAPI scaffolding)
stays cheap to import — important for fast test startup and for
environments where only inference is needed and the training stack
isn't installed.

Why ``main()`` catches ImportError + re-raises
----------------------------------------------
If a user runs ``python -m src.training.train`` without having
installed the requirements, they'd get a cryptic
``ModuleNotFoundError`` from inside the lazy import. By catching
it at the top level, we log a friendly "run pip install" message
before re-raising — better DX for first-time users.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..config import config
from ..data.label_schema import ID2LABEL, LABEL2ID, LABELS, num_labels
from ..data.load_dataset import load_stance_dataset
from ..data.preprocess import Splits, split_dataset, to_model_inputs
from ..utils.logging import get_logger
from ..utils.paths import FIGURES_DIR, METRICS_DIR, ensure_dirs
from .metrics import (
    build_classification_report,
    build_confusion_matrix,
    compute_metrics,
    save_confusion_matrix_png,
    save_metrics,
)

logger = get_logger("train")


def main(dataset_path: Optional[Path] = None) -> None:
    """Top-level training entry point. Orchestrates load → split → fine-
    tune → evaluate → persist.

    Parameters
    ----------
    dataset_path
        Override ``config.dataset_path``. Useful for the test suite,
        which calls ``main(some_tmp_csv)`` against a tiny synthetic
        dataframe to validate the pipeline without relying on the
        full gold-standard training corpus or running real backprop.
    """
    ensure_dirs()

    csv_path = dataset_path or config.dataset_path
    logger.info("Loading dataset from %s", csv_path)
    df = load_stance_dataset(csv_path)
    logger.info(
        "Loaded %d rows. Label distribution:\n%s", len(df), df["label"].value_counts()
    )

    splits = split_dataset(
        df, test_size=config.test_size, val_size=config.val_size, seed=config.seed
    )
    logger.info(
        "Splits — train=%d val=%d test=%d",
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )

    try:
        _train_with_transformers(splits)
    except ImportError as exc:
        logger.error(
            "Could not import transformers/torch. Install the full requirements first:\n"
            "    pip install -r requirements.txt\n"
            "Underlying error: %s",
            exc,
        )
        raise


def _train_with_transformers(splits: Splits) -> None:
    """The heavy-import inner training function.

    Why split out from ``main``: torch + transformers + datasets are
    only imported here, so a "I just want to validate dataset
    splitting" use case doesn't pay the ~3 second import cost.

    Steps (matches the module docstring):
      1. Lazy imports.
      2. Set the global PyTorch + numpy + transformers seed for
         reproducibility.
      3. Load tokenizer for ``config.base_model``.
      4. ``encode(texts, labels)`` builds a Hugging Face Dataset
         from a parallel ``(texts, labels)`` pair. The tokenizer is
         applied via ``ds.map(_tokenize_batch, batched=True)`` which is
         dramatically faster than per-row tokenization.
      5. Build the model with the right number of output labels
         and id2label/label2id maps (so saved model_card has the
         human-readable mapping).
      6. Configure ``TrainingArguments`` (eval-per-epoch,
         load-best-at-end with metric_for_best_model=f1_macro).
      7. Build the Trainer, train(), predict() on test.
      8. Persist model + tokenizer + model_card + metrics + figure.
    """
    # Heavy imports kept local so the rest of the package stays light.
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    logger.info("Base model: %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    # Use multiple CPU procs for tokenization so it doesn't bottleneck the
    # GPU loop on multi-core boxes. Cap at 4 to keep memory predictable; the
    # speedup tails off after that for typical batch sizes.
    tokenize_num_proc = min(4, max(1, (os.cpu_count() or 1)))

    def encode(texts, labels):
        """Build a tokenized Hugging Face ``Dataset`` from parallel lists.

        Wraps ``(texts, labels)`` in a ``Dataset`` and applies the
        tokenizer batched via ``.map`` (with multi-proc tokenization) —
        much faster than per-row tokenization.
        """
        ds = Dataset.from_dict({"text": texts, "label": labels})

        def _tokenize_batch(
            batch,
        ):  # pragma: no cover  (called by HF Dataset.map; covered by the real train run)
            """Tokenize one batch of texts (truncated to ``max_length``)."""
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=config.max_length,
            )

        return ds.map(_tokenize_batch, batched=True, num_proc=tokenize_num_proc)

    train_texts, train_labels = to_model_inputs(splits.train)
    val_texts, val_labels = to_model_inputs(splits.val)
    test_texts, test_labels = to_model_inputs(splits.test)
    use_class_weights = os.environ.get("USE_CLASS_WEIGHTS", "false").lower() in {
        "1",
        "true",
        "yes",
    }

    train_ds = encode(train_texts, train_labels)
    val_ds = encode(val_texts, val_labels)
    test_ds = encode(test_texts, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=num_labels(),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.model_dir / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
        seed=config.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=min(8, max(1, os.cpu_count() or 1)),
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    def _compute_trainer_metrics(
        eval_pred,
    ):  # pragma: no cover  (HF Trainer callback; covered by the real train run)
        """HF Trainer ``compute_metrics`` callback.

        Argmaxes the logits and delegates to the shared
        :func:`compute_metrics`, asserting ``f1_macro`` is present (the
        metric the Trainer selects the best checkpoint on).
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = compute_metrics(labels, preds)
        # `metric_for_best_model` above is "f1_macro" — fail loudly if the
        # metrics dict doesn't contain that key, instead of letting HF
        # silently fall back to "loss" and picking the wrong best checkpoint.
        if "f1_macro" not in metrics:
            raise RuntimeError(
                f"compute_metrics must return 'f1_macro' for best-checkpoint selection; got keys {sorted(metrics)}"
            )
        return metrics

    class WeightedTrainer(Trainer):
        """Trainer variant with inverse-frequency class weights.

        Opt-in via USE_CLASS_WEIGHTS=true. This is useful for ThoughtTracker's
        human-reviewed labels, where `unclear` can otherwise dominate the loss.
        """

        def __init__(self, *args, class_weights=None, **kwargs):
          """Store optional ``class_weights`` then defer to ``Trainer.__init__``."""
          super().__init__(*args, **kwargs)
          self.class_weights = class_weights

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):  # pragma: no cover  (HF Trainer callback; covered by the real train run)
            """Cross-entropy loss, optionally weighted by ``class_weights``.

            Falls through to the stock ``Trainer.compute_loss`` when no
            class weights are configured; otherwise applies an
            inverse-frequency-weighted ``CrossEntropyLoss`` so a dominant
            label (e.g. ``unclear``) doesn't swamp the gradient.
            """
            if self.class_weights is None:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    class_weights = None
    if use_class_weights:
        counts = np.bincount(train_labels, minlength=num_labels()).astype(float)
        safe_counts = np.maximum(counts, 1.0)
        weights = len(train_labels) / (num_labels() * safe_counts)
        weights = weights / weights.mean()
        weights = np.clip(weights, 0.2, 6.0)
        class_weights = torch.tensor(weights, dtype=torch.float)
        logger.info(
            "Using class-weighted loss: %s",
            {ID2LABEL[i]: round(float(weight), 3) for i, weight in enumerate(weights)},
        )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_trainer_metrics,
        class_weights=class_weights,
    )

    logger.info("Starting training…")
    trainer.train()

    logger.info("Evaluating on held-out test split…")
    test_preds = trainer.predict(test_ds)
    test_pred_ids = test_preds.predictions.argmax(axis=-1).tolist()

    test_metrics = compute_metrics(test_labels, test_pred_ids)
    test_report = build_classification_report(test_labels, test_pred_ids)
    confusion_matrix_values = build_confusion_matrix(test_labels, test_pred_ids)

    config.model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.model_dir))
    tokenizer.save_pretrained(str(config.model_dir))
    (config.model_dir / "model_card.json").write_text(
        json.dumps(
            {
                "modelVersion": config.model_version,
                "baseModel": config.base_model,
                "labels": list(LABELS),
                "maxLength": config.max_length,
            },
            indent=2,
        )
    )

    save_metrics(
        {
            "modelVersion": config.model_version,
            "baseModel": config.base_model,
            "testMetrics": test_metrics,
            "classificationReport": test_report,
            "confusionMatrix": confusion_matrix_values,
            "labels": list(LABELS),
        },
        METRICS_DIR / "test_metrics.json",
    )
    save_confusion_matrix_png(
        confusion_matrix_values,
        FIGURES_DIR / "confusion_matrix.png",
        title="Test confusion matrix",
    )

    logger.info("Saved model artifacts to %s", config.model_dir)
    logger.info("Saved metrics to %s", METRICS_DIR / "test_metrics.json")
    logger.info("Saved figure to %s", FIGURES_DIR / "confusion_matrix.png")


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover
