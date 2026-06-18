"""
Metric utilities shared by the training + evaluation scripts.

Why a separate module
---------------------
Both ``train.py`` (compute metrics during training + write final
report) and ``evaluate.py`` (re-evaluate a saved model against the
test split) need the same metric calculations. Centralizing here
guarantees both scripts compute identical numbers — important for
reproducibility and for "did this training run actually improve
over the previous one" comparisons.

Why scikit-learn for the math
-----------------------------
We're computing standard supervised-learning metrics (accuracy,
precision, recall, F1, confusion matrix). ``sklearn.metrics`` is
the canonical implementation; using it means our numbers match
what a researcher would compute from our JSON outputs in a notebook.

Two flavors of F1
-----------------
We report both ``f1_macro`` (unweighted mean across labels) and
``f1_weighted`` (weighted by support). These can disagree wildly on
imbalanced datasets — transcript corpora often have way more ``opposed`` than
``mixed``, so macro penalizes us harder for missing ``mixed`` calls
while weighted papers over it. We report both so callers can pick
the framing that matches their use case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..data.label_schema import LABELS


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Compute the five-metric summary we report for every training run.

    All metrics are coerced to plain Python ``float`` (not ``np.float64``)
    so the resulting dict serializes cleanly via ``json.dumps`` without
    needing a custom encoder.

    ``zero_division=0`` on precision/recall/F1 makes a label with zero
    predicted positives produce a score of 0 instead of warning + ``NaN``.
    Cleaner output, and the macro-F1 still reflects the missing
    coverage (a 0 drags the average down as intended).

    Parameters
    ----------
    y_true, y_pred
        Parallel sequences of integer label ids (0..len(LABELS)-1).
        Lengths must match.

    Returns
    -------
    dict with keys: ``accuracy``, ``precision_macro``, ``recall_macro``,
    ``f1_macro``, ``f1_weighted``. All values in [0, 1].
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    # Pin the label set so the macro/weighted averages are computed over ALL
    # classes, not just those PRESENT in this particular y_true/y_pred. Without
    # `labels=`, a batch/fold missing a class would silently average over fewer
    # labels, making metrics incomparable across runs (and inconsistent with
    # build_classification_report, which already pins them).
    all_labels = list(range(len(LABELS)))
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(
            precision_score(
                y_true_arr, y_pred_arr, labels=all_labels, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true_arr, y_pred_arr, labels=all_labels, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true_arr, y_pred_arr, labels=all_labels, average="macro", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true_arr, y_pred_arr, labels=all_labels, average="weighted", zero_division=0
            )
        ),
    }


def build_classification_report(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Produce a per-label precision/recall/F1 breakdown.

    Returns a dict keyed by label name (plus ``accuracy`` / ``macro avg``
    / ``weighted avg`` totals). We force ``labels=range(len(LABELS))`` so
    even labels with zero support in the test split show up in the
    output — otherwise downstream consumers (the JSON viewer in the
    UI, etc.) would need to handle "this label is missing" gracefully.

    ``zero_division=0`` is deliberate: a rare label with zero support (no
    examples of it in this split) reports precision/recall/F1 of 0 rather
    than warning + ``NaN``. That 0 is "we couldn't measure this class
    here", not "the model is broken" — it's by design so the report stays
    a clean, JSON-serializable, fully-populated table. (See the stance
    model card, where ``mixed``/``unclear`` land at 0 for exactly this
    reason.)
    """
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )


def build_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> List[List[int]]:
    """Return the confusion matrix as a plain nested-list ``List[List[int]]``.

    Rows = true labels, columns = predicted labels, in ``LABELS``
    order. Returned as plain Python lists (not numpy arrays) so
    callers can JSON-serialize the result without a special encoder.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return cm.tolist()


def save_metrics(payload: dict, out_path: Path) -> None:
    """Atomic-ish JSON write to disk.

    Creates the parent directory if missing. Indent 2 for human
    readability — these JSON files end up in PRs and code reviews.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def save_confusion_matrix_png(
    confusion_matrix_values: Sequence[Sequence[int]],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Render a confusion matrix as a labeled heatmap PNG.

    Matplotlib is imported INSIDE the function (not at module top)
    because pulling it in eagerly would slow every test module load
    by ~500 ms and pollute the test process with matplotlib's
    rcParams. The Agg backend is forced so this works in headless
    environments (CI, Docker, a tmux session over SSH).

    Visual choices:
      - ``cmap="Blues"`` — color-blind-friendly sequential palette.
      - Numeric annotation on each cell, colored white-on-dark-cells
        so the value stays readable regardless of color intensity.
      - 30-degree x-axis tick rotation so long label names don't overlap.
      - 150 DPI on save — sharp on retina screens, file size is still
        reasonable (~30 KB for a 5×5 matrix).

    Parameters
    ----------
    confusion_matrix_values
        The 2D matrix from ``build_confusion_matrix``.
    out_path
        Where to write the PNG. Parent dirs are created if needed.
    title
        Optional figure title (the eval script passes "Eval confusion
        matrix" so the file from train.py and the file from
        evaluate.py are visually distinguishable).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless / CI-friendly
    import matplotlib.pyplot as plt

    matrix_array = np.asarray(confusion_matrix_values)
    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap = ax.imshow(matrix_array, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticklabels(LABELS)

    for i in range(matrix_array.shape[0]):
        for j in range(matrix_array.shape[1]):
            value = int(matrix_array[i, j])
            # White text on the darker cells, black on the lighter
            # cells — keeps everything readable.
            color = "white" if value > matrix_array.max() / 2 else "black"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(heatmap, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
