"""
Dataset loader + validator for the stance training corpus.

This is the single seam between "a CSV somewhere on disk" and the
training/evaluation code. Every code path that needs labeled stance
data goes through ``load_stance_dataset`` so the schema contract is
enforced exactly once.

Why a strict validator
----------------------
Stance datasets are usually hand-curated or scraped. A typo in the
label column (``Supprotive``) or a missing ``text`` column silently
corrupts training — you'd get a model that returns garbage and only
notice on the eval metrics. The validator catches these at load time
with a loud ``DatasetValidationError`` instead.

Schema we enforce
-----------------
Every CSV passed to ``load_stance_dataset`` must have these columns
(extra columns are fine, they're ignored):

  - ``id``: unique row id, kept around for debuggability + reproducibility.
  - ``topic``: the topic the chunk is being labeled against. Free-form
    natural language (e.g. ``"climate change is a real concern"``).
    Used by ``build_input_text`` to prompt the encoder.
  - ``text``: the chunk of transcript / tweet / paragraph to classify.
  - ``label``: must be one of the canonical labels in
    ``src.data.label_schema.LABELS`` (case-insensitive — we lowercase
    before validation).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .label_schema import LABELS


#: The four columns every stance dataset must carry. The order here is
#: documentation, not enforcement — we check membership, not position.
REQUIRED_COLUMNS: List[str] = ["id", "topic", "text", "label"]


class DatasetValidationError(Exception):
    """Raised when a CSV doesn't satisfy the schema contract.

    Distinct class (not bare ``ValueError``) so callers can ``except``
    just this one if they want to wrap validation failures with a
    helpful CLI hint.
    """


def load_stance_dataset(csv_path: Path) -> pd.DataFrame:
    """Load + validate + normalize a stance-labeled CSV.

    Steps:
      1. Existence check — explicit ``FileNotFoundError`` with a
         pointer to ``DATASET_PATH``, friendlier than pandas'
         default error.
      2. ``pd.read_csv`` — pandas handles encoding detection,
         quote-escaping, etc.
      3. Schema validation via ``validate_dataset``.
      4. Normalize: strip whitespace on every text column and
         lowercase the label so case differences don't break the
         label-lookup table.
      5. Reset index so downstream code can rely on contiguous
         0..N integer indexes.

    Parameters
    ----------
    csv_path
        Absolute or relative path to a CSV with the required columns.

    Returns
    -------
    A pandas DataFrame with at minimum ``id, topic, text, label``
    columns, all whitespace-normalized, labels lowercased.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` doesn't exist.
    DatasetValidationError
        If the CSV exists but the schema is broken.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Set DATASET_PATH in your .env "
            f"to a local stance CSV with id, topic, text, and label columns."
        )

    df = pd.read_csv(csv_path)
    validate_dataset(df)
    df["text"] = df["text"].astype(str).str.strip()
    df["topic"] = df["topic"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> None:
    """Assert the dataset matches our schema contract.

    Three checks:
      1. Every required column is present.
      2. The dataset has at least one row (a zero-row split is
         almost certainly an upstream pipeline bug, not "intentional").
      3. Every label, after lowercasing, is in the canonical
         ``LABELS`` set. Unknown labels usually mean the dataset
         used a different schema (e.g. ``FAVOR`` / ``AGAINST``) and
         skipped the adapter step.

    Raises ``DatasetValidationError`` on any failure with a message
    that names the specific thing that's wrong, not a generic
    "dataset invalid". The point is to make debugging a bad
    dataset take 30 seconds, not 30 minutes.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Dataset missing required columns: {missing}")

    if len(df) == 0:
        raise DatasetValidationError("Dataset is empty.")

    unknown = sorted(set(df["label"].astype(str).str.lower()) - set(LABELS))
    if unknown:
        raise DatasetValidationError(
            f"Dataset contains labels outside the supported schema: {unknown}. "
            f"Allowed labels: {list(LABELS)}"
        )


def build_input_text(topic: str, text: str) -> str:
    """Format one ``(topic, text)`` pair into the string the encoder sees.

    Why a delimiter
    ---------------
    Transformer encoders attend over a single sequence of tokens. To
    give the model a way to attend over the TOPIC separately from the
    TEXT, we delimit them with sentinel markers (``[TOPIC]`` /
    ``[TEXT]``). Without this, the model has no way to know "the topic
    is X, classify the rest" and ends up mixing topic words into the
    chunk representation, dragging accuracy down.

    Why this exact format
    ---------------------
    The markers are bracketed so they tokenize as distinct multi-
    character symbols rather than being broken into characters. They're
    NOT in the tokenizer's special-tokens vocabulary (which would
    require us to also register them at load time); they're plain
    text that happens to be visually distinctive.

    Consistency between training and inference matters: if training
    uses ``[TOPIC] climate [TEXT] foo`` and inference uses
    ``Topic: climate. Text: foo``, the model fails silently. Keep
    these two paths going through THIS function.

    Parameters
    ----------
    topic
        The topic / claim being asserted (free-form text).
    text
        The chunk to classify.

    Returns
    -------
    The combined input string, with whitespace trimmed on both
    pieces so leading/trailing newlines from a CSV don't end up in
    the encoder's input.
    """
    topic_clean = (topic or "").strip()
    text_clean = (text or "").strip()
    return f"[TOPIC] {topic_clean} [TEXT] {text_clean}"
