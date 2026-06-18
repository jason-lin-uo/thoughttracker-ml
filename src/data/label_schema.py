"""Canonical stance-label schema for the ThoughtTracker ML classifier.

Why this lives in one file
--------------------------
Every part of the ML pipeline — dataset adapters, training, evaluation,
inference, the FastAPI service contract — references "the labels". If
those labels existed as a literal in each file, adding a label later
would mean hunting through five files and remembering one. By
centralizing here, a future "split `mixed` into `weakly_supportive` and
`weakly_opposed`" only touches this file plus the dataset.

Labels we use
-------------
The five labels the classifier ever returns:

  - ``supportive``: speaker explicitly endorses the topic / takes a
    positive stance toward it.
  - ``opposed``: speaker explicitly rejects / argues against the topic.
  - ``neutral``: speaker describes the topic factually with no
    discernible stance signal.
  - ``mixed``: speaker takes BOTH supportive and opposed positions in
    the same chunk (real-world case for nuanced takes).
  - ``unclear``: speaker mentions the topic but stance is ambiguous OR
    the chunk doesn't contain enough signal to call it.

Why not ``insufficient_evidence``?
----------------------------------
The main ThoughtTracker app (TypeScript backend) ALSO recognizes a
sixth label, ``insufficient_evidence``, applied when the relevance
score between a chunk and a topic is too low to even attempt
classification. That's a pre-classifier gate, not a model output —
the model sees only chunks that passed the gate, so its output space
stays clean. The backend's `mapMlLabelToDbLabel` helper handles the
collapse: an ML output of ``unclear`` with low confidence becomes
``insufficient_evidence`` in the DB.

Determinism
-----------
``LABEL2ID`` and ``ID2LABEL`` are derived from a tuple-literal
``LABELS`` so the mapping is stable across imports and across machines.
The Hugging Face Trainer needs integer labels; the FastAPI service
emits string labels — these two maps round-trip cleanly.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


#: Tuple instead of a list — immutable, so accidental ``LABELS.append(...)``
#: somewhere in the pipeline can't silently shift the integer indexes.
LABELS: Tuple[str, ...] = (
    "supportive",
    "opposed",
    "neutral",
    "mixed",
    "unclear",
)

#: Label → integer index. The integer is what the Hugging Face Trainer
#: sees during training and what the model emits in its logits tensor.
LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}

#: Integer index → label. Inverse of LABEL2ID; used at inference to
#: turn an argmax index back into the human-readable label the FastAPI
#: service emits.
ID2LABEL: Dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}


def num_labels() -> int:
    """Total label count — handed to the Hugging Face
    ``AutoModelForSequenceClassification`` so it builds a final layer
    with the right number of output neurons."""
    return len(LABELS)


def label_to_id(label: str) -> int:
    """Convert a string label to its integer index.

    Raises
    ------
    ValueError
        If ``label`` is not one of the canonical five. The error message
        lists the allowed labels so a typo in the dataset is caught
        loudly at load time, not silently mapped to a random index.
    """
    try:
        return LABEL2ID[label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown label {label!r}. Allowed labels: {list(LABELS)}"
        ) from exc


def id_to_label(idx: int) -> str:
    """Inverse of ``label_to_id``. Used at inference to turn the model's
    argmax index back into a human-readable label.

    Raises
    ------
    ValueError
        On any index outside ``0..len(LABELS)-1``. A model emitting an
        out-of-range index means something is structurally wrong (we
        loaded the wrong checkpoint, the schema drifted), so a loud
        ValueError is the right answer.
    """
    try:
        return ID2LABEL[idx]
    except KeyError as exc:
        raise ValueError(
            f"Unknown label id {idx!r}. Valid range: 0..{len(LABELS) - 1}"
        ) from exc


def all_labels() -> List[str]:
    """Return the labels as a fresh list. Used by callers (like the
    FastAPI ``/health`` endpoint and the OpenAPI spec generator) that
    want a JSON-serializable copy without exposing the underlying
    immutable tuple."""
    return list(LABELS)
