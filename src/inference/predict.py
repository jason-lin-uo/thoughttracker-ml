"""
Single-example stance inference: CLI + reusable ``predict()`` helper.

This module is the bridge between "a trained DistilBERT sitting in
``models/stance-classifier/``" and the rest of the system. Two ways
to use it:

  1. **Library**: ``from src.inference.predict import predict``,
     then ``predict(topic="...", text="...")``. Used by the FastAPI
     service at ``src/api/main.py`` to power the ``/predict`` endpoint.

  2. **CLI**: ``python -m src.inference.predict --topic ... --text ...``
     prints a JSON payload to stdout. Useful for debugging, smoke
     tests, and integration sanity checks from shell scripts.

Two execution paths, switched at runtime
----------------------------------------
The function dispatches based on (in order):

  - **Real path** (``_predict_real``): used when a saved model exists
    at ``config.model_dir/config.json``. Loads the tokenizer + model
    via ``model_loader.load_model()``, runs a forward pass with
    softmax, returns the labeled probability distribution.

  - **Mock path** (``_predict_mock``): used when no model is on disk
    AND ``ENABLE_MOCK_INFERENCE=true`` (the default for the demo).
    Implements a deterministic keyword-cue heuristic — same text
    always produces the same label, so the FastAPI service can serve
    realistic-looking predictions without any model downloaded yet.

If no model AND mock is OFF, we raise ``FileNotFoundError`` so a
production deployment can't accidentally serve garbage when training
hasn't run yet.

Example:
    python -m src.inference.predict \\
        --topic "foreign policy" \\
        --text "I disagree with this approach and I worry about its impact."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Dict, List, Optional

from ..config import config
from ..data.label_schema import LABELS
from ..data.load_dataset import build_input_text
from ..utils.logging import get_logger
from .model_loader import is_model_available, load_model

logger = get_logger("predict")


def predict(topic: str, text: str) -> dict:
    """Compute a stance prediction for one ``(topic, text)`` pair.

    Routes to the real DistilBERT inference path if a trained model is
    on disk, OR the deterministic mock path when ``ENABLE_MOCK_INFERENCE``
    is true. Raises ``FileNotFoundError`` when neither is available, so
    a misconfigured production deployment never silently serves garbage.

    Parameters
    ----------
    topic
        The topic / claim to classify the chunk against. Non-empty
        required — empty string raises ``ValueError``.
    text
        The chunk of transcript to classify. Non-empty required.

    Returns
    -------
    dict with keys:
      - ``topic``, ``text``: echoed back for the caller's bookkeeping.
      - ``predictedLabel``: one of the canonical labels (see
        ``label_schema.LABELS``).
      - ``confidence``: probability of the predicted label, 0-1.
      - ``labelScores``: dict mapping each label to its probability
        (sums to ~1.0 across all labels).
      - ``modelVersion``: the version of the model that produced the
        prediction. Mock predictions have a ``-mock`` suffix so callers
        can distinguish them.

    Raises
    ------
    ValueError
        ``topic`` or ``text`` is empty / whitespace-only.
    FileNotFoundError
        No trained model on disk AND mock inference is disabled.
    """
    if not text or not text.strip():
        raise ValueError("`text` is required")
    if not topic or not topic.strip():
        raise ValueError("`topic` is required")

    if is_model_available():
        return _predict_real(topic, text)

    if config.enable_mock_inference:
        return _predict_mock(topic, text)

    raise FileNotFoundError(
        f"No trained model found at {config.model_dir}. "
        f"Run `python -m src.training.train` first, or set ENABLE_MOCK_INFERENCE=true "
        f"for demo fallback."
    )


# ---------------------------------------------------------------------------
# Real prediction path
# ---------------------------------------------------------------------------


def _predict_real(topic: str, text: str) -> dict:
    """Run an inference forward pass with the saved DistilBERT model.

    Steps:
      1. ``load_model()`` returns the cached tokenizer + model
         (loaded lazily on the first call so module import is cheap).
      2. ``build_input_text`` formats the (topic, text) pair using the
         same delimiter format the model saw during training.
      3. Tokenize with ``return_tensors="pt"`` so we get torch tensors,
         truncated to the model's TRAINED ``max_length`` (from model_card.json,
         via ``loaded.max_length``) so inference matches training exactly.
      4. ``torch.no_grad()`` block — we're inference-only, so we skip
         autograd to save memory + speed.
      5. Forward pass returns logits; ``softmax(dim=-1)`` produces
         the per-label probability distribution.
      6. ``_build_prediction_response`` packages everything into the JSON-shaped dict.

    ``torch`` is imported locally so the module top-level stays light
    (the FastAPI ``/health`` endpoint doesn't need torch loaded just to
    answer "alive").
    """
    import torch

    loaded = load_model()
    tokenizer = loaded.tokenizer
    model = loaded.model

    encoded = build_input_text(topic, text)
    with torch.no_grad():
        inputs = tokenizer(
            encoded,
            return_tensors="pt",
            truncation=True,
            max_length=loaded.max_length,
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        logits = model(**inputs).logits[0]
        probs = torch.nn.functional.softmax(logits, dim=-1).tolist()

    return _build_prediction_response(topic, text, probs, loaded.model_version)


# ---------------------------------------------------------------------------
# Mock prediction path (deterministic; used only when ENABLE_MOCK_INFERENCE=true)
# ---------------------------------------------------------------------------


_SUPPORTIVE_CUES = (
    "support",
    "i'm in favor",
    "i am in favor",
    "embrace",
    "we should",
    "i believe",
    "i agree",
    "right call",
)
_OPPOSED_CUES = (
    "disagree",
    "against",
    "worry",
    "harmful",
    "shouldn't",
    "should not",
    "concerns me",
    "problem with",
)
_NEUTRAL_CUES = (
    "according to the data",
    "research shows",
    "on balance",
    "i'll describe",
    "lay out the facts",
)
_MIXED_CUES = ("on the other hand", "however", "at the same time", "both sides")
_UNCLEAR_CUES = (
    "not sure",
    "haven't decided",
    "could be read",
    "sarcastic",
    "hypothetical",
)


def _predict_mock(topic: str, text: str) -> dict:
    """Deterministic keyword-based stance predictor for demo / dev use.

    Algorithm:
      1. Lowercase the text.
      2. For each label, count occurrences of its cue list
         (``_SUPPORTIVE_CUES``, etc.) in the text. Weight ``mixed``
         and ``unclear`` cues at 0.8 to avoid them dominating when
         multiple cues happen to appear.
      3. If no cues match at all, fall back to ``unclear`` = 1.0 so
         the output is never all-zero (which would make softmax
         non-deterministic).
      4. Add a tiny deterministic sha256-derived perturbation per
         label so ties resolve consistently for the same input.
      5. Softmax the scores and package via ``_build_prediction_response`` with a
         ``-mock`` suffix on the model version.

    Two important properties:
      - **Same input always produces the same output.** Useful for
        snapshot tests + reproducible demos.
      - **Reasonable face validity.** "I support X" → supportive,
        "I disagree with X" → opposed, etc. Not state-of-the-art,
        but believable enough that a recruiter doesn't immediately
        see the mock is broken.
    """
    text_lower = text.lower()
    score: Dict[str, float] = {label: 0.0 for label in LABELS}

    for cue in _SUPPORTIVE_CUES:
        if cue in text_lower:
            score["supportive"] += 1.0
    for cue in _OPPOSED_CUES:
        if cue in text_lower:
            score["opposed"] += 1.0
    for cue in _NEUTRAL_CUES:
        if cue in text_lower:
            score["neutral"] += 1.0
    for cue in _MIXED_CUES:
        if cue in text_lower:
            score["mixed"] += 0.8
    for cue in _UNCLEAR_CUES:
        if cue in text_lower:
            score["unclear"] += 0.8

    if sum(score.values()) == 0:
        score["unclear"] = 1.0

    # Deterministic tiny perturbation so equal counts resolve consistently.
    # The hex string is 64 chars (sha256), but we tile it defensively in
    # case a future hash swap makes it shorter than 2*len(LABELS).
    hash_hex = hashlib.sha256(f"{topic}::{text}".encode()).hexdigest()
    needed = 2 * len(LABELS)
    if len(hash_hex) < needed:  # pragma: no cover - unreachable: sha256 is fixed-width 64 chars
        hash_hex = (hash_hex * ((needed // len(hash_hex)) + 1))[:needed]
    for i, label in enumerate(LABELS):
        score[label] += (int(hash_hex[i * 2 : i * 2 + 2], 16) / 255.0) * 0.05

    probs = _softmax([score[label] for label in LABELS])
    return _build_prediction_response(topic, text, probs, f"{config.model_version}-mock")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _softmax(values: List[float]) -> List[float]:
    """Numerically-stable softmax. Subtracting ``max(values)`` before
    exponentiating prevents overflow when one logit is much larger
    than the others. Returns probabilities summing to 1.0 (or, in
    the degenerate empty-input case, an empty list)."""
    import math

    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _build_prediction_response(topic: str, text: str, probs: List[float], model_version: str) -> dict:
    """Package a probability distribution into the JSON dict the
    FastAPI ``/predict`` endpoint hands back.

    Probabilities are rounded to 4 decimal places — full float64
    precision adds noise that's never visible to the caller and
    makes snapshot tests brittle.

    A hard assertion guards the ``len(probs) == len(LABELS)`` invariant.
    If a checkpoint's classification head drifts out of sync with the
    canonical label schema (wrong ``num_labels`` at train time, a stale
    model directory, etc.), ``zip`` would silently truncate to the
    shorter sequence and the response would mis-map labels to scores
    without any error. Failing loudly here turns a silent correctness
    bug into an obvious 500 + logged stack trace.
    """
    if len(probs) != len(LABELS):
        raise ValueError(
            f"Model emitted {len(probs)} probabilities but the label schema has "
            f"{len(LABELS)} labels {list(LABELS)}; refusing to mis-map scores. "
            f"This usually means the checkpoint's classification head is out of "
            f"sync with src.data.label_schema."
        )
    label_scores = {label: round(float(p), 4) for label, p in zip(LABELS, probs)}
    # Deterministic argmax tie-break: ``max`` over ``range`` with a key
    # returns the FIRST index that attains the maximum, so when two labels
    # share the top probability the earlier label in ``LABELS`` always wins
    # (rather than ``np.argmax``'s "first" being subject to dtype/version
    # quirks). Keeps the predicted label reproducible for snapshot tests.
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "topic": topic,
        "text": text,
        "predictedLabel": LABELS[best_idx],
        "confidence": round(float(probs[best_idx]), 4),
        "labelScores": label_scores,
        "modelVersion": model_version,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build the CLI parser. Both ``--topic`` and ``--text`` are
    required so we never silently classify an empty input."""
    parser = argparse.ArgumentParser(description="Predict transcript stance.")
    parser.add_argument("--topic", required=True, help="Topic the excerpt is about.")
    parser.add_argument("--text", required=True, help="Transcript excerpt.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns an exit code so callers (smoke tests,
    shell scripts) can branch on success/failure:

      - 0: success — JSON printed to stdout.
      - 1: ValueError — bad input. Error message to stderr.
      - 2: FileNotFoundError — no model on disk + mock disabled.
        Helpful "run train.py first" message to stderr.
    """
    args = _parse_args(argv)
    try:
        result = predict(args.topic, args.text)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())  # pragma: no cover
