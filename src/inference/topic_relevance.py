"""Inference helpers for the topic relevance gate.

Thread-safety + cache
---------------------
``load_topic_relevance_model`` caches the loaded model in a
process-global slot guarded by a ``threading.Lock`` — the same pattern
as ``model_loader.py``. FastAPI serves requests from a thread pool, so
without the lock two concurrent first-requests could each kick off a
full transformer load (wasting CPU + RAM and racing on the shared
slot). The function keeps a ``.cache_clear()`` attribute so callers /
tests that previously relied on the ``functools.lru_cache`` API keep
working unchanged.

Why joblib instead of pickle
----------------------------
The sklearn fallback artifact is loaded with ``joblib.load`` rather
than ``pickle.load``. joblib is the scikit-learn-recommended
serializer (it handles large numpy arrays far more efficiently) and,
just as importantly, narrows the deserialization surface: a raw
``pickle.load`` on a model directory that ever becomes
attacker-writable is a classic remote-code-execution vector. joblib
reads our own legacy pickle artifacts transparently, so this is a
drop-in hardening with no retraining required.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import Dict, List, Optional

import joblib

from ..config import config
from ..data.load_dataset import build_input_text
from ..utils.logging import get_logger
from ._device import select_device

logger = get_logger("topic_relevance")

LABELS = ["irrelevant", "relevant"]
MODEL_FILE = "topic_relevance_model.pkl"
TRANSFORMER_CONFIG_FILE = "config.json"

#: Process-global cache + lock for the relevance model. ``None`` = nothing
#: loaded yet. The lock serializes concurrent first-loads.
_lock = threading.Lock()
_cache: Optional[dict] = None
#: Last error string from a failed load, mirroring ``model_loader``. The
#: FastAPI ``/health`` endpoint reads this via :func:`get_load_error` so a
#: broken relevance model surfaces as ``status="degraded"`` instead of
#: silently passing as healthy (the warmup ``except`` only logged before).
_load_error: Optional[str] = None


def get_load_error() -> Optional[str]:
    """Return the last relevance-model load error, or ``None`` on success.

    Surfaced by ``/health`` so an operator sees *why* the relevance model
    failed to load rather than just inferring it from disk state. Cleared
    back to ``None`` on the next successful load.
    """
    return _load_error


def is_topic_relevance_model_available() -> bool:
    """Cheap "is a relevance model on disk?" probe.

    True if either a transformer ``config.json`` OR the sklearn-fallback
    ``.pkl`` exists under the configured relevance model dir — those are
    the two artifact shapes :func:`load_topic_relevance_model` knows how
    to load.
    """
    return (
        (config.topic_relevance_model_dir / TRANSFORMER_CONFIG_FILE).exists()
        or (config.topic_relevance_model_dir / MODEL_FILE).exists()
    )


def _load_topic_relevance_model_uncached() -> dict:
    """Load the relevance model from disk (no caching, no lock).

    Prefers a transformer artifact (``config.json`` present → load via
    Hugging Face, move to GPU when available, eval mode) and otherwise
    falls back to the sklearn ``.pkl`` (loaded with joblib). Raises
    ``FileNotFoundError`` when neither artifact is present.

    Probe / load-branch agreement
    -----------------------------
    :func:`is_topic_relevance_model_available` counts EITHER artifact as
    "available", so the load branch must accept either too — otherwise the
    probe and the load could disagree. The specific trap: a stub or
    truncated ``config.json`` (e.g. a half-written download) sitting next
    to a perfectly loadable ``.pkl``. Naively preferring the transformer
    whenever ``config.json`` merely *exists* would explode in
    ``from_pretrained`` even though a usable sklearn fallback is right
    there. So when the transformer config is present but fails to load, we
    fall back to the ``.pkl`` if one exists (re-raising the original
    transformer error only when there's no fallback to offer).
    """
    transformer_config = config.topic_relevance_model_dir / TRANSFORMER_CONFIG_FILE
    model_path = config.topic_relevance_model_dir / MODEL_FILE
    if transformer_config.exists():
        try:
            return _load_transformer_relevance_model()
        except Exception as exc:
            # A present-but-broken transformer config shouldn't mask a
            # valid sklearn fallback. If no .pkl exists, the transformer
            # error IS the real failure, so re-raise it unchanged.
            if not model_path.exists():
                raise
            logger.warning(
                "Topic-relevance transformer artifact at %s failed to load (%s); "
                "falling back to the sklearn .pkl.",
                config.topic_relevance_model_dir,
                exc,
            )

    if not model_path.exists():
        raise FileNotFoundError(
            f"No topic relevance model found at {config.topic_relevance_model_dir}"
        )
    # joblib.load reads our existing pickle artifacts transparently while
    # avoiding a bare pickle.load on a potentially-writable model dir.
    return {"kind": "sklearn", "model": joblib.load(model_path)}


def _load_transformer_relevance_model() -> dict:
    """Load the Hugging Face transformer relevance artifact (no fallback).

    Split out from :func:`_load_topic_relevance_model_uncached` so the
    caller can wrap *just* the transformer load in a try/except and fall
    back to the sklearn ``.pkl`` on failure without also catching a
    fallback-path error.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.topic_relevance_model_dir,
        fix_mistral_regex=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.topic_relevance_model_dir
    )
    # cuda > mps (Apple Silicon) > cpu — shared, defensively-probed
    # selection (tolerates a stubbed torch without `backends`).
    device = select_device(torch)
    model.to(device)
    model.eval()
    return {"kind": "transformer", "tokenizer": tokenizer, "model": model}


def _clear_topic_relevance_cache() -> None:
    """Drop the cached relevance model (compat shim for ``cache_clear``).

    Exposed as ``load_topic_relevance_model.cache_clear`` so the test
    suite's autouse cache-reset fixture (written against the old
    ``functools.lru_cache``) keeps working after the switch to a manual
    lock-guarded cache.
    """
    global _cache, _load_error
    with _lock:
        _cache = None
        _load_error = None


def load_topic_relevance_model() -> dict:
    """Return the cached relevance model, loading it once under a lock.

    Double-checked locking: a fast unlocked read returns the cached
    instance on the hot path; only the first (cache-miss) caller takes
    the lock and loads, and a second check inside the lock prevents a
    duplicate load if several threads raced to the miss.
    """
    global _cache, _load_error
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is None:
            try:
                _cache = _load_topic_relevance_model_uncached()
            except Exception as exc:
                # Record the failure so ``/health`` can report it (the
                # warmup caller swallows the exception after logging). Then
                # re-raise so per-request callers still get the real error.
                _load_error = str(exc)
                raise
            _load_error = None
        return _cache


#: Backwards-compatible alias for the old ``functools.lru_cache`` method
#: name, so existing callers/tests can still invalidate the cache.
load_topic_relevance_model.cache_clear = _clear_topic_relevance_cache  # type: ignore[attr-defined]


def predict_topic_relevance(topic: str, text: str) -> dict:
    """Score whether ``text`` is actually about ``topic``.

    Returns the same envelope shape as the stance predictor —
    ``predictedLabel`` (``relevant`` / ``irrelevant``), ``confidence``,
    a full ``labelScores`` map, and ``modelVersion`` — so the backend's
    false-positive gate can parse it uniformly. Dispatches to the
    transformer or sklearn path depending on the loaded artifact kind.

    Raises ``ValueError`` if ``topic`` or ``text`` is empty/whitespace.
    """
    if not topic or not topic.strip():
        raise ValueError("`topic` is required")
    if not text or not text.strip():
        raise ValueError("`text` is required")

    loaded = load_topic_relevance_model()
    encoded = build_input_text(topic, text)

    if loaded["kind"] == "transformer":
        probabilities = _predict_transformer(loaded, encoded)
        classes = LABELS
    else:
        model = loaded["model"]
        probabilities = model.predict_proba([encoded])[0]
        classes = list(model.classes_)

    label_scores: Dict[str, float] = {
        label: round(float(probabilities[classes.index(label)]), 4)
        if label in classes
        else 0.0
        for label in LABELS
    }
    predicted = max(LABELS, key=lambda label: label_scores[label])
    return {
        "topic": topic,
        "text": text,
        "predictedLabel": predicted,
        "confidence": label_scores[predicted],
        "labelScores": label_scores,
        "modelVersion": config.topic_relevance_model_version,
    }


def _predict_transformer(loaded, encoded: str) -> List[float]:
    """Run a transformer forward pass and return ``[p_irrelevant, p_relevant]``.

    Tokenizes ``encoded`` (already formatted by ``build_input_text``),
    moves inputs to the model's device, runs a no-grad forward pass, and
    softmaxes the logits. ``torch`` is imported locally to keep module
    import light.
    """
    import torch

    tokenizer = loaded["tokenizer"]
    model = loaded["model"]
    with torch.no_grad():
        inputs = tokenizer(
            encoded,
            return_tensors="pt",
            truncation=True,
            max_length=config.topic_relevance_max_length,
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        logits = model(**inputs).logits[0]
        probs = torch.nn.functional.softmax(logits, dim=-1).tolist()
    return [float(probs[0]), float(probs[1])]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build the CLI parser. Both ``--topic`` and ``--text`` are required."""
    parser = argparse.ArgumentParser(description="Predict topic relevance.")
    parser.add_argument("--topic", required=True, help="Topic to test.")
    parser.add_argument("--text", required=True, help="Transcript excerpt.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Prints the relevance JSON to stdout.

    Exit codes mirror the stance CLI: 0 success, 1 ``ValueError`` (bad
    input), 2 ``FileNotFoundError`` (no relevance model on disk).
    """
    args = _parse_args(argv)
    try:
        result = predict_topic_relevance(args.topic, args.text)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
