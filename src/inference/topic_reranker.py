"""Inference helpers for the TF-IDF topic candidate generator.

Thread-safety + cache
---------------------
``load_topic_reranker_model`` caches the loaded TF-IDF + SGD bundle in
a process-global slot guarded by a ``threading.Lock`` (mirroring
``model_loader.py``). FastAPI's thread pool means concurrent first
requests could otherwise each unpickle the (large) vectorizer + model,
wasting CPU + RAM. A ``.cache_clear()`` attribute is preserved so code
written against the old ``functools.lru_cache`` API keeps working.

Path/mtime cache invalidation
-----------------------------
The cache is keyed on the artifact's identity — its resolved path plus
the file's modification time — not just "have we loaded anything yet".
The old ``functools.lru_cache`` had no arguments, so once a bundle was
loaded it was pinned for the life of the process: retraining the
reranker, repointing ``config.topic_reranker_model_dir`` at a different
model, or hot-swapping the ``.pkl`` on disk would all be silently
ignored and the stale bundle served forever. We now stash the
``(path, mtime)`` we loaded from alongside the bundle and reload
automatically when either changes, so an operator who drops in a new
model gets it on the next request without a process restart.

Why joblib instead of pickle
----------------------------
The bundle is loaded with ``joblib.load`` rather than ``pickle.load``:
joblib is scikit-learn's recommended serializer (efficient for the
large vocabulary arrays a TF-IDF vectorizer carries) and narrows the
deserialization attack surface — a bare ``pickle.load`` on a model dir
that ever becomes attacker-writable is a remote-code-execution vector.
joblib reads our existing pickle artifacts transparently.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np

from ..config import config

MODEL_FILE = "topic_reranker_model.pkl"

#: Process-global cache + lock for the reranker bundle.
_lock = threading.Lock()
_cache = None
#: Identity of the artifact behind ``_cache``: ``(resolved_path, mtime_ns)``.
#: Compared on every load so the cache invalidates when the model file is
#: swapped, retrained, or ``config.topic_reranker_model_dir`` is repointed —
#: see the module docstring's "Path/mtime cache invalidation" note.
_cache_key: Optional[Tuple[str, int]] = None
#: Last error string from a failed load, mirroring ``model_loader``. The
#: FastAPI ``/health`` endpoint reads this via :func:`get_load_error` so a
#: broken reranker surfaces as ``status="degraded"`` instead of silently
#: passing as healthy (the warmup ``except`` only logged before).
_load_error: Optional[str] = None


def _current_cache_key() -> Optional[Tuple[str, int]]:
    """Return the ``(resolved_path, mtime_ns)`` of the on-disk artifact.

    ``None`` when the file is absent (so a missing model never matches a
    previously-cached key and we fall through to the real load, which
    raises the operator-facing ``FileNotFoundError``). ``mtime_ns`` gives
    nanosecond resolution so a fast retrain-then-reload in the same second
    still invalidates the cache.
    """
    model_path = config.topic_reranker_model_dir / MODEL_FILE
    try:
        stat = model_path.stat()
    except OSError:
        return None
    return (str(Path(model_path).resolve()), stat.st_mtime_ns)


def get_load_error() -> Optional[str]:
    """Return the last reranker load error, or ``None`` on success.

    Surfaced by ``/health`` so an operator sees *why* the reranker failed
    to load rather than just inferring it from disk state. Cleared back to
    ``None`` on the next successful load.
    """
    return _load_error


def is_topic_reranker_model_available() -> bool:
    """True if the reranker ``.pkl`` exists under the configured dir."""
    return (config.topic_reranker_model_dir / MODEL_FILE).exists()


def _load_topic_reranker_model_uncached():
    """Load the reranker bundle from disk (no caching, no lock).

    Raises ``FileNotFoundError`` when the artifact is absent; otherwise
    returns the joblib-deserialized dict (``vectorizer`` / ``classifier``
    / ``multiLabelBinarizer`` / optional ``modelVersion``).
    """
    model_path = config.topic_reranker_model_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"No topic reranker model found at {config.topic_reranker_model_dir}"
        )
    # joblib.load reads our existing pickle artifacts transparently while
    # avoiding a bare pickle.load on a potentially-writable model dir.
    return joblib.load(model_path)


def _clear_topic_reranker_cache() -> None:
    """Drop the cached reranker bundle (compat shim for ``cache_clear``)."""
    global _cache, _cache_key, _load_error
    with _lock:
        _cache = None
        _cache_key = None
        _load_error = None


def load_topic_reranker_model():
    """Return the cached reranker bundle, loading it once under a lock.

    Double-checked locking: an unlocked fast read on the hot path; the
    lock + second check guard against duplicate loads when concurrent
    threads race to a cache miss.

    The fast-path read is gated on the cache key still matching the
    artifact's current ``(path, mtime)`` — if the model file was swapped,
    retrained, or the configured dir was repointed, the key no longer
    matches and we fall through to a fresh load under the lock instead of
    serving the stale bundle.
    """
    global _cache, _cache_key, _load_error
    key = _current_cache_key()
    if _cache is not None and key is not None and key == _cache_key:
        return _cache
    with _lock:
        # Re-read the key inside the lock: another thread may have just
        # finished loading the current artifact while we waited.
        key = _current_cache_key()
        if _cache is None or key is None or key != _cache_key:
            try:
                _cache = _load_topic_reranker_model_uncached()
                # Re-stat after the (possibly slow) load so the stored key
                # reflects the bytes we actually deserialized.
                _cache_key = _current_cache_key()
            except Exception as exc:
                # Record the failure so ``/health`` can report it (the
                # warmup caller swallows the exception after logging). Then
                # re-raise so per-request callers still get the real error.
                _cache = None
                _cache_key = None
                _load_error = str(exc)
                raise
            _load_error = None
        return _cache


#: Backwards-compatible alias for the old ``functools.lru_cache`` method.
load_topic_reranker_model.cache_clear = _clear_topic_reranker_cache  # type: ignore[attr-defined]


def predict_topic_candidates(
    text: str,
    *,
    limit: int = 12,
    min_score: float = 0.2,
) -> dict:
    """Return up to ``limit`` controlled-taxonomy topic candidates for ``text``.

    Vectorizes ``text``, runs the multi-label classifier's
    ``predict_proba``, and returns candidates sorted by descending
    probability, dropping any below ``min_score``. This is a high-recall
    generator — the backend still applies relevance, confidence, and
    display policy on top.

    Raises ``ValueError`` for empty ``text``, ``limit < 1``, or
    ``min_score`` outside ``[0, 1]``.
    """
    if not text or not text.strip():
        raise ValueError("`text` is required")
    if limit < 1:
        raise ValueError("`limit` must be at least 1")
    if min_score < 0 or min_score > 1:
        raise ValueError("`min_score` must be between 0 and 1")

    loaded = load_topic_reranker_model()
    vectorizer = loaded["vectorizer"]
    classifier = loaded["classifier"]
    mlb = loaded["multiLabelBinarizer"]
    model_version = str(loaded.get("modelVersion") or config.topic_reranker_model_version)

    matrix = vectorizer.transform([text])
    probabilities = np.asarray(classifier.predict_proba(matrix)[0], dtype=float)
    labels = list(mlb.classes_)

    # Rank by descending probability with a DETERMINISTIC tie-break.
    # ``np.argsort`` defaults to an unstable quicksort, so two topics with
    # identical probabilities could come back in either order from run to
    # run (and ``[::-1]`` on a stable sort would flip ties into reverse
    # label order, which is just as arbitrary). We instead sort the negated
    # probabilities with ``kind="stable"``: highest probability first, and
    # on an exact tie the lower original index (i.e. the earlier label in
    # ``mlb.classes_``) wins. That makes the candidate list — and therefore
    # any downstream snapshot/equality test — fully reproducible.
    ranked_indices = np.argsort(-probabilities, kind="stable")
    topics = []
    for index in ranked_indices:
        score = float(probabilities[index])
        if score < min_score:
            continue
        topics.append(
            {
                "topicSlug": labels[index],
                "confidence": round(score, 4),
            }
        )
        if len(topics) >= limit:
            break

    return {
        "topics": topics,
        "modelVersion": model_version,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build the CLI parser (``--text`` required; ``--limit`` / ``--min-score`` optional)."""
    parser = argparse.ArgumentParser(description="Predict topic candidates.")
    parser.add_argument("--text", required=True, help="Transcript excerpt.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-score", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Prints the candidate JSON to stdout.

    Exit codes: 0 success, 1 ``ValueError`` (bad input), 2
    ``FileNotFoundError`` (no reranker model on disk).
    """
    args = _parse_args(argv)
    try:
        result = predict_topic_candidates(
            args.text,
            limit=args.limit,
            min_score=args.min_score,
        )
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
