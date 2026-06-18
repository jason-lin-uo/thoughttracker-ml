"""
Sentence-embedding inference: mean-pooled ``distilbert-base-uncased``.

Supports ThoughtTracker's owner/offline embedding refresh path when the backend
runs with ``EMBEDDING_PROVIDER=ml``. The public app no longer exposes semantic
search, but the endpoint remains useful when rebuilding stored vectors. The
base checkpoint is already in the local Hugging Face cache (it's what the
stance classifier was fine-tuned from), so this needs no download and no new
dependency - just torch + transformers, which are already installed for the
classifier.

Design mirrors ``model_loader``/``predict``:
  - **Lazy, thread-safe singleton** - the ~270 MB model loads once, on first
    use, not at import. Unlike the stance/topic models it is NOT eagerly
    warmed at API startup, so the first ``/embed`` request pays the load cost.
  - **Mock fallback** - a deterministic, unit-normalized hash vector of the
    SAME dimension, returned when ``ENABLE_MOCK_INFERENCE`` is set or torch/the
    model can't load. Keeps the endpoint + tests runnable with no model, and
    keeps mock vectors dimension-compatible with real ones.

Embeddings are mean-pooled over the (attention-masked) last hidden state and
L2-normalized, so cosine similarity == dot product downstream.
"""

from __future__ import annotations

import hashlib
import math
from threading import Lock
from typing import List, Optional, Tuple

from ..config import config
from ..utils.logging import get_logger

logger = get_logger("embed")

#: DistilBERT hidden size - the embedding dimensionality.
EMBED_DIM = 768
#: Truncate inputs to the first N tokens. Search cares about the gist, and
#: capping the window keeps re-embedding the full corpus tractable on CPU.
EMBED_MAX_TOKENS = 256

_lock: Lock = Lock()
#: (tokenizer, model, torch-module) once loaded; ``None`` until then / on failure.
_loaded: Optional[Tuple[object, object, object]] = None
_load_error: Optional[str] = None
_load_attempted = False


def _mock_vector(text: str) -> List[float]:
    """Deterministic, unit-normalized hash embedding (a bag-of-words sketch).

    Same dimension as the real model so the two are interchangeable in storage,
    and stable so a given text always maps to the same vector. Used only when
    the real model is unavailable (mock mode / no torch / load failure).
    """
    vec = [0.0] * EMBED_DIM
    for token in text.lower().split():
        bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % EMBED_DIM
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _build_encoder() -> Tuple[object, object, object]:  # pragma: no cover - loads the real model; exercised in real runs / the smoke test, not hermetic CI
    """Load the base DistilBERT tokenizer + encoder + torch. Isolated so the
    surrounding load/error bookkeeping stays unit-testable without torch."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = config.base_model
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return (tokenizer, model, torch)


def _load() -> None:
    """Load the encoder into the process cache. Records the error (rather than
    raising) on failure so callers degrade to the mock instead of 500-ing."""
    global _loaded, _load_error, _load_attempted
    _load_attempted = True
    try:
        _loaded = _build_encoder()
        _load_error = None
        logger.info("embed model loaded", extra={"model": config.base_model})
    except Exception as exc:  # noqa: BLE001 - any load failure degrades to mock
        _load_error = str(exc)
        _loaded = None
        logger.warning("embed model load failed; using mock", extra={"error": str(exc)})


def load_embed_model() -> None:
    """Load the embedding model into the process cache (idempotent, thread-safe).

    NOTE: unlike the stance/topic models, this is NOT called from the API
    startup warmup - ``_warmup_model()`` in ``src/api/main.py`` does not touch
    it - so the FIRST ``/embed`` request after boot pays a one-time lazy-load
    cost. It is invoked on demand by ``embed_texts`` on that first call;
    subsequent calls reuse the cached encoder.
    """
    with _lock:
        if not _load_attempted:
            _load()


def is_embed_model_available() -> bool:
    """True when the real encoder is loaded (vs. the mock fallback)."""
    return _loaded is not None


def get_embed_load_error() -> Optional[str]:
    """The last load error string, or ``None`` if the load succeeded / not tried."""
    return _load_error


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts into L2-normalized 768-d vectors.

    Returns mock vectors when mock mode is on or the model isn't available, so
    the result shape is identical regardless of path.
    """
    if config.enable_mock_inference:
        return [_mock_vector(t) for t in texts]
    load_embed_model()
    if _loaded is None:
        return [_mock_vector(t) for t in texts]
    return _encode_with_model(texts)


def _encode_with_model(texts: List[str]) -> List[List[float]]:  # pragma: no cover - real torch inference; exercised in real runs / the smoke test, not hermetic CI
    """Masked-mean-pool the encoder's last hidden state, L2-normalized."""
    tokenizer, model, torch = _loaded  # type: ignore[misc]
    with torch.no_grad():
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=EMBED_MAX_TOKENS,
            return_tensors="pt",
        )
        output = model(**enc)
        last_hidden = output.last_hidden_state  # (batch, tokens, hidden)
        mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized.tolist()
