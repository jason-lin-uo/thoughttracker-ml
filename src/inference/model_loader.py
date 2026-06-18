"""
Lazy, thread-safe loader for the saved stance classifier.

Why "lazy"
----------
The FastAPI service imports its routes at startup, which transitively
imports this module. If we eagerly loaded the model here, every
``uvicorn --reload`` cycle would re-pay the ~5 second tokenizer +
DistilBERT load cost. By making the load lazy, the process starts in
<1 second and only the first ``/predict`` request (or the explicit
``load_model()`` call we make in the startup hook) pays the cost.

Why "thread-safe"
-----------------
FastAPI handles requests in a thread pool. Without a lock, two
concurrent first-requests could each trigger a full model load,
wasting CPU + memory. The lock ensures one-and-only-one load
happens regardless of how many threads pile in simultaneously.

Why a singleton (process-global cache)
--------------------------------------
DistilBERT is ~268 MB on disk; we don't want N copies in memory.
The module-level ``_loaded`` cache means every thread in the
process sees the same loaded instance.

How tests work around the singleton
-----------------------------------
The cache is module-global, which makes unit tests awkward. Tests
that exercise the loader monkey-patch ``model_loader._loaded`` to
``None`` (force reload), or pre-stuff it with a sentinel object
(skip the real load entirely). See ``tests/test_model_loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

from ..config import config
from ._device import select_device
from ..utils.logging import get_logger

logger = get_logger("model_loader")


@dataclass
class LoadedModel:
    """Bundle of "everything you need to do inference once":

    - ``tokenizer``: Hugging Face tokenizer instance.
    - ``model``: the ``AutoModelForSequenceClassification`` in eval mode.
    - ``model_version``: human-readable version string for response payloads.
    - ``base_model``: the base checkpoint we fine-tuned from (when known),
      surfaced via ``/health`` so a debug session can tell "is this an
      old DistilBERT or a fresh RoBERTa run?".

    Fields are typed as ``object`` so this dataclass doesn't drag a
    torch / transformers import into the typing surface — the runtime
    behavior is unchanged; static analyzers just get less help.
    """

    tokenizer: object
    model: object
    model_version: str
    base_model: Optional[str]
    #: Token truncation length the model was TRAINED with (from model_card.json's
    #: ``maxLength``). Inference must truncate to the same value — using a longer
    #: window (the config default) feeds the model positions it never saw in
    #: fine-tuning, degrading long-excerpt predictions. Defaults to the config
    #: value when no card is present (fresh/un-carded model).
    max_length: int = config.max_length


#: Process-global cache. ``None`` = nothing loaded yet (or factory-reset).
_lock: Lock = Lock()
_loaded: Optional[LoadedModel] = None
#: Last error string from a failed load. Exposed via ``get_load_error()``
#: so the FastAPI ``/health`` endpoint can surface it without retrying
#: a guaranteed-to-fail load on every request.
_load_error: Optional[str] = None


def is_model_available() -> bool:
    """Cheap "is a saved model on disk?" probe.

    Returns True iff ``<model_dir>/config.json`` exists — that's the
    file Hugging Face writes on a successful ``save_pretrained()``,
    so its presence is a reliable proxy for "the directory is a
    complete model artifact" without paying the cost of actually
    loading anything.
    """
    return (config.model_dir / "config.json").exists()


def get_load_error() -> Optional[str]:
    """Return the last load-error message, or ``None`` if the most
    recent load (or initialization) succeeded. Used by ``/health`` so
    operators see "model not loaded because <reason>" instead of just
    ``modelLoaded: false``."""
    return _load_error


def load_model(force: bool = False) -> LoadedModel:
    """Load the saved tokenizer + model, caching the result.

    Behavior:
      - First call (or with ``force=True``): reads from disk, returns
        the new ``LoadedModel``. Pays the ~5 second load cost.
      - Subsequent calls: returns the cached instance instantly.

    Parameters
    ----------
    force
        Pass ``True`` to bypass the cache and reload from disk.
        Useful for tests that overwrite the saved model mid-run,
        or for the eventual /reload admin endpoint.

    Raises
    ------
    FileNotFoundError
        If no model is on disk (``is_model_available() == False``).
        The message tells the operator exactly which path was checked
        and how to populate it (``python -m src.training.train``).
    RuntimeError
        If ``transformers`` isn't installed. Distinct from the
        ``FileNotFoundError`` so deployment scripts can tell "I
        haven't trained yet" from "my requirements.txt is broken".
    """
    global _loaded, _load_error

    with _lock:
        if _loaded is not None and not force:
            return _loaded

        if not is_model_available():
            _load_error = (
                f"No trained model found at {config.model_dir}. "
                f"Run `python -m src.training.train` first."
            )
            raise FileNotFoundError(_load_error)

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            _load_error = (
                "transformers is not installed. Run `pip install -r requirements.txt`."
            )
            raise RuntimeError(_load_error) from exc

        logger.info("Loading tokenizer + model from %s", config.model_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(config.model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(config.model_dir)
            )
            # cuda > mps (Apple Silicon) > cpu — see select_device for the
            # defensive capability probing and the rationale behind the order.
            device = select_device(torch)
            model.to(device)
            logger.info("Loaded stance model on %s", device)
            # ``eval()`` switches off dropout / batchnorm running-stats
            # updates. Critical for reproducible inference; trivially
            # cheap to call once and remember.
            model.eval()
        except Exception as exc:
            # A corrupt / incompatible saved model (bad config.json, version
            # skew, OOM) raises here. Record the reason so /health surfaces it
            # instead of a stale/None error ("false-healthy modelLoaded:false"),
            # then re-raise so the caller still fails loudly.
            _load_error = f"Failed to load model from {config.model_dir}: {exc}"
            raise

        # Best-effort read of the base-model name we trained from. The
        # model_card.json is written by train.py; if it's missing or
        # malformed we silently fall back to the config default so a
        # corrupt sidecar file doesn't take the whole load down.
        base_model_name = None
        card_max_length = None
        model_card_path = config.model_dir / "model_card.json"
        if model_card_path.exists():
            import json

            try:
                card = json.loads(model_card_path.read_text())
                base_model_name = card.get("baseModel")
                card_max_length = card.get("maxLength")
            except Exception:
                # Sidecar exists but is malformed. Keep going (the loader
                # falls back to config defaults below), but log so the
                # operator can investigate — silent fallback hides
                # genuine corruption.
                logger.warning(
                    "Could not parse model_card.json at %s; falling back to config defaults",
                    model_card_path,
                    exc_info=True,
                )
                base_model_name = None
                card_max_length = None

        _loaded = LoadedModel(
            tokenizer=tokenizer,
            model=model,
            model_version=config.model_version,
            base_model=base_model_name or config.base_model,
            max_length=int(card_max_length) if card_max_length else config.max_length,
        )
        _load_error = None
        return _loaded
