"""Centralized configuration loaded from environment variables.

All training / inference / API code reads from this module so behavior is
controllable via .env without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # pragma: no cover  (only executes when python-dotenv is installed)
except (
    ImportError
):  # pragma: no cover  (only fires when python-dotenv is absent — uninteresting branch)
    pass

from .utils.paths import DEFAULT_STANCE_DATA_PATH, MODELS_DIR, PROJECT_ROOT


class ConfigError(ValueError):
    """Raised when an environment variable holds an unparseable value.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers
    still catch it, while the distinct type lets boot scripts surface a
    "your .env is misconfigured" message instead of a generic crash.
    """


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Returns ``default`` when the variable is unset; otherwise treats
    ``1/true/yes/on`` (case-insensitive, whitespace-trimmed) as ``True``
    and everything else as ``False``. Boolean parsing is intentionally
    permissive (no exception on a bad value) because the failure mode is
    benign — a typo just falls back to the documented default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    """Resolve a filesystem-path environment variable to an absolute path.

    Unset / empty → ``default``. A relative value is resolved against
    ``PROJECT_ROOT`` so behavior is independent of the caller's CWD; an
    absolute value is used verbatim.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable, failing loudly on garbage.

    Unset / empty → ``default``. A non-integer value raises
    ``ConfigError`` naming the variable, the bad value, and the expected
    type — far friendlier than the bare ``ValueError: invalid literal for
    int()`` that ``int(os.environ[...])`` would otherwise throw at import
    time, where the traceback doesn't even mention which env var was at
    fault.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {name}={raw!r} is not a valid integer. "
            f"Set it to a whole number (e.g. {default}) or unset it to use the default."
        ) from exc


def _env_float(name: str, default: float) -> float:
    """Parse a float environment variable, failing loudly on garbage.

    Mirrors :func:`_env_int` but for floating-point settings (learning
    rate, split fractions, etc.). Unset / empty → ``default``; an
    unparseable value raises a ``ConfigError`` that names the offending
    variable instead of an opaque ``invalid literal`` traceback.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {name}={raw!r} is not a valid number. "
            f"Set it to a decimal value (e.g. {default}) or unset it to use the default."
        ) from exc


@dataclass(frozen=True)
class Config:
    """Immutable, environment-driven configuration for the ML service.

    Every field reads from an environment variable (with a sensible
    default) via the ``_env_*`` helpers above, so behavior is tunable
    through ``.env`` without code changes. Frozen so a stray write
    can't mutate shared config mid-run; tests override it with
    ``dataclasses.replace``. Numeric fields fail loudly at import time
    on an unparseable value (see :func:`_env_int` / :func:`_env_float`).
    """

    # Model artifacts
    model_dir: Path = _env_path("MODEL_DIR", MODELS_DIR / "stance-classifier")
    model_version: str = os.environ.get("MODEL_VERSION", "stance-classifier-v1")
    topic_relevance_model_dir: Path = _env_path(
        "TOPIC_RELEVANCE_MODEL_DIR",
        MODELS_DIR / "topic-relevance-classifier-supervalidation-hardneg2x-l512",
    )
    topic_relevance_model_version: str = os.environ.get(
        # Default matches the version recorded in the on-disk model_card.json
        # for the relevance classifier (and .env.example), so an unset env
        # var reports the same version the artifact was trained/published as.
        "TOPIC_RELEVANCE_MODEL_VERSION",
        "topic-relevance-supervalidation-hardneg2x-l512",
    )
    topic_relevance_max_length: int = _env_int("TOPIC_RELEVANCE_MAX_LENGTH", 512)
    topic_reranker_model_dir: Path = _env_path(
        "TOPIC_RERANKER_MODEL_DIR", MODELS_DIR / "topic-reranker-tfidf-sgd-supervalidation"
    )
    topic_reranker_model_version: str = os.environ.get(
        # Fallback version when the loaded reranker bundle has no embedded
        # ``modelVersion`` (the bundle's own value wins at runtime). Aligned
        # to the version recorded in the reranker's model_card.json / .pkl
        # bundle so the documented default never disagrees with the artifact.
        "TOPIC_RERANKER_MODEL_VERSION",
        "topic-reranker-tfidf-sgd-v1",
    )

    # Base pretrained checkpoint to fine-tune
    base_model: str = os.environ.get("BASE_MODEL", "distilbert-base-uncased")

    # Training hyperparameters (intentionally light for laptops)
    max_length: int = _env_int("MAX_LENGTH", 256)
    train_batch_size: int = _env_int("TRAIN_BATCH_SIZE", 8)
    eval_batch_size: int = _env_int("EVAL_BATCH_SIZE", 16)
    num_train_epochs: int = _env_int("NUM_TRAIN_EPOCHS", 3)
    learning_rate: float = _env_float("LEARNING_RATE", 5e-5)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.01)
    seed: int = _env_int("SEED", 42)

    # Dataset
    dataset_path: Path = _env_path("DATASET_PATH", DEFAULT_STANCE_DATA_PATH)
    test_size: float = _env_float("TEST_SIZE", 0.2)
    val_size: float = _env_float("VAL_SIZE", 0.1)

    # API
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = _env_int("API_PORT", 8000)

    # Test-only mock mode
    enable_mock_inference: bool = _env_bool("ENABLE_MOCK_INFERENCE", False)


config = Config()
