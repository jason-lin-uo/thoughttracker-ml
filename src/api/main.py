"""
FastAPI service exposing ThoughtTracker's transcript-intelligence models
(stance, topic relevance, topic reranking) plus a sentence-embedding endpoint.

This is the runtime surface the main ThoughtTracker backend talks to: the
stance/topic clients call it when ``STANCE_ANALYSIS_PROVIDER=custom_ml`` (or
``hybrid``), and the embedding client (``embeddingClient.ts`` ``embedViaMl``)
calls ``/embed`` when ``EMBEDDING_PROVIDER=ml``. Five endpoints:

  - ``GET /health`` - liveness + "are the models loaded yet" probe. Returns
    ``{ status, modelLoaded, topicRelevanceModelLoaded,
    topicRerankerModelLoaded, modelVersion, mockInference, loadError }``.
    The backend's ``mlClassifierClient.healthCheck()`` calls this
    on startup + via the operations dashboard.

  - ``POST /predict`` - accepts ``{ topic, text }`` and returns
    ``{ predictedLabel, confidence, labelScores, modelVersion }``.
    Routes through ``src.inference.predict.predict()`` which
    dispatches to the real model OR the mock based on availability
    + env config.

  - ``POST /embed`` - accepts ``{ texts: [str, ...] }`` and returns
    ``{ vectors, dim, modelVersion, mockInference }`` (768-d, L2-normalized,
    mean-pooled ``distilbert-base-uncased``). Retained for owner/offline
    embedding refreshes; degrades to a deterministic mock vector of the same
    dimension (``mockInference: true``) when the encoder can't load.

  - ``POST /predict-topic-relevance`` - accepts ``{ topic, text }`` and
    returns ``{ predictedLabel, confidence, labelScores, modelVersion }``;
    the backend uses it as a false-positive gate on topic assignments.

  - ``POST /predict-topics`` - accepts ``{ text, limit, minScore }`` and
    returns ``{ topics: [{ topicSlug, confidence }], modelVersion }``;
    high-recall controlled-taxonomy topic candidates.

Design choices
--------------
- **No multi-pair batching for /predict.** The backend calls /predict per
  chunk-topic pair; batching could shave latency but adds complexity (response
  ordering, partial failures). At ~50 ms per call we have plenty of
  throughput for the demo's volume. (``/embed`` is the exception — it accepts a
  batch of texts in one call, where the per-call tokenizer/model overhead
  makes batching worthwhile.)
- **Errors are JSON, not HTML.** ``RequestValidationError`` is caught
  and rewrapped to ``{ error: "INVALID_INPUT", message: "<field>: <msg>; …" }``
  (a STRING, matching the ``message: str`` contract the backend client reads)
  so the backend can parse failures consistently with the rest of
  the API surface.
- **Eager startup warmup.** The stance + topic models are loaded in the
  FastAPI ``lifespan`` handler so the first /predict isn't a cold start.
  Without this, the first request takes ~5 seconds (DistilBERT +
  tokenizer load) and any caller with a sensible timeout fails. We use
  the modern ``lifespan`` context manager rather than the deprecated
  ``@app.on_event("startup")`` hook (removed in a future Starlette).
  (The ``/embed`` encoder is deliberately NOT warmed here — it loads lazily
  on the first ``/embed`` request, which therefore pays a one-time cold start.)

Run::

    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from ..config import config
from ..inference.embed import EMBED_DIM, embed_texts, is_embed_model_available
from ..inference.model_loader import get_load_error, is_model_available, load_model
from ..inference.predict import predict
from ..inference.topic_relevance import (
    get_load_error as get_topic_relevance_load_error,
)
from ..inference.topic_relevance import (
    is_topic_relevance_model_available,
    load_topic_relevance_model,
    predict_topic_relevance,
)
from ..inference.topic_reranker import (
    get_load_error as get_topic_reranker_load_error,
)
from ..inference.topic_reranker import (
    is_topic_reranker_model_available,
    load_topic_reranker_model,
    predict_topic_candidates,
)
from ..utils.logging import get_logger

logger = get_logger("api")

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """FastAPI lifespan handler — the modern replacement for the
    deprecated ``@app.on_event("startup")`` / ``"shutdown"`` hooks.

    Everything before the ``yield`` runs once at startup (we warm the
    model so the first /predict isn't a cold start); everything after
    runs at shutdown (nothing to tear down here — the model lives in a
    process-global cache the OS reclaims on exit). Keeping warmup in a
    plain ``_warmup_model()`` function means tests can call it directly
    without spinning up the ASGI lifecycle.
    """
    _warmup_model()
    yield


app = FastAPI(
    title="ThoughtTracker ML — Stance Classifier",
    version="0.1.0",
    description=(
        "Classifies a transcript excerpt's expressed stance toward a topic. "
        "It does not infer the speaker's private beliefs."
    ),
    lifespan=_lifespan,
)


def _warmup_model() -> None:
    """Eagerly load the model so the first /predict isn't a cold-start.

    Cold-loading a 268 MB DistilBERT on first request takes 5-10s on CPU and
    causes any caller with a sensible timeout (e.g. ML_CLASSIFIER_TIMEOUT_MS
    defaulting to 4000) to fail their very first call. Warming up here pays
    that cost once at boot.
    """
    if not is_model_available():
        if config.enable_mock_inference:
            logger.info("No trained model on disk; running in mock-inference mode")
        else:
            logger.warning(
                "No trained model on disk at %s. /predict will return 503 "
                "until you run `python -m src.training.train`.",
                config.model_dir,
            )
        return
    try:
        load_model()
        logger.info("Model warmed up and ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model warmup failed; will retry on first /predict: %s", exc)

    if is_topic_relevance_model_available():
        try:
            load_topic_relevance_model()
            logger.info("Topic relevance model warmed up and ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Topic relevance model warmup failed; will retry on first request: %s",
                exc,
            )

    if is_topic_reranker_model_available():
        try:
            load_topic_reranker_model()
            logger.info("Topic reranker model warmed up and ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Topic reranker model warmup failed; will retry on first request: %s",
                exc,
            )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``.

    Both fields are required and must be non-empty. Pydantic's
    ``min_length=1`` short-circuits empty-string inputs to a 422
    before our handler even runs, so ``predict()`` doesn't need to
    duplicate the validation.
    """

    topic: str = Field(..., min_length=1, description="Topic the excerpt is about.")
    text: str = Field(..., min_length=1, description="Transcript excerpt to classify.")


class PredictResponse(BaseModel):
    """Success response for ``POST /predict``.

    Always returned as a complete bundle — the caller never has to
    do a second request to fetch (e.g.) the per-label scores."""

    predictedLabel: str
    confidence: float
    labelScores: Dict[str, float]
    modelVersion: str


class EmbedRequest(BaseModel):
    """Request body for ``POST /embed`` — a batch of texts to embed."""

    texts: List[str] = Field(..., min_length=1, description="Texts to embed (1 or more).")


class EmbedResponse(BaseModel):
    """Success response for ``POST /embed`` — one L2-normalized vector per input,
    plus the dimension and whether the deterministic mock was used."""

    vectors: List[List[float]]
    dim: int
    modelVersion: str
    mockInference: bool


class TopicRelevancePredictRequest(BaseModel):
    """Request body for ``POST /predict-topic-relevance``."""

    topic: str = Field(..., min_length=1, description="Topic to test.")
    text: str = Field(..., min_length=1, description="Transcript excerpt.")


class TopicRelevancePredictResponse(BaseModel):
    """Success response for ``POST /predict-topic-relevance``."""

    predictedLabel: str
    confidence: float
    labelScores: Dict[str, float]
    modelVersion: str


class TopicCandidate(BaseModel):
    """One candidate topic from the local topic reranker."""

    topicSlug: str
    confidence: float


class TopicCandidatePredictRequest(BaseModel):
    """Request body for ``POST /predict-topics``."""

    text: str = Field(..., min_length=1, description="Transcript excerpt.")
    limit: int = Field(12, ge=1, le=20, description="Maximum candidate topics.")
    minScore: float = Field(
        0.2,
        ge=0,
        le=1,
        description="Minimum candidate probability before the relevance gate.",
    )


class TopicCandidatePredictResponse(BaseModel):
    """Success response for ``POST /predict-topics``."""

    topics: list[TopicCandidate]
    modelVersion: str


class HealthResponse(BaseModel):
    """Response for ``GET /health``. The fields together let an
    operator distinguish:

      - service alive but no model loaded yet (cold start in flight)
      - service alive, model loaded, real predictions
      - service alive, mock mode active (demo scenario, no model)
      - service alive but the LAST load attempt FAILED (``loadError`` is
        set and ``status`` is ``"degraded"``) — distinct from "no model
        on disk", which is a clean not-yet-trained state.
    """

    status: str
    modelLoaded: bool
    topicRelevanceModelLoaded: bool
    topicRerankerModelLoaded: bool
    modelVersion: str
    mockInference: bool
    loadError: str | None


class ErrorResponse(BaseModel):
    """Common error envelope shared across non-2xx responses.

    Matches the main backend's error contract so the
    ``mlClassifierClient`` can parse failures uniformly.
    """

    error: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Cheap liveness + readiness probe.

    Always 200 (the service is up if you can hit this endpoint).
    Fields:
      - ``status``: ``"ok"`` normally, or ``"degraded"`` when the last
        model-load attempt FAILED (``get_load_error()`` is set). The
        operations dashboard uses this to distinguish "service
        responded healthy" from "service responded but its model is
        broken" — both are HTTP-200, so the body must carry the signal.
      - ``modelLoaded``: whether ``config/model_dir/config.json`` is
        on disk. Distinct from "model has been loaded into memory" —
        the model is lazily loaded on first /predict.
      - ``modelVersion``: the version string the model would emit.
      - ``mockInference``: whether the service is in mock mode.
      - ``loadError``: the message from the most recent FAILED load across
        ANY of the three models (stance, topic-relevance, topic-reranker),
        or ``None`` when none have failed. Surfacing it here means an
        operator sees "model not loaded because <reason>" instead of
        having to grep server logs. We read the implemented
        ``get_load_error()`` for each model rather than relying on disk
        existence alone, so a corrupt-but-present model dir reports as
        degraded rather than falsely healthy. Previously only the stance
        model's error was surfaced; a topic-relevance/reranker warmup
        failure left ``/health`` falsely reporting ``"ok"`` because the
        warmup ``except`` block merely logged it.
    """
    # Aggregate the per-model load errors. The first non-empty message
    # (stance first, then the two topic models) becomes the reported
    # ``loadError`` and flips the status to ``degraded``; the joined
    # message makes a multi-model failure visible at a glance.
    model_errors = {
        "stance": get_load_error(),
        "topicRelevance": get_topic_relevance_load_error(),
        "topicReranker": get_topic_reranker_load_error(),
    }
    failed = {name: msg for name, msg in model_errors.items() if msg}
    load_error = (
        "; ".join(f"{name}: {msg}" for name, msg in failed.items()) if failed else None
    )
    return HealthResponse(
        status="degraded" if load_error else "ok",
        modelLoaded=is_model_available(),
        topicRelevanceModelLoaded=is_topic_relevance_model_available(),
        topicRerankerModelLoaded=is_topic_reranker_model_available(),
        modelVersion=config.model_version,
        mockInference=bool(config.enable_mock_inference),
        loadError=load_error,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_endpoint(req: PredictRequest):
    """Classify one ``(topic, text)`` pair.

    Three failure paths, each mapped to a distinct status code so
    the caller can branch correctly:
      - ``FileNotFoundError`` (no model on disk) → 503 + MODEL_NOT_LOADED.
      - ``ValueError`` (bad input — shouldn't happen post-pydantic
        validation but guards against runtime drift) → 400 + INVALID_INPUT.
      - Anything else → 500 + INTERNAL_ERROR.

    **Wire messages are generic on every error path.** The original
    ``str(exc)`` for a ``FileNotFoundError`` embeds the absolute model
    path ("No trained model found at /app/models/...") and a 500 can
    carry token fragments or PII from upstream libraries. We log the
    full exception server-side and return a fixed, path-free message so
    the contract's privacy stance holds for 503/400, not just 500.
    """
    try:
        result = predict(req.topic, req.text)
    except FileNotFoundError:
        logger.warning("Prediction requested but no model is loaded", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "error": "MODEL_NOT_LOADED",
                "message": "Model is not loaded — see server logs for details.",
            },
        )
    except ValueError:
        logger.info("Rejected /predict request with invalid input", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT",
                "message": "Invalid input — `topic` and `text` are required.",
            },
        )
    except Exception:  # noqa: BLE001
        # Log the full exception (incl. stack + any sensitive details
        # from underlying libraries) server-side, but return a generic
        # message on the wire. `str(exc)` here can leak local paths,
        # token fragments, or PII from upstream models.
        logger.exception("Unhandled error during prediction")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "Internal server error — see server logs for details.",
            },
        )

    return PredictResponse(
        predictedLabel=result["predictedLabel"],
        confidence=result["confidence"],
        labelScores=result["labelScores"],
        modelVersion=result["modelVersion"],
    )


@app.post(
    "/embed",
    response_model=EmbedResponse,
    responses={500: {"model": ErrorResponse}},
)
def embed_endpoint(req: EmbedRequest):
    """Embed a batch of texts into L2-normalized sentence vectors.

    Always 200 on valid input: when the real encoder can't load (no torch /
    mock mode) we degrade to the deterministic mock of the same dimension,
    flagged via ``mockInference`` so the caller knows. Any unexpected failure
    is logged in full server-side and returned as a generic 500 (no path/PII
    leakage), matching /predict's privacy stance.
    """
    try:
        vectors = embed_texts(req.texts)
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error during embedding")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "Internal server error — see server logs for details.",
            },
        )
    return EmbedResponse(
        vectors=vectors,
        dim=EMBED_DIM,
        modelVersion=config.base_model,
        mockInference=not is_embed_model_available(),
    )


@app.post(
    "/predict-topic-relevance",
    response_model=TopicRelevancePredictResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_topic_relevance_endpoint(req: TopicRelevancePredictRequest):
    """Classify whether a transcript excerpt is actually about a topic.

    Same error contract and generic-message privacy stance as
    ``/predict``: 503 MODEL_NOT_LOADED, 400 INVALID_INPUT, 500
    INTERNAL_ERROR, none of which echo the underlying exception text
    (which can leak model paths or PII).
    """
    try:
        result = predict_topic_relevance(req.topic, req.text)
    except FileNotFoundError:
        logger.warning(
            "Topic relevance requested but no model is loaded", exc_info=True
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": "MODEL_NOT_LOADED",
                "message": "Model is not loaded — see server logs for details.",
            },
        )
    except ValueError:
        logger.info(
            "Rejected /predict-topic-relevance request with invalid input",
            exc_info=True,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT",
                "message": "Invalid input — `topic` and `text` are required.",
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error during topic relevance prediction")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "Internal server error — see server logs for details.",
            },
        )

    return TopicRelevancePredictResponse(
        predictedLabel=result["predictedLabel"],
        confidence=result["confidence"],
        labelScores=result["labelScores"],
        modelVersion=result["modelVersion"],
    )


@app.post(
    "/predict-topics",
    response_model=TopicCandidatePredictResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_topics_endpoint(req: TopicCandidatePredictRequest):
    """Return high-recall controlled-taxonomy topic candidates for a chunk.

    Same error contract and generic-message privacy stance as
    ``/predict``: 503 MODEL_NOT_LOADED, 400 INVALID_INPUT, 500
    INTERNAL_ERROR, none of which echo the underlying exception text.
    """
    try:
        result = predict_topic_candidates(
            req.text,
            limit=req.limit,
            min_score=req.minScore,
        )
    except FileNotFoundError:
        logger.warning(
            "Topic candidates requested but no model is loaded", exc_info=True
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": "MODEL_NOT_LOADED",
                "message": "Model is not loaded — see server logs for details.",
            },
        )
    except ValueError:
        logger.info(
            "Rejected /predict-topics request with invalid input", exc_info=True
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT",
                "message": "Invalid input — `text` is required.",
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error during topic candidate prediction")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "Internal server error — see server logs for details.",
            },
        )

    return TopicCandidatePredictResponse(
        topics=result["topics"],
        modelVersion=result["modelVersion"],
    )


def _format_validation_errors(exc) -> str:
    """Render Pydantic validation errors as a single human-readable string.

    ``ErrorResponse`` declares ``message: str`` and the ``mlClassifierClient``
    reads ``message`` as a string — returning the raw ``exc.errors()`` LIST here
    made the client silently drop the detail (it fell back to ``HTTP 400``).
    """
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
    )


@app.exception_handler(RequestValidationError)
def _request_validation_handler(_request, exc: RequestValidationError):
    """Pydantic validation failures (e.g. ``topic`` missing or empty)
    default to a 422 in FastAPI. We rewrap to 400 + INVALID_INPUT so
    the wire shape matches every other 4xx response in the API surface.
    """
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT", "message": _format_validation_errors(exc)},
    )


@app.exception_handler(ValidationError)
def _validation_handler(_request, exc: ValidationError):
    """Catches Pydantic ``ValidationError``s that escape route-level
    handling (rare, but possible if a model is constructed by hand
    inside a handler). Same envelope as the request-validation
    handler above so the caller sees a consistent shape."""
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT", "message": _format_validation_errors(exc)},
    )
