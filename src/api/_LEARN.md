# \_LEARN.md - thoughttracker-ml/src/api

This folder exposes the ML service over HTTP. The main ThoughtTracker backend
uses it for stance, embeddings, topic relevance, and topic candidate reranking.

## `main.py`

FastAPI application with five endpoints:

- `GET /health`
- `POST /predict`
- `POST /embed`
- `POST /predict-topic-relevance`
- `POST /predict-topics`

## Health

`GET /health` reports whether the stance, relevance, and reranker artifacts are
available. A degraded response means the caller should treat the service as not
ready for the corresponding runtime path.

## Prediction

`POST /predict` accepts:

```json
{ "topic": "AI safety", "text": "I think AI is fantastic..." }
```

It returns:

```json
{
  "predictedLabel": "supportive",
  "confidence": 0.83,
  "labelScores": {
    "supportive": 0.83,
    "opposed": 0.05,
    "neutral": 0.07,
    "mixed": 0.03,
    "unclear": 0.02
  },
  "modelVersion": "stance-classifier-v1"
}
```

## Embeddings

`POST /embed` accepts:

```json
{ "texts": ["first text", "second text"] }
```

It returns one 768-dimensional L2-normalized vector per text. The backend keeps
this endpoint available for owner/offline embedding refreshes when
`EMBEDDING_PROVIDER=ml`; the public app no longer exposes a semantic-search UI.

## Topic Endpoints

- `/predict-topic-relevance` checks whether a chunk is truly about a topic.
- `/predict-topics` proposes controlled-taxonomy topic candidates.

The TypeScript backend applies the final topic-selection policy after these
model outputs.

## Error Contract

Errors are JSON, not HTML:

```json
{
  "error": "MODEL_NOT_LOADED",
  "message": "Model is not loaded - see server logs for details."
}
```

Common codes:

- `400 INVALID_INPUT`
- `503 MODEL_NOT_LOADED`
- `500 INTERNAL_ERROR`

Messages stay generic and path-free. Detailed exceptions are logged server-side.

## Startup Behavior

The FastAPI lifespan hook warms the stance/topic models. The embedding encoder
loads lazily on first `/embed` request. Missing artifacts are reported clearly
instead of producing fabricated runtime results.
