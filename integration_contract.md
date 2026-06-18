# Integration Contract - ThoughtTracker <-> ThoughtTracker ML

Author: Jason Lin

This contract describes how the TypeScript ThoughtTracker backend calls the
Python ML service. It is the source of truth for runtime request/response
shapes, provider settings, and expected failure behavior.

## Purpose

The ML service supplies:

- stance prediction for a `(topic, transcript excerpt)` pair
- topic relevance scoring
- controlled-taxonomy topic candidate reranking
- sentence embeddings for owner/offline embedding refreshes

The ML service does not write reports, choose final report wording, or mutate
the main database. The TypeScript backend owns persistence, final policy
application, evidence linking, and report generation.

## Required Runtime Artifacts

Fetch large artifacts with Git LFS:

```bash
git lfs pull
```

Required folders:

```text
models/stance-classifier
models/topic-relevance-classifier-supervalidation-hardneg2x-l512
models/topic-reranker-tfidf-sgd-supervalidation
models/topic-selection-policy-gold-standard
```

If a required artifact is missing, the service should return an explicit error
or fail startup checks. It should not fabricate runtime predictions.

## Main App Environment

Recommended local product values:

```env
AI_PROVIDER=local
AI_MODEL=llama3.1:8b
LOCAL_LLM_BASE_URL=http://localhost:11434

EMBEDDING_PROVIDER=ml
YOUTUBE_PROVIDER=youtube
STANCE_ANALYSIS_PROVIDER=custom_ml
ML_CLASSIFIER_URL=http://localhost:8000
ML_CLASSIFIER_TIMEOUT_MS=4000

TOPIC_ASSIGNMENT_PROVIDER=final_policy
TOPIC_SELECTION_POLICY_PATH=../../thoughttracker-ml/models/topic-selection-policy-gold-standard/policy.json
TOPIC_RERANKER_LIMIT=12
TOPIC_RERANKER_MIN_SCORE=0.2
TOPIC_RELEVANCE_THRESHOLD=0.8
```

Allowed main-app stance providers are `llm`, `custom_ml`, and `hybrid`.

## Health

`GET /health`

Successful response:

```json
{
  "status": "ok",
  "modelLoaded": true,
  "topicRelevanceModelLoaded": true,
  "topicRerankerModelLoaded": true,
  "modelVersion": "stance-classifier-v1",
  "mockInference": false,
  "loadError": null
}
```

## Stance Prediction

`POST /predict`

Request:

```json
{
  "topic": "foreign policy",
  "text": "I disagree with this approach and I worry about its impact."
}
```

Success:

```json
{
  "predictedLabel": "opposed",
  "confidence": 0.82,
  "labelScores": {
    "supportive": 0.02,
    "opposed": 0.82,
    "neutral": 0.08,
    "mixed": 0.05,
    "unclear": 0.03
  },
  "modelVersion": "stance-classifier-v1"
}
```

The ML service emits five labels: `supportive`, `opposed`, `neutral`, `mixed`,
and `unclear`. The main app also supports `insufficient_evidence` for chunks
without enough signal.

## Topic Relevance

`POST /predict-topic-relevance`

Request:

```json
{
  "topic": "AI policy",
  "text": "The segment discusses regulation of foundation models."
}
```

Success:

```json
{
  "predictedLabel": "relevant",
  "confidence": 0.91,
  "labelScores": {
    "irrelevant": 0.09,
    "relevant": 0.91
  },
  "modelVersion": "topic-relevance-supervalidation-hardneg2x-l512"
}
```

The backend uses this as a false-positive gate before final topic selection.

## Topic Candidates

`POST /predict-topics`

Request:

```json
{
  "text": "The segment compares AI regulation, model safety, and OpenAI.",
  "limit": 12,
  "minScore": 0.2
}
```

Success:

```json
{
  "topics": [
    { "topicSlug": "ai_policy_and_regulation", "confidence": 0.86 },
    { "topicSlug": "openai_company", "confidence": 0.64 }
  ],
  "modelVersion": "topic-reranker-tfidf-sgd-supervalidation"
}
```

This endpoint proposes candidates. The final backend policy still applies
taxonomy, relevance, thresholds, and display rules.

## Embeddings

`POST /embed`

Request:

```json
{
  "texts": ["the first text to embed", "the second text"]
}
```

Success:

```json
{
  "vectors": [
    [0.0123, -0.0456],
    [0.0789, -0.0123]
  ],
  "dim": 768,
  "modelVersion": "distilbert-base-uncased",
  "mockInference": false
}
```

The response contains one L2-normalized vector per input text. The backend
stores/query vectors in a `vector(768)` pgvector column.

`mockInference: true` is allowed only when `ENABLE_MOCK_INFERENCE=true` is set
for tests or local diagnostics. In normal runtime mode, missing embedding
artifacts must return `503 MODEL_NOT_LOADED` so the main app never stores
fabricated vectors.

## Error Shape

Errors use this envelope:

```json
{
  "error": "MODEL_NOT_LOADED",
  "message": "Model is not loaded; see server logs for details."
}
```

| HTTP | Code               | Meaning                                                     |
| ---- | ------------------ | ----------------------------------------------------------- |
| 400  | `INVALID_INPUT`    | Request body is missing required fields or has wrong types. |
| 503  | `MODEL_NOT_LOADED` | Required model artifact is unavailable.                     |
| 500  | `INTERNAL_ERROR`   | Unexpected inference failure.                               |

Messages should stay generic and path-free. Log detailed exceptions server-side.

## Backend Degradation Rules

The backend should:

- use successful ML responses directly
- surface explicit configuration/artifact failures in owner workflows
- avoid saving fabricated report text or fabricated vectors
- fail clearly when the configured local/ML services are unavailable
- keep test doubles inside tests only

## Local Run

```bash
cd thoughttracker-ml
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Windows:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
