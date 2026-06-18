# \_LEARN.md - `thoughttracker-ml/src/inference/`

> Runtime use of trained artifacts: stance prediction, embeddings, topic
> relevance, topic reranking, and device selection.

## What This Folder Does

Training creates artifacts. Inference loads those artifacts and answers
requests from either the CLI or the FastAPI service.

Files:

- `model_loader.py`: lazy, cached, thread-safe stance-model loader.
- `predict.py`: stance prediction CLI and library function.
- `embed.py`: mean-pooled DistilBERT embeddings for owner/offline refreshes.
- `topic_relevance.py`: "is this excerpt actually about the topic"
- `topic_reranker.py`: high-recall controlled-taxonomy topic candidates.
- `_device.py`: chooses `cuda`, then `mps`, then `cpu`.

Runtime inference requires the expected artifact to exist. Missing or corrupt
artifacts raise explicit errors so the main app does not mistake a degraded
state for the gold-standard model.

## `model_loader.py`

Loads the saved stance model from disk on first use and caches it for the
process lifetime. A lock prevents two concurrent requests from loading the
same model twice.

Use `clear_cache()` in tests when a test needs to force a reload.

## `predict.py`

`predict(topic, text)` formats input exactly like training did, loads the
stance model, runs a forward pass, applies softmax, and returns:

```python
{
 "topic": "AI safety",
 "text": "I think AI...",
 "predictedLabel": "supportive",
 "confidence": 0.83,
 "labelScores": {
 "supportive": 0.83,
 "opposed": 0.05,
 "neutral": 0.07,
 "mixed": 0.03,
 "unclear": 0.02,
 },
 "modelVersion": "stance-classifier-v1",
}
```

The API route returns the same prediction fields without echoing `topic` and
`text`.

The most important invariant: training and inference must use the same
`build_input_text(topic, text)` format. If that format drifts, the model can
receive inputs unlike the data it was trained on.

## `embed.py`

Builds 768-dimensional L2-normalized sentence embeddings used by owner/offline
refresh workflows. The public app no longer exposes a semantic-search UI.
Missing encoder dependencies or invalid runtime output raise explicit errors.

## Topic Models

`topic_relevance.py` and `topic_reranker.py` support the final topic-selection
policy in the main app:

- the reranker proposes likely controlled-taxonomy candidates;
- the relevance model checks each candidate against the transcript chunk;
- the main app applies the frozen policy thresholds/margins/suppression rules.

## Prediction Flow

```text
predict(topic, text)
 -> build_input_text(topic, text)
 -> model_loader.load_model()
 -> tokenizer(..., max_length=<trained length>, truncation=True)
 -> model.forward(...)
 -> softmax probabilities
 -> ID_TO_LABEL[argmax]
 -> typed prediction payload
```

## Debug Map

| Symptom                          | Start Here                                                                                |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| Prediction says model not loaded | Confirm `MODEL_DIR` contains a real Hugging Face artifact                                 |
| Prediction quality looks wrong   | Confirm training/inference formatting both use `build_input_text`                         |
| First request is slow            | Lazy model loading; warm the service or call `/health` before demos                       |
| Embedding route fails            | `embed.py`, ML dependencies, and `ML_CLASSIFIER_URL` from the main app                    |
| Topic final policy fails         | Topic relevance/reranker artifacts and `topic-selection-policy-gold-standard/policy.json` |
