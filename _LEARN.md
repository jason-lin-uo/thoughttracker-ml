# \_LEARN.md - thoughttracker-ml

Author: Jason Lin

This repo is the ML half of ThoughtTracker. It serves local models for the main
TypeScript app and stores the final transcript/model artifacts used by the
portfolio product.

## What This Repo Owns

- real transcript text files for the five current creators
- stance classifier runtime artifacts
- topic relevance runtime artifacts
- topic reranker runtime artifacts
- final topic-selection policy artifacts
- sentence embedding endpoint for offline embedding refreshes and future owner workflows
- transcript refresh and owner-onboarding scripts
- pytest suite with 100% coverage

## Runtime Contract

The main app calls this service at `ML_CLASSIFIER_URL`, usually
`http://localhost:8000`.

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /embed`
- `POST /predict-topic-relevance`
- `POST /predict-topics`

The service should use committed runtime artifacts fetched through Git LFS. If
a required artifact is missing, it returns a clear error instead of inventing a
prediction.

## Important Files

| Path                               | Purpose                                               |
| ---------------------------------- | ----------------------------------------------------- |
| `README.md`                        | Main repo overview and setup.                         |
| `integration_contract.md`          | Wire contract between this repo and `thoughttracker`. |
| `.env.example`                     | Runtime and training configuration template.          |
| `src/api/main.py`                  | FastAPI app and endpoint definitions.                 |
| `src/inference/predict.py`         | Stance inference.                                     |
| `src/inference/embed.py`           | Sentence embeddings.                                  |
| `src/inference/topic_relevance.py` | Topic relevance inference.                            |
| `src/inference/topic_reranker.py`  | Candidate topic inference.                            |
| `src/training`                     | Optional retraining and evaluation code.              |
| `scripts/fetch_transcripts.py`     | YouTube transcript refresh utility.                   |
| `models/`                          | Runtime model artifacts.                              |
| `reports/metrics/`                 | Final audit metrics.                                  |

## Current Product Artifacts

```text
models/stance-classifier
models/topic-relevance-classifier-supervalidation-hardneg2x-l512
models/topic-reranker-tfidf-sgd-supervalidation
models/topic-selection-policy-gold-standard
data/processed/thoughttracker_topic_relevance_gold_standard.csv
data/processed/thoughttracker_topic_reranker_gold_standard.csv
```

## Current Metrics

| Metric      | Result |
| ----------- | -----: |
| Exact match | 95.44% |
| Micro F1    | 98.40% |
| Precision   | 97.82% |
| Recall      | 98.98% |
| Macro F1    | 75.35% |

## Local Run

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Windows:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Verification

```bash
python -m pytest -q
```

Current verified status: 183 tests passing with 100% coverage.

## What Not To Recreate

Do not bring back old ChatGPT packet folders, failed labeling rounds, temporary
VTT caches, or ad hoc calibration outputs. The clean product state is the final
transcripts, runtime artifacts, processed gold-standard CSVs, metrics reports,
and source code.
