# ThoughtTracker ML - Transcript Intelligence Pipeline

Built by Jason Lin, Senior Full-Stack AI/ML Software Engineer, as the
machine-learning and reanalysis pipeline behind ThoughtTracker.

[![CI](https://github.com/jason-lin-uo/thoughttracker-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/jason-lin-uo/thoughttracker-ml/actions/workflows/ci.yml)

This is the ML half of **ThoughtTracker**, not a throwaway side repo. It
fine-tunes and serves transcript classifiers, owns topic relevance/reranking
artifacts, stores final calibration metrics, and provides the owner workflow for
refreshing creators and promoting a new product snapshot.

## Product preview

This repo powers the ML and reanalysis side of the public app. The user-facing
product lives in [`jason-lin-uo/thoughttracker`](https://github.com/jason-lin-uo/thoughttracker)
and is deployed at <https://thoughttracker-web-415a.onrender.com/>.

![ThoughtTracker dashboard showing the real five-creator corpus and a featured report.](https://github.com/jason-lin-uo/thoughttracker/raw/main/docs/assets/screenshots/dashboard.png)

> The classifier scores the **stance expressed in a transcript excerpt**
> toward a given topic. It does not claim to know the speaker's private
> beliefs. It is an excerpt classifier, not a truth engine.

---

## What This Demonstrates

- Practical ML engineering with PyTorch, Hugging Face Transformers, FastAPI,
  scikit-learn, pandas, and pytest.
- Model-serving discipline: explicit contracts, health checks, predictable
  error shapes, and no fabricated runtime predictions when artifacts are
  missing.
- Product-aware ML calibration: topic relevance, topic reranking, final policy
  thresholds, and honest metrics for rare-topic limitations.
- Cross-repo integration with the TypeScript product through
  [`integration_contract.md`](integration_contract.md).
- Operational ownership: transcript refresh scripts, one-command creator
  onboarding, Git LFS artifacts, and CI-enforced 100% test coverage.

---

## Tech And Concepts

**Stack:** Python, FastAPI, PyTorch, Hugging Face Transformers, Hugging Face
Datasets, scikit-learn, pandas, NumPy, matplotlib, pytest, Ruff, Git LFS,
DistilBERT.

**ML/NLP:** stance classification, topic relevance, topic reranking,
sentence embeddings, model serving, calibration, precision, recall, F1,
confusion matrices, hard-negative validation, active learning, and model
artifact packaging.

**Engineering:** API contracts, health checks, explicit error envelopes,
cross-repo integration, transcript ingestion, idempotent refresh workflows,
owner-controlled reanalysis, reproducible metrics, and no fabricated runtime
inference when model artifacts are missing.

---

## Future Enhancements And Upcoming Features

- Active learning loop for new creators: use the frozen policy first, review
  uncertain rows, append validated labels, and retrain only when metrics justify
  it.
- MCP-compatible evidence service so AI tools can query transcript evidence,
  labels, topic candidates, and model outputs through a standard interface.
- RAG-style retrieval experiments over transcript chunks, with relevance gates
  and human-readable source citations before exposing any search-like UX.
- Model-drift monitoring across new creators, topics, and transcript styles so
  retraining is triggered by evidence instead of guesswork.
- Richer evaluation dashboards for rare-topic macro F1, false positives,
  hard negatives, per-creator drift, and post-refresh quality audits.
- Optional production serving path with autoscaling, observability, and
  versioned model rollout/rollback controls.

---

## What To Review First

- Start with
  [`reports/metrics/topic_selection_policy_gold_standard_metrics.md`](reports/metrics/topic_selection_policy_gold_standard_metrics.md)
  for the frozen topic-selection results.
- Review [`integration_contract.md`](integration_contract.md) to see how this
  FastAPI service connects to the TypeScript product.
- Inspect [`src/inference/`](src/inference/) and [`scripts/`](scripts/) for model
  loading, topic relevance, reranking, and owner refresh workflows.
- Use the main app screenshots and live app for the product-facing result of
  this pipeline.

---

## How it complements ThoughtTracker

| Concern                                                                     | Owner                                |
| --------------------------------------------------------------------------- | ------------------------------------ |
| YouTube ingestion, transcripts, chunking, DB, UI, charts, evidence, reports | **ThoughtTracker** (main TS app)     |
| Stance label + confidence on a transcript excerpt                           | **ThoughtTracker ML** (this project) |
| Topic relevance, reranking, calibration, and evaluation artifacts           | **ThoughtTracker ML** (this project) |
| Rationale + evidence-quote selection + creator reports                      | LLM in the main app                  |

The main app selects a stance-analysis provider via env var:

- `llm` - local/OpenAI/Anthropic LLM analysis in the main app
- `custom_ml` - call **this** FastAPI service for stance + confidence
- `hybrid` - this service produces stance + confidence; the LLM produces rationale + evidence quote

Topic relevance/reranking are wired to this service for owner analysis flows.
The `/embed` endpoint is retained for offline embedding refreshes, but the
public app no longer exposes a semantic-search UI. The final product expects
the committed runtime artifacts when ML-backed analysis is enabled; see
[`integration_contract.md`](./integration_contract.md).

---

## Detailed Stack Notes

- **Python 3.11+**
- **PyTorch** + **Hugging Face Transformers** (DistilBERT base)
- **Hugging Face Datasets** for tokenization
- **scikit-learn** for splitting + metrics
- **pandas** + **numpy** for data handling
- **matplotlib** for charts
- **FastAPI** + **uvicorn** + **pydantic** for serving
- **pytest** for tests

---

## Project layout

```
thoughttracker-ml/
+-- README.md
+-- requirements.txt
+-- .env.example
+-- integration_contract.md
+-- data/
| +-- processed/ # committed gold-standard CSVs + ignored local outputs
| +-- transcripts/ # committed five-creator .txt corpus + ignored raw/excluded files
+-- src/
| +-- config.py # env-driven configuration
| +-- data/
| | +-- label_schema.py # 5-label canonical schema
| | +-- load_dataset.py # csv loader + validation
| | +-- preprocess.py # train/val/test split + tokenization prep
| +-- training/
| | +-- train.py # fine-tune + save model
| | +-- evaluate.py # re-eval saved model on test split
| | +-- metrics.py # accuracy, precision, recall, F1, CM, charts
| +-- inference/
| | +-- model_loader.py # lazy, cached stance-model loader
| | +-- predict.py # CLI + reusable predict()
| | +-- embed.py # mean-pooled DistilBERT sentence embeddings (POST /embed)
| | +-- topic_relevance.py # topic-relevance classifier inference
| | +-- topic_reranker.py # controlled-taxonomy topic candidate reranker
| | +-- _device.py # cuda > mps > cpu device selection
| +-- api/main.py # FastAPI service (5 endpoints; see "Run the API")
| +-- utils/{paths,logging}.py
+-- tests/ # pytest: schema, dataset, predict, api
+-- models/ # committed runtime artifacts + ignored local checkpoints
+-- reports/
 +-- metrics/ # JSON metric files
 +-- figures/ # confusion-matrix PNGs
```

---

## Dataset format

CSV with these columns:

| column  | type | notes                                                          |
| ------- | ---- | -------------------------------------------------------------- |
| `id`    | int  | row identifier                                                 |
| `topic` | str  | topic the excerpt is about                                     |
| `text`  | str  | transcript excerpt to classify                                 |
| `label` | str  | one of: `supportive`, `opposed`, `neutral`, `mixed`, `unclear` |

The clean repo does not ship a toy stance corpus. For retraining, set
`DATASET_PATH` in `.env` to a local CSV with the schema above.

---

## Current portfolio pipeline

The current portfolio path is already trained and audited. It uses:

- a custom stance classifier served by this FastAPI app
- a topic relevance model to reject bad topic assignments
- a topic reranker integration used by the ThoughtTracker backend
- a frozen gold-standard topic-selection policy at
  `models/topic-selection-policy-gold-standard`
- a curated final database generated from validated topic assignments

Current frozen topic-selection metrics for the validated five-creator corpus:

| Metric      | Result |
| ----------- | -----: |
| Exact match | 95.44% |
| Micro F1    | 98.40% |
| Precision   | 97.82% |
| Recall      | 98.98% |
| Macro F1    | 75.35% |

Checkpoint folders and local run logs are intentionally kept local and ignored
by Git. Final runtime model artifacts are committed via Git LFS. The
public-facing proof lives in the final metrics artifacts:

```text
reports/metrics/topic_selection_policy_gold_standard_metrics.md
reports/metrics/topic_selection_policy_gold_standard_metrics.json
```

Those files record the frozen audit, metrics, current limitations, and the
interview-ready explanation. Broad manual labeling/browser workflows are retired
for the demo pipeline.

### Gold-standard artifact packaging

The production runtime artifacts are committed with the repository so a fresh
clone can run the same gold-standard pipeline:

- `models/stance-classifier`
- `models/topic-relevance-classifier-supervalidation-hardneg2x-l512`
- `models/topic-reranker-tfidf-sgd-supervalidation`
- `models/topic-selection-policy-gold-standard`
- `data/processed/thoughttracker_topic_relevance_gold_standard.csv`
- `data/processed/thoughttracker_topic_reranker_gold_standard.csv`

Large binaries and gold-standard CSVs are stored with Git LFS. Training
checkpoint directories remain ignored because they are redundant local history,
not runtime requirements.

---

## Local setup

The quickest path is the included Makefile, which pins pip to public PyPI so
a corporate `pip.conf` (e.g. an Artifactory mirror) doesn't block installs:

```bash
cd thoughttracker-ml
make install # creates .venv + installs from https://pypi.org/simple/
cp .env.example .env
```

Or do it by hand. **Use the venv's own `pip` explicitly** - relying on
`source activate && pip ...` is fragile on macOS with pyenv/Homebrew, where
`pip` can resolve to a different Python and the install lands in your
user-site instead of the venv:

```bash
cd thoughttracker-ml
# Python 3.11+ is required  -  the code uses PEP 604 `X | None` annotations
# (3.10+) and sklearn >=3.11. A bare `python` may be an older pyenv default,
# which builds a venv that fails at import; pin the interpreter explicitly.
python3.11 -m venv .venv
.venv/bin/pip install --index-url https://pypi.org/simple/ --upgrade pip
.venv/bin/pip install --index-url https://pypi.org/simple/ -r requirements.txt
cp .env.example .env
```

To verify the install landed in the venv (not user-site):

```bash
.venv/bin/python -c "import torch, fastapi; print(torch.__file__)"
# should print a path inside .venv/lib/...
```

> **Why the `--index-url`** If your `pip` is configured to use a corporate
> Artifactory or any private mirror that's unreachable from your network,
> the install will hang on retries. Passing `--index-url https://pypi.org/simple/`
> forces public PyPI for this one project.

> **Heads-up:** `torch` is the biggest dependency (~700 MB on macOS, more
> with CUDA wheels on Linux). On a fresh laptop expect 3 - 5 minutes.

Once installed, the rest of the workflow has shortcut targets:

```bash
DATASET_PATH=data/processed/stance_training.csv make train
make evaluate # re-evaluate the saved model on the test split
make test # run pytest
make serve # uvicorn on port 8000 (eagerly warms the model at boot)
make predict TOPIC="ai" TEXT="I support this and I am in favor."
```

---

## Train

```bash
python -m src.training.train
```

Reads `DATASET_PATH`, fine-tunes `BASE_MODEL` (DistilBERT by default), saves:

- model + tokenizer -> `models/stance-classifier/`
- test metrics -> `reports/metrics/test_metrics.json`
- confusion matrix PNG -> `reports/figures/confusion_matrix.png`

Training time depends on the dataset size and hardware.

---

## Evaluate (against a saved model)

```bash
python -m src.training.evaluate
```

Re-runs evaluation on the held-out test split using the model in
`MODEL_DIR`. Saves a fresh `reports/metrics/eval_metrics.json` and
`reports/figures/confusion_matrix_eval.png`.

---

## Predict (CLI)

```bash
python -m src.inference.predict \
 --topic "foreign policy" \
 --text "I disagree with this approach and I worry about its impact."
```

Example output:

```json
{
  "topic": "foreign policy",
  "text": "I disagree with this approach and I worry about its impact.",
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

If a required model artifact is missing, the script exits with a clear
instruction. Test-only inference shortcuts are reserved for automated tests and
local wiring checks; the portfolio product should use the committed final
artifacts.

---

## Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

- `GET /health` -> `{ status, modelLoaded, topicRelevanceModelLoaded, topicRerankerModelLoaded, modelVersion, mockInference, loadError }`
- `POST /predict` -> `{ predictedLabel, confidence, labelScores, modelVersion }` - stance of an excerpt toward a topic
- `POST /embed` -> `{ vectors, dim, modelVersion, mockInference }` - 768-d L2-normalized sentence embeddings (mean-pooled DistilBERT) for owner/offline embedding refreshes via `EMBEDDING_PROVIDER=ml`
- `POST /predict-topic-relevance` -> `{ predictedLabel, confidence, labelScores, modelVersion }` - is the excerpt actually about the topic
- `POST /predict-topics` -> `{ topics: [{ topicSlug, confidence }], modelVersion }` - high-recall controlled-taxonomy topic candidates

Error responses use `{ error, message }` with codes `MODEL_NOT_LOADED` (503),
`INVALID_INPUT` (400), and `INTERNAL_ERROR` (500). If a required runtime
artifact is unavailable, the service returns an explicit error instead of
fabricating a prediction or vector.

See [`integration_contract.md`](./integration_contract.md) for the full
request/response shapes. OpenAPI docs are auto-mounted at
<http://localhost:8000/docs>.

---

## Run tests

```bash
pytest -q
```

Tests cover label schema, dataset validation, real-path prediction behavior
with lightweight test doubles, embeddings, topic relevance/reranker inference,
and the FastAPI endpoint shapes. They keep 100% line coverage
(CI-enforced via `--cov-fail-under=100`) without requiring a full training run.

---

## Limitations

- Current topic-selection baseline lives in the frozen policy at
  `models/topic-selection-policy-gold-standard`.
- Stance is inferred from text only. No audio, tone, prosody, or sarcasm signal.
- `unclear` is a first-class label, but the model is not a sarcasm detector.
- This service is single-process. Add a reverse proxy / autoscaler for prod.

---

## Integration with ThoughtTracker

See [`integration_contract.md`](./integration_contract.md) for the full
request/response contract, environment variables, provider-mode behavior,
and graceful-degradation rules.

Topic relevance/reranking call this service in owner analysis flows; the
embedding endpoint remains available for offline refreshes. Stance analysis is
env-gated and ready via the provider switch. The stance flow in the main app is:

```
transcript chunk + topic
 -> stanceAnalysis.service.ts
 -> provider switch (llm | custom_ml | hybrid)
 -> in hybrid: ML classifier -> stance + confidence
 LLM -> rationale + evidence quote
 -> store ChunkTopicAnalysis
```
