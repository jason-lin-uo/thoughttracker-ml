# \_LEARN.md — `thoughttracker-ml/tests/`

> The proof that the ML repo works. Same pattern as the backend tests:
> ~190 tests across 21 files at 100% line coverage (CI-enforced via
> `--cov-fail-under=100`), mirroring the `src/` layout.

---

## The story of this folder

If `src/` is the factory floor, this folder is the **quality
inspector** who walks the line every shift and verifies every machine
is calibrated. For an ML repo, "verified" means a slightly different
thing than for a backend:

- **We don't test that the model is accurate** — that's what
  `reports/metrics/test_metrics.json` is for. Accuracy is a property
  of the training run, not the code.
- **We test that the code paths work** — the dataset loader rejects
  malformed CSVs, the tokenizer concatenation matches between
  training and inference, the API returns the right shape, the mock
  path activates when no model is present.

In other words: we're verifying the **plumbing** is correct, not
that **the water tastes good**.

Tests use `pytest` + `pytest-cov`. The whole suite runs in <5 seconds
because tests use small synthetic datasets and mock out the heavy
ML calls. Pytest discovers them automatically (`pytest.ini` points at
`tests/`).

---

## File-by-file

### `__init__.py`

Empty. Required for pytest to treat `tests/` as a package.

---

### Source-mirroring tests

Each test file targets one layer:

| File                                                            | Layer                                                    | What it covers                                                                              |
| --------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `test_config.py`                                                | `src/config.py`                                          | Env-var loading, defaults, path resolution                                                  |
| `test_label_schema.py`                                          | `src/data/label_schema.py`                               | All five labels, ID round-trip, `is_valid_label`                                            |
| `test_dataset.py`                                               | `src/data/load_dataset.py`                               | Loads valid CSV; rejects missing columns, bad labels, blank rows; `build_input_text` format |
| `test_preprocess.py`                                            | `src/data/preprocess.py`                                 | Leakage-safe / stratified split shape; `to_model_inputs` (`[TOPIC] … [TEXT] …`)             |
| `test_metrics.py`                                               | `src/training/metrics.py`                                | Accuracy / F1 / confusion matrix numbers match sklearn                                      |
| `test_training_modules.py`                                      | `src/training/train.py` + `evaluate.py`                  | Trains and evaluates against a tiny toy dataset end-to-end                                  |
| `test_model_loader.py`                                          | `src/inference/model_loader.py`                          | Lazy load, thread-safety with concurrent calls, cached singleton                            |
| `test_predict_mock.py` / `test_predict_extra.py`                | `src/inference/predict.py`                               | Mock heuristic labels; CLI mode (JSON to stdout) + error cases                              |
| `test_embed.py`                                                 | `src/inference/embed.py`                                 | Mock + real embedding shape (768-d, L2-normalized)                                          |
| `test_topic_inference_modules.py`                               | `src/inference/topic_relevance.py` + `topic_reranker.py` | Relevance gate + candidate reranking paths                                                  |
| `test_device_select.py`                                         | `src/inference/_device.py`                               | `cuda > mps > cpu` device selection                                                         |
| `test_api.py` / `test_api_extra.py`                             | `src/api/main.py`                                        | All five endpoints' happy paths + error envelopes via httpx TestClient                      |
| `test_fetch_transcripts.py` / `test_fetch_transcripts_ytdlp.py` | `scripts/fetch_transcripts*.py`                          | The transcript-fetching CLIs (mocked YouTube / yt-dlp)                                      |
| `test_ingest_transcripts.py`                                    | `scripts/ingest_transcripts.py`                          | Bulk-import POST + status polling (mocked backend)                                          |

---

### Coverage-mop files

Following the same pattern as the main repo, these exist to hit
branches the natural-flow tests don't reach:

| File                          | Targets                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `test_coverage_lift.py`       | First-pass mop — error branches in load_dataset, edge cases in dataset validation |
| `test_final_lift.py`          | Final-mile branches — defensive `else` paths, optional-arg combinations           |
| `test_full_coverage_edges.py` | Edge paths in the topic API + inference modules                                   |

Without these, coverage sits at ~94%. With them, it hits 100% line. As
in the TypeScript repo, this is intentional engineering: rather than
weaken the coverage gate, write focused tests for the edges.

Separately, `test_hardening.py` is **not** a coverage-mop file: every test in
it asserts a *behavior* a careful reviewer would probe — group-aware split /
leakage guard, loader thread-safety, the exact error code per the integration
contract (no 400-or-422 ambiguity), generic path-free 503 messages, the
device selector (cuda > mps > cpu), env-var validation, the `/health`
load-error surface + lifespan warmup, and the `len(probs) == len(LABELS)`
schema-drift guard.

---

## How the trickier tests work

### Training tests (`test_training_modules.py`)

Training tests are tricky because the real training run takes 10
minutes and downloads DistilBERT. The test does:

1. Writes a **tiny synthetic CSV** (~50 rows, ~10 per label).
2. Sets env vars: `NUM_TRAIN_EPOCHS=1`, `MAX_LENGTH=32`,
   `TRAIN_BATCH_SIZE=4`. (In plain terms: turn down all the dials — one pass through the data, short inputs, small batches — so the test finishes in seconds instead of minutes.)
3. Mocks out the downloaded checkpoint (or uses a deliberately tiny
   model like `prajjwal1/bert-tiny`). (A checkpoint is just a saved model snapshot from Hugging Face's public library.)
4. Runs `train.train()` and verifies:

- The model directory exists after.
- `model_card.json` is present and parseable.
- `test_metrics.json` has all the expected keys.

Result: training tests run in ~30 seconds locally, full integration
with the real DistilBERT path, but small enough to fit in CI.

### Thread-safety test (`test_model_loader.py`)

```python
def test_concurrent_load_triggers_one_load(monkeypatch):
 clear_cache()
 call_count = 0
 real_load = model_loader._do_actual_load
 def counting_load():
 nonlocal call_count
 call_count += 1
 return real_load()
 monkeypatch.setattr(model_loader, "_do_actual_load", counting_load)

 threads = [Thread(target=load_model) for _ in range(10)]
 for t in threads: t.start()
 for t in threads: t.join()

 assert call_count == 1 # the lock did its job
```

This is the test that catches whether the double-checked locking
actually works. (Double-checked locking is a trick to make sure that even when several callers ask "is the model loaded yet" at the same time, only one of them actually does the loading — like one person volunteering to make the coffee while everyone else waits.) Without the lock, you'd see `call_count >= 2`.

### API tests (`test_api.py`)

Uses FastAPI's built-in `TestClient` (which is `httpx` under the hood):

```python
from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)

def test_predict_happy_path(monkeypatch):
 monkeypatch.setenv("ENABLE_MOCK_INFERENCE", "true")
 res = client.post("/predict", json={"topic": "AI", "text": "I love this!"})
 assert res.status_code == 200
 body = res.json()
 assert body["predictedLabel"] in {"supportive", "opposed", "neutral", "mixed", "unclear"}
 assert 0 <= body["confidence"] <= 1
```

No HTTP server actually started — TestClient calls the FastAPI app
directly via ASGI, in-process. (ASGI is the standard Python protocol for talking to a web app without going through a real network — think of it as the test walking up to the customer-service window through a back door instead of driving around to the front.) Fast, deterministic.

---

## How tests/ connects to everything else

```
src/* (production code)
 ▲
 │ imports + exercises
 │
tests/test_*.py
 │
 │ discovered + run by
 ▼
pytest (configured by pytest.ini)
 │
 │ coverage reported by
 ▼
pytest-cov → .coverage (gitignored) → terminal report
```

Production code never imports from `tests/`. The dependency arrow is
strictly one-way.

---

## "Where do I look when X happens"

| You want to fix...                  | Open...                                                                                              |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Specific test failing               | The file matching the source it tests (e.g. `test_predict_mock.py` for `inference/predict.py`)       |
| Coverage dropped below 100%         | `pytest --cov-report=term-missing` lists uncovered lines; usually a `coverage-*` file needs updating |
| Tests take too long                 | The training tests are the slowest; check `MAX_LENGTH` and `NUM_TRAIN_EPOCHS` overrides are set      |
| New source file needs tests         | Mirror the layout — new `src/X/Y.py` gets a `tests/test_Y.py`                                        |
| Test crashes with "model not found" | Set `ENABLE_MOCK_INFERENCE=true` in the test's env, or mock `is_model_available` to return false     |
| Want to test against real model     | Run `make train` first, then `pytest` will use the real model                                        |
