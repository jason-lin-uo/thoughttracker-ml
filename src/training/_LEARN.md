# \_LEARN.md — `thoughttracker-ml/src/training/`

> Three files. Where the model **learns** from labeled data.

---

## The story of this folder

This is the **apprenticeship room** of the factory. The model arrives
knowing very little (it's a pretrained DistilBERT — it knows general
English, but not stance classification). It then sits in this room
and reads ~4,000 labeled examples ("here's a sentence, here's the
stance"), gradually adjusting its internal parameters until it can
predict the stance of unseen sentences.

The room has three workstations:

- `train.py` — runs the apprenticeship from scratch (loads data,
  trains, saves the trained model).
- `evaluate.py` — gives the trained model a test it hasn't seen
  before, scores it, generates a report card.
- `metrics.py` — the shared rubric used by both above (so the scoring
  is consistent).

Once the apprenticeship is done, the model graduates to `inference/`
and `api/`, which use it to answer real questions.

---

## File-by-file

### `__init__.py`

Empty package marker.

---

### `train.py`

**What it is:** the **main training script**. About 250 lines.
End-to-end:

1. Read the labeled dataset from `config.dataset_path`.
2. Split it into train / val / test (reproducible via the seed).
3. Tokenize using the base model's tokenizer (`AutoTokenizer`). (The tokenizer is the model's chopping board — it slices text into the bite-size pieces the model can chew. Each model has its own preferred chop, so we use the one that came with DistilBERT.)
4. Initialize the model:
   `AutoModelForSequenceClassification.from_pretrained(config.base_model,
num_labels=5)`.
5. Set up the Hugging Face `Trainer` with:

- `evaluation_strategy="epoch"` — eval on the val split after each
  epoch (one epoch = one full pass through the training set).
- `load_best_model_at_end=True` — keep the checkpoint with the
  best F1, not necessarily the last. (A checkpoint is a saved snapshot of the model at one moment, like a video game save file — we keep the best save, not the most recent.)
- `metric_for_best_model="f1_macro"` — what to optimize.

6. Train for `config.num_train_epochs` epochs (default 3).
7. Run final eval on the test split.
8. Save:

- The model + tokenizer to `config.model_dir` (via
  `save_pretrained` — the standard "save the whole model to disk" helper from Hugging Face).
- A `model_card.json` sidecar with version + base model + label
  schema (the FastAPI `/health` endpoint reads this).
- `reports/metrics/test_metrics.json` with the test numbers.
- `reports/figures/confusion_matrix.png` for the README.

**Why a HF Trainer:** it handles all the unsexy bits — batching,
gradient accumulation, mixed-precision (if GPU available),
checkpointing, eval-on-epoch. (In plain terms: the Trainer is the teaching coach that knows how to run a class — when to give quizzes, when to save snapshots, when to use shortcuts on a fast computer. We just hand it the textbook and the apprentice.) Writing that loop from scratch would be
~500 lines and full of subtle bugs.

**Why DistilBERT and not BERT/RoBERTa:** DistilBERT is 40% smaller,
60% faster, and retains ~95% of BERT's performance. For a portfolio
project running on a laptop CPU, that's the right trade.

**Why "intentionally light" hyperparameters:** batch size 8, 3 epochs,
256 max length, learning rate 5e-5. These defaults fit on a M-series
Mac CPU and complete the ThoughtTracker gold-standard transcript labels training in ~10 minutes. Beefier
hardware would tune up easily.

**Usage:**

```bash
python -m src.training.train
# or override:
NUM_TRAIN_EPOCHS=5 LEARNING_RATE=3e-5 python -m src.training.train
```

---

### `evaluate.py`

**What it is:** a script to **re-evaluate** an already-trained model
against the test split. About 100 lines.

**Why a separate script:** `train.py` already evaluates as its final
step, so why have this Three use cases:

1. "I tweaked the test-split seed — what do the new numbers look
   like without retraining"
2. "I swapped in a different held-out dataset to spot-check robustness."
3. "I want to verify that a saved model still performs as the metrics
   report claimed."

The script loads the saved model from `config.model_dir`, loads the
dataset, splits, and re-evaluates. Outputs land in distinct files
(`eval_metrics.json`, `confusion_matrix_eval.png`) so the training-run
artifacts aren't overwritten.

**Loading is per-example, not batched.** That's intentional — the
code reads top-to-bottom and the perf cost is negligible against a
few-hundred-row test split. If you ever need to re-eval against a
huge set, batch it.

---

### `metrics.py`

**What it is:** the **shared metric calculator**. Used by both
`train.py` (to compute metrics during training) and `evaluate.py`
(for re-eval).

**What it computes:**

- `accuracy` — fraction correct.
- `f1_macro` — unweighted mean F1 across labels. **The metric we
  optimize for.** (F1 is a balanced score that gives equal weight to "did we catch all the right ones" and "did we avoid false alarms." Macro F1 averages the score across every label evenly — like grading a student equally on every chapter, not just the popular ones.)
- `f1_weighted` — weighted by support per label. (Weighted F1 instead gives more credit for the popular chapters; a model can ace this while bombing the rare labels.)
- Per-label `precision`, `recall`, `f1` (a sklearn
  `classification_report` dict).
- The full `confusion_matrix` as a 2D array. (A confusion matrix is a "who-got-mistaken-for-whom" grid: each row is the true label, each column is what the model guessed.)

**Why two flavors of F1:** macro and weighted F1 can disagree wildly
on imbalanced data. ThoughtTracker gold-standard transcript labels has way more `opposed` than `mixed`,
so weighted F1 looks good as long as the model handles the common
labels — but it can hide that the model is awful on `mixed`. Macro F1
exposes that, which is why it's our headline metric.

**Why sklearn for the math:** sklearn is the canonical implementation
of these metrics. Using it means our numbers match what a researcher
would compute in a notebook from our JSON outputs. No DIY math, no
silent drift.

**Used by:** `train.py` (in `compute_metrics`, which HF Trainer calls
during eval), `evaluate.py` (final scoring).

---

## How a training run progresses

```
[config.dataset_path]
 │
 ▼ load_stance_dataset
[DataFrame: id, topic, text, label]
 │
 ▼ split_dataset (stratified, seeded)
[train_df, val_df, test_df]
 │
 ▼ to_model_inputs
[texts: list[str], label_ids: list[int]] (each split)
 │
 ▼ AutoTokenizer
[tokenized HF Dataset] (each split)
 │
 ▼ Trainer
[for epoch in 3:
 - train on train split
 - eval on val split with metrics.py
 - save checkpoint if best]
 │
 ▼ trainer.evaluate(test_dataset)
[final test metrics]
 │
 ├──▶ models/stance-classifier/ (model weights + tokenizer)
 ├──▶ models/stance-classifier/model_card.json
 ├──▶ reports/metrics/test_metrics.json
 └──▶ reports/figures/confusion_matrix.png
```

A successful run prints something like:

```
Epoch 1: loss=1.32 f1_macro=0.41
Epoch 2: loss=0.78 f1_macro=0.63
Epoch 3: loss=0.51 f1_macro=0.71
Final test: accuracy=0.73 f1_macro=0.72 f1_weighted=0.74
```

---

## What the model card is

`model_card.json` is a small sidecar saved alongside the weights. It
documents what the model is, what data it was trained on, and what
labels it returns. The FastAPI `/health` endpoint serves this so any
client can verify:

- Which model version is running.
- Which label schema it returns.
- Whether the artifact is even loaded.

This pattern is borrowed from the Hugging Face Model Card convention
— it's how trained-model artifacts get a passport in 2024+ ML. (Think of a model card as a nutrition label glued to the model: what it was trained on, what it's good at, what to watch out for.)

---

## "Where do I look when X happens"

| You want to fix...          | Open...                                                                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Training is slow            | `train.py` — bump `train_batch_size` if you have memory; reduce `max_length` if not                                                     |
| Loss isn't going down       | `train.py` — try a different `learning_rate` (5e-5 → 3e-5 → 1e-5); check the data isn't corrupted                                       |
| Test accuracy is bad        | Look at `confusion_matrix.png` first — which labels are confused with which Then either get more data or change the schema             |
| Need different metrics      | `metrics.py` — add to `compute_classification_metrics`                                                                                  |
| Model is too big to ship    | `train.py` — base_model="prajjwal1/bert-tiny" gets you a 17MB model at the cost of ~5pp accuracy                                        |
| Test metrics report missing | Check `reports/metrics/` and `reports/figures/` — `ensure_dirs()` should have created them, but maybe the script crashed before writing |
