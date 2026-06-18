# \_LEARN.md - `thoughttracker-ml/src/data/`

> Four files. Everything between "a CSV on disk" and "a tokenized
> dataset ready for the model."

---

## The story of this folder

Imagine you're teaching a class. Before students can learn, you need
to:

1. Get the textbook (the dataset on disk).
2. Make sure the textbook is correct - no missing pages, no typos in
   the answer key (validate the CSV).
3. Split the textbook into "stuff for class," "stuff for practice
   quizzes," and "stuff for the final exam" (train/val/test split).
4. Translate the words into the language the student understands
   (tokenize for the model).

That's what this folder does. It's the **preparation pipeline** that
turns raw labeled data into something a neural network can consume.

The folder has four files (`__init__.py`, `label_schema.py`,
`load_dataset.py`, `preprocess.py`), and they form a chain:

```
load_dataset.py ---> preprocess.py ---> (training / inference)
 (load + validate | split + to_model_inputs
 + build_input_text) | uses
 v
 label_schema.py
 (the canonical labels)
```

---

## File-by-file

### `__init__.py`

**What it is:** the package marker. Empty.

---

### `label_schema.py`

**What it is:** the **canonical list of stance labels** the classifier
returns, plus helpers to convert between string labels and integer
IDs. About 60 lines.

The labels:

- `supportive` - speaker explicitly endorses the topic.
- `opposed` - speaker explicitly rejects the topic.
- `neutral` - speaker describes the topic factually with no stance.
- `mixed` - speaker shows both supportive and opposed positions.
- `unclear` - stance is too ambiguous to label confidently.

Exports:

- `LABELS: list[str]` - the ordered list, defining label-to-int mapping.
- `LABEL_TO_ID: dict[str, int]` and `ID_TO_LABEL: dict[int, str]`.
- `is_valid_label(s)`, `label_id(s)`, `id_label(i)` helpers.

**Why centralize this:** the labels are referenced from at least 5
places - the dataset adapter, the training loop, evaluation, inference,
the FastAPI response schema. If they lived as string literals in each
file, adding a sixth label would mean a 5-file hunt. Here, it's one
file plus the dataset.

**The order matters.** The position in `LABELS` defines the integer
ID. Training data, model weights, and inference outputs all rely on
that mapping. **Never reorder the list** - it would invalidate every
saved model.

**Used by:** every other file in this folder, `src/training/train.py`,
`src/training/evaluate.py`, `src/inference/predict.py`, the FastAPI
response schemas.

---

### `load_dataset.py`

**What it is:** the **single seam** between "a CSV on disk" and the
training code. The whole codebase uses one function:
`load_stance_dataset(path)`.

What it does:

1. Reads the CSV with pandas.
2. Validates the schema - must have `id`, `topic`, `text`, `label`
   columns; extras are ignored.
3. Validates every label - must be in `label_schema.LABELS`. Catches
   typos like `Supprotive` at **load time** instead of letting them
   silently corrupt training.
4. Strips whitespace, drops blank rows.
5. Returns a clean DataFrame.

On failure: raises `DatasetValidationError` with a precise message -
"row 47: label 'Suppprotive' not in canonical set."

**Why the strict validator:** stance datasets are usually hand-curated
or scraped. A typo or a missing column corrupts training, and you only
notice when the model returns garbage. A loud error at load time is
infinitely better than a quiet degradation at deploy time.

**Used by:** `src/training/train.py`, `src/training/evaluate.py`. Also
by tests that need to verify dataset shape.

**Also exposes `build_input_text(topic, text)`** - the shared formatter that
combines a pair into `[TOPIC] <topic> [TEXT] <text>`, the exact string both
training (`preprocess.to_model_inputs`) and inference (`predict`) feed the
encoder. Centralizing it here is what keeps training- and inference-time
formatting identical.

---

### `preprocess.py`

**What it is:** the **train/val/test splitter** + **model-input
shaper**. Pure functions, no I/O.

Two main exports:

**`split_dataset(df, test_size, val_size, seed)`**

- Stratifies by label so each split has roughly the same label
  distribution as the source. (Stratified just means "evenly mixed" - if 20% of the textbook is about cats, then 20% of the practice quizzes and 20% of the final exam are also about cats.)
- Falls back to a non-stratified split if any label has fewer rows
  than the number of splits (rare, but happens with tiny local
  datasets).
- Returns `(train_df, val_df, test_df)`.
- Reproducible via `seed`.

**`to_model_inputs(df)`**

- Formats each row via `build_input_text` into `[TOPIC] <topic> [TEXT] <text>`
  - the model is **topic-aware**, so the topic is part of the input.
- Returns `(texts: list[str], label_ids: list[int])` - the shape the
  Hugging Face Trainer wants. (The Hugging Face Trainer is a popular training assistant - think of it as a teaching coach that handles all the standard training-loop chores so you don't have to write them yourself.)

**Why no torch import at the top:** ML libraries are slow to import
(torch + transformers can take 5+ seconds). Keeping this module
torch-free means importing it is fast, and unit tests can run in
milliseconds against tiny in-memory DataFrames.

**Why the `[TOPIC] ... [TEXT] ...` formatting matters:** stance depends
on what you're talking *about*. "I'm strongly opposed" means different
things depending on whether the topic is "AI" or "raisin cookies."
Without the topic in the input, the model would only see the bare text
and lose half the signal.

**The training/inference symmetry:** training and inference both call
`build_input_text`, so both feed the encoder `[TOPIC] <topic> [TEXT] <text>`.
If they ever diverged - say, inference put the topic on the end - the model
would perform terribly. Sharing the one formatter is what prevents that.

**Used by:** `src/training/train.py`, `src/training/evaluate.py`,
`src/inference/predict.py`.

---

## How data/ connects to everything else

```
DATASET_PATH -> local stance CSV (id, topic, text, label)
 |
 v
src/data/load_dataset.py -- load + validate (+ build_input_text)
 |
 v
src/data/preprocess.py -- split + to_model_inputs
 |
 +--> src/training/train.py
 |
 +--> src/inference/predict.py
 (uses build_input_text)
```

Everything that touches data goes through `load_dataset.py` first -
that's the validation gate.

---

## "Where do I look when X happens"

| You want to fix...                  | Open...                                                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------- |
| New stance label                    | `label_schema.py` - add to `LABELS`, retrain                                                             |
| CSV format changed                  | `load_dataset.py` - schema check at the top                                                              |
| Train/test split feels unbalanced   | `preprocess.py` - check stratification fallback isn't kicking in                                         |
| Different dataset source            | Write a new adapter alongside `load_dataset.py`, or load raw + map labels                                |
| Model trained but predicts nonsense | First check `build_input_text` - is inference using the same `[TOPIC] ... [TEXT] ...` shape as training |
| `DatasetValidationError` at load    | The error message has the row + the offending value - usually a typo in the labels column                |
