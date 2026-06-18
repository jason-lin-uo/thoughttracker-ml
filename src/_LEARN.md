# \_LEARN.md - `thoughttracker-ml/src/`

> Where the actual Python lives. One config file, five subfolders, one
> single underlying job: take a (topic, text) pair, return a stance
> label.

---

## The story of this folder

If the repo is a small factory, this folder is the **production
floor**. Each subfolder is one workstation:

- `data/` - receiving dock and prep area (load datasets, preprocess,
  tokenize)
- `training/` - apprenticeship room (where the model learns from
  examples)
- `inference/` - production line (where the trained model classifies
  new examples)
- `api/` - customer service window (HTTP requests come in, predictions
  go out)
- `utils/` - janitor closet (logging, file paths - the unglamorous
  bits everyone needs)

Plus one file sitting at the top floor - `config.py` - which is the
**dispatcher's clipboard**. Everything else asks the clipboard:
"where's the data", "what's the batch size", "what port should I
listen on"

The flow of work, day-to-day:

- During **training**: `data/` produces a tokenized dataset -> `training/`
  fine-tunes the model -> result is a folder of weights.
- During **inference**: `api/` receives a request -> calls into
  `inference/` -> which uses `data/` for preprocessing and loads the
  weights produced by `training/`.

---

## File-by-file (root of `src/`)

### `__init__.py`

**What it is:** the file that turns `src/` into a Python package.
Almost empty - has a one-line version string.

**Why it exists:** without it, `import src.api.main` wouldn't work.
Python's import system needs the marker file.

**Used by:** every other module that imports from `src.*`.

---

### `config.py`

**What it is:** the **single source of truth for config**. Reads env
vars (or `.env` file via `python-dotenv`), wraps them in a frozen
dataclass (think of it as a sealed clipboard - once written, the values can't be changed), and exposes a singleton `config` for the whole codebase.

Fields it manages:

- **Model artifacts:** `model_dir`, `model_version`, `base_model`
  (which Hugging Face checkpoint to fine-tune - a checkpoint is just a saved snapshot of a pre-trained model, downloaded from Hugging Face, the public model library).
- **Training hyperparameters:** `max_length`, `train_batch_size`,
  `eval_batch_size`, `num_train_epochs`, `learning_rate`,
  `weight_decay`, `seed`. (Hyperparameters are the knobs you set before training starts - like an oven's temperature and timer. Learning rate is how big a step the model takes when adjusting; weight decay nudges it to stay humble and avoid memorizing.) Defaults chosen to fit on a laptop CPU.
- **Dataset:** `dataset_path`, `test_size`, `val_size`.
- **API:** `api_host`, `api_port`.
  don't have a trained model on disk.

**Why a dataclass and not just module-level constants**

- `frozen=True` makes it immutable - accidental writes raise an error. (In plain terms: once we lock the clipboard, nobody can scribble new values on it during a run.)
- Single object passed around explicitly is easier to replace with test fixtures.
- Type annotations give every field a known shape (no
  `os.environ.get("X")` returning `None` surprise).

**Why path resolution is done here:** the helper `_path()` resolves
env-var paths relative to `PROJECT_ROOT` (defined in `utils/paths.py`).
That way `DATASET_PATH=data/mine.csv` and
`DATASET_PATH=/abs/path.csv` both work - the user doesn't have to
remember to use absolute paths.

**Used by:** every other file in `src/`. The pattern is
`from src.config import config` at the top, then `config.batch_size`
where needed.

---

## Subfolders

Each has its own `_LEARN.md`:

| Folder       | What it does                                     |
| ------------ | ------------------------------------------------ |
| `data/`      | Load datasets, preprocess text, normalize labels |
| `training/`  | Fine-tune DistilBERT, evaluate, plot metrics     |
| `inference/` | Load the trained model, make predictions         |
| `api/`       | The FastAPI HTTP server                          |
| `utils/`     | Path constants, logging setup                    |

---

## How a workflow flows through `src/`

### Training workflow (run once per dataset)

```
DATASET_PATH local stance CSV
 |
 v
src/data/load_dataset.py -- splits into train/val/test
 |
 v
src/data/preprocess.py -- formats "[TOPIC] ... [TEXT] ..." (build_input_text), tokenizes
 |
 v
src/training/train.py -- loads distilbert-base-uncased,
 fine-tunes for 3 epochs,
 saves to models/stance-classifier/
 |
 v
src/training/evaluate.py -- runs test split, dumps metrics
 + confusion matrix to reports/
```

### Inference workflow (per request)

```
HTTP request:
 POST /predict
 { topic: "AI safety", text: "I think AI..." }
 |
 v
src/api/main.py -- FastAPI route, validates with Pydantic (the bouncer that checks the request shape)
 |
 v
src/inference/predict.py -- orchestrates the prediction
 |
 +--> src/inference/model_loader.py
 | (singleton-loads from models/ via config.model_dir)
 |
 +--> src/data/preprocess.py
 | (same tokenization as training  -  chopping text into pieces the model speaks)
 |
 +--> torch forward pass -> softmax -> argmax (model runs the text -> scores turn into clean percentages -> pick the winner)
 |
 v
src/data/label_schema.py -- maps int -> "supportive" / "opposed" / etc.
 |
 v
HTTP response:
 { predictedLabel: "supportive", confidence: 0.83, labelScores: {...}, modelVersion: "stance-classifier-v1" }
```

The crucial property: **input formatting is shared** between training and
inference - both call `build_input_text` (in `src/data/load_dataset.py`),
which produces `[TOPIC] <topic> [TEXT] <text>`. If training used that format
and inference used a different one, the model would get nonsense. That's why
the formatter lives in `data/` (shared) and not duplicated.

---

## "Where do I look when X happens"

| You want to fix...                     | Open...                                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------------------- |
| Config value isn't taking effect       | `config.py` - check env var name + cast                                                 |
| Default paths are wrong                | `utils/paths.py` (constants), `config.py` (env-var resolution)                          |
| Whole repo broke during import         | Probably `config.py` or `__init__.py` - `config` is imported transitively by everything |
| New hyperparameter needs to be tunable | Add to `Config` dataclass with env-var fallback, use it from the training module        |
