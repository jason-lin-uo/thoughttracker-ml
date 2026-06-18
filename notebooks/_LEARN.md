# \_LEARN.md — `thoughttracker-ml/notebooks/`

> One file: `.gitkeep`. An intentionally empty placeholder folder.

---

## The story of this folder

This folder is **a chair pulled up to a table that nobody's sat at
yet**. It exists so that future-you (or a future contributor) has an
obvious place to drop exploratory Jupyter notebooks — the kind of
work that's part of ML practice but doesn't belong in `src/`.

Right now there are no actual notebooks here. The `.gitkeep` file is
a Git convention: Git won't track empty directories, so you put a
zero-byte file with a known name to force the folder to exist in the
repo.

---

## File-by-file

### `.gitkeep`

**What it is:** an empty file. Convention name for "this is a
placeholder so Git tracks the folder."

**Why it exists:** without it, the `notebooks/` folder wouldn't appear
in a fresh clone, and the next person to want to write a notebook
would have to create the folder themselves (or get a confusing
"directory not found" when trying to commit a notebook).

---

## What notebooks would go here

Notebooks are great for:

- **Exploratory data analysis** — open the ThoughtTracker gold-standard transcript labels CSV, compute label
  distribution, plot text-length histograms, eyeball mislabeled
  examples.
- **Error analysis on the trained model** — load the model, run it on
  examples that were misclassified, look for patterns ("the model
  fails on sarcasm; the model fails on questions; the model is biased
  toward `neutral` for short inputs").
- **Hyperparameter exploration** — sweep `learning_rate` × `batch_size`,
  plot the results.
- **Comparison studies** — DistilBERT vs RoBERTa vs deberta-v3-small;
  same data, same training loop, plot the F1 curves side by side.
- **Calibration analysis** — when the model says "confidence 0.8," is
  it right 80% of the time Make a reliability diagram.

These workflows produce charts and tables that aren't part of the
production pipeline but are part of **how an ML practitioner thinks
through a problem**. Keeping them in the repo means future-you can
look back and remember "oh right, I tried that and it didn't work."

---

## Naming convention (for when notebooks land here)

A common convention to follow:

```
NN_short-description.ipynb
```

- `NN` is a sequential number so notebooks sort chronologically.
- Description is kebab-case, brief.

Examples:

- `01_explore-topic-label-distribution.ipynb`
- `02_baseline-distilbert-vs-bert-tiny.ipynb`
- `03_error-analysis-mixed-label.ipynb`

---

## Notebook hygiene tips

If you do drop notebooks here:

1. **Strip output cells before committing.** Notebook diffs are
   miserable to review when they include base64-encoded image output.
   Run `jupyter nbconvert --clear-output --inplace *.ipynb` or use a
   pre-commit hook.
2. **Pin imports to absolute paths** at the top:
   `from src.config import config` — not random `sys.path.append`
   hacks.
3. **Don't put production code in notebooks.** If a notebook produces
   useful logic, **port it into `src/`** and re-import it. Notebooks
   are scratch space, not deployment artifacts.

---

## "Where do I look when X happens"

| You want to...                                       | Action                                                                                      |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Start exploratory analysis                           | Create a new `.ipynb` here following the naming convention                                  |
| Move notebook logic into the codebase                | Port to `src/data/`, `src/training/`, etc., then re-import in the notebook                  |
| Remove the placeholder once there are real notebooks | Delete `.gitkeep` — once there are any real files in the folder, Git tracks it without help |
