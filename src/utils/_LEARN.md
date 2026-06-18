# \_LEARN.md - `thoughttracker-ml/src/utils/`

> Two tiny modules. The boring, necessary plumbing every script in the
> repo touches.

---

## The story of this folder

If `src/` is a small factory, `utils/` is the **janitor's closet**.
What lives in here

- The map of where stuff is (paths)
- The clipboard for writing things down (logging)

Nothing here does ML. Nothing here is exciting. But if any of it
breaks, *every* script breaks, because all of them call into here at
import time.

The pattern: keep utils tiny, keep them stable, keep them
domain-agnostic. If you find yourself adding ML-specific logic to
`utils/`, it belongs somewhere else.

---

## File-by-file

### `__init__.py`

**What it is:** the package marker. Empty.

**Why it exists:** Python needs it for `from src.utils.paths import
PROJECT_ROOT` to work.

---

### `paths.py`

**What it is:** a list of `Path` constants - every folder anywhere in
the repo gets a constant here so other code doesn't have to know the
folder layout. About 50 lines.

The constants:

| Constant                   | Resolves to                                             |
| -------------------------- | ------------------------------------------------------- |
| `PROJECT_ROOT`             | The repo root (two levels above this file)              |
| `DATA_DIR`                 | `<root>/data`                                           |
| `DEFAULT_STANCE_DATA_PATH` | `<root>/data/processed/stance_training.csv`             |
| `RAW_DATA_DIR`             | `<root>/data/raw` - where raw creator/export data lands |
| `PROCESSED_DATA_DIR`       | `<root>/data/processed` - after adapter scripts run     |
| `MODELS_DIR`               | `<root>/models` - trained-model artifact tree           |
| `REPORTS_DIR`              | `<root>/reports`                                        |
| `FIGURES_DIR`              | `<root>/reports/figures` - PNG plots                    |
| `METRICS_DIR`              | `<root>/reports/metrics` - JSON metrics                 |

Plus one function:

- `ensure_dirs()` - creates every output directory if it doesn't
  already exist. Called once at the top of `train.py` /
  `evaluate.py` so the rest of the script can write files
  without missing-parent errors.

**Why a separate file for this:** every script needed to know
"where's the dataset" or "where do I write metrics" Without
centralization, every script would have its own hardcoded path string

- change one and forget to change the others, you've got a bug.

**Why `Path` objects, not strings:** `Path` objects from `pathlib` are
the modern Python way (think of them as smart address labels instead of plain text - they know how to do path math). They support `.resolve()` (turn relative into
absolute), `.exists()`, the `/` operator (`DATA_DIR / "processed"`),
and they're cross-platform-aware (don't break on Windows-style backslashes).

**The `.resolve()` discipline:** `PROJECT_ROOT = Path(__file__).resolve().parents[2]`
gives an absolute path no matter where the caller's `CWD` is. So
running `python -m src.training.train` from the repo root or from a
subfolder both work.

**Used by:** `config.py` (which uses it to resolve env-var paths
relative to `PROJECT_ROOT`), `train.py`, `evaluate.py`, and any test
that needs to write output files.

---

### `logging.py`

**What it is:** a tiny wrapper around Python's stdlib `logging`. One
function: `get_logger(name)`.

**Why it exists:**

- Every script wants the same log format.
- The format is `2026-05-23T12:34:56 INFO src.training.train :: Epoch 1 of 3`
  - the `::` separator makes grepping logs easy (no false matches against
    other colons).
- `LOG_LEVEL` env var controls the threshold (`INFO` by default,
  `DEBUG` for verbose runs).
- **Idempotent** - calling `get_logger("foo")` twice doesn't
  double-attach handlers (which would cause each line to print twice).
  (Idempotent just means "safe to call more than once" - the second call is a no-op, like flipping an already-on switch.)
  This is a real bug Python's stdlib hits if you naively call
  `basicConfig` from multiple scripts.

**Why not just use `print()`:**

- `print()` always goes to stdout regardless of severity.
- No timestamps.
- No way to filter (DEBUG/INFO/WARNING/ERROR).
- Doesn't show which module the message came from.
- Harder to redirect or capture in production.

Logging is the small-but-real difference between a script and a
production service.

**Used by:** every script. The pattern is:

```python
from src.utils.logging import get_logger
log = get_logger(__name__)
log.info("Training started; epochs=%d", config.num_train_epochs)
```

---

## How utils/ connects to everything else

```
src/utils/paths.py <-- imported by -- src/config.py
 (which uses _path() to resolve
 env-var paths against PROJECT_ROOT)

src/utils/paths.py <-- imported by -- src/training/train.py
 src/training/evaluate.py
 (which call ensure_dirs())

src/utils/logging.py <-- imported by -- every CLI / module that logs
```

`utils/` is the foundation. Nothing in `utils/` imports from any other
`src/*` package - it can't, because `config.py` and the rest _depend
on_ `utils/`. A circular import here would be a disaster.

---

## What this folder deliberately doesn't contain

- **Math or ML helpers** - those go in `src/training/metrics.py` or
  similar.
- **Dataset-specific utilities** - those go in `src/data/`.
- **HTTP helpers** - those go in `src/api/`.
- **Anything that depends on torch, transformers, or sklearn** - utils
  should be importable without the heavy ML stack loaded (so tests
  can run quickly).

---

## "Where do I look when X happens"

| You want to fix...                   | Open...                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Output landing in the wrong place    | `paths.py`                                                                                             |
| Need a new shared directory constant | Add to `paths.py` and include in `ensure_dirs()`                                                       |
| Logs format is wrong                 | `logging.py` - the `Formatter(fmt=...)` line                                                           |
| Logs print twice                     | The idempotency check in `get_logger` - `if logger.handlers: return logger` should already handle this |
| Want DEBUG-level logs                | `LOG_LEVEL=DEBUG python -m src.training.train`                                                         |
