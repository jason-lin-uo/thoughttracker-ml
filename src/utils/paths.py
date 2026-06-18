"""
Filesystem path helpers — keep every script anchored at the project root.

Resolved at import time so every path is absolute and consistent
regardless of which CWD the caller is in. This file is the single
source of truth for "where does the data live?", "where do we
write models?", etc.
"""

from __future__ import annotations

from pathlib import Path


#: Project root = two levels above this file (``src/utils/paths.py``).
#: ``.resolve()`` makes the path absolute so symlinks + relative CWDs
#: don't confuse downstream callers.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Root of all dataset files.
DATA_DIR: Path = PROJECT_ROOT / "data"
#: Where raw creator/export datasets land before processing. Gitignored.
RAW_DATA_DIR: Path = DATA_DIR / "raw"
#: Where adapter scripts write the converted, training-ready CSVs.
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
#: Conventional local stance training CSV path. The clean repo does not ship a
#: sample corpus; set ``DATASET_PATH`` to the real local CSV when retraining.
DEFAULT_STANCE_DATA_PATH: Path = PROCESSED_DATA_DIR / "stance_training.csv"

#: Where ``train.py`` writes the saved model artifact tree.
MODELS_DIR: Path = PROJECT_ROOT / "models"
#: Top-level reports directory containing both figures and metrics.
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
#: PNGs of confusion matrices, loss curves, etc.
FIGURES_DIR: Path = REPORTS_DIR / "figures"
#: JSON files with classification reports, eval metrics, etc.
METRICS_DIR: Path = REPORTS_DIR / "metrics"


def ensure_dirs() -> None:
    """Create the entire output directory tree if it doesn't exist.

    Called once at the top of train.py / evaluate.py so the rest of
    the script can write files without worrying about missing
    parents. ``exist_ok=True`` makes this idempotent — re-runs don't
    error out on the second invocation.
    """
    for d in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
