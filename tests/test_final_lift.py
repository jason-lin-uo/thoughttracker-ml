"""Final coverage lift for defensive ML utility branches."""

from __future__ import annotations


def test_ensure_dirs_creates_all_directories(tmp_path, monkeypatch):
    """Point path constants at a temp tree and confirm every output dir exists."""
    from src.utils import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "RAW_DATA_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DATA_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(paths, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(paths, "METRICS_DIR", tmp_path / "reports" / "metrics")

    paths.ensure_dirs()

    for sub in [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "reports/figures",
        "reports/metrics",
    ]:
        assert (tmp_path / sub).is_dir()


def test_train_main_keeps_lazy_transformer_entrypoint():
    """The training module should keep heavy transformer imports behind main."""
    from src.training import train as train_mod

    assert callable(train_mod.main)
    assert callable(train_mod._train_with_transformers)
