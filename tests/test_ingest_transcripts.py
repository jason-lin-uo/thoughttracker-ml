"""
Tests for scripts/ingest_transcripts.py.

Focus is the folderPath computation (audit §7 "POSTs absolute path"):
the script now sends a path RELATIVE to the configured transcripts root
by default, only falling back to absolute when asked or when the folder
isn't under the root. No HTTP is performed here — we exercise the pure
path/header helpers.
"""

import importlib.util
import pathlib
import sys


def _load_ingest():
    """Load scripts/ingest_transcripts.py as an importable module."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    src_path = repo_root / "scripts" / "ingest_transcripts.py"
    spec = importlib.util.spec_from_file_location("ingest_under_test", src_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ingest_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


ingest = _load_ingest()


def test_admin_headers_omits_pin_when_absent():
    """Behavioral: no PIN → only Content-Type; PIN present → X-Admin-Pin added."""
    assert ingest._admin_headers(None) == {"Content-Type": "application/json"}
    assert ingest._admin_headers("") == {"Content-Type": "application/json"}
    assert ingest._admin_headers("1234") == {
        "Content-Type": "application/json",
        "X-Admin-Pin": "1234",
    }


def test_resolve_folder_path_with_absolute_root(tmp_path: pathlib.Path):
    """Behavioral: an ABSOLUTE transcripts_root → absolute folderPath.

    A relative folderPath is only meaningful to the backend when the root
    itself is relative (resolved against the backend's cwd). With an
    absolute root we hand back the absolute folder so the backend doesn't
    misresolve a relative path against its own different cwd.
    """
    folder = tmp_path / "data" / "transcripts" / "huberman"
    folder.mkdir(parents=True)
    out = ingest.resolve_folder_path_for_backend(
        folder, tmp_path / "data" / "transcripts", "relative"
    )
    assert out == str(folder.resolve())


def test_resolve_folder_path_relative_root_yields_portable_path(tmp_path: pathlib.Path):
    """Behavioral: with a RELATIVE transcripts_root, output is portable.

    e.g. folder=<tmp>/data/transcripts/huberman, root=data/transcripts →
    'data/transcripts/huberman' (no host-absolute prefix).
    """
    folder = tmp_path / "data" / "transcripts" / "huberman"
    folder.mkdir(parents=True)
    # Make CWD the tmp tree so the relative root resolves correctly.
    import os

    prev = os.getcwd()
    try:
        os.chdir(tmp_path)
        out = ingest.resolve_folder_path_for_backend(
            folder, pathlib.Path("data/transcripts"), "relative"
        )
    finally:
        os.chdir(prev)
    assert out == str(pathlib.Path("data/transcripts/huberman"))


def test_resolve_folder_path_absolute_mode(tmp_path: pathlib.Path):
    """Behavioral: --path-mode absolute returns the resolved absolute path."""
    folder = tmp_path / "data" / "transcripts" / "mkbhd"
    folder.mkdir(parents=True)
    out = ingest.resolve_folder_path_for_backend(
        folder, pathlib.Path("data/transcripts"), "absolute"
    )
    assert out == str(folder.resolve())
    assert pathlib.Path(out).is_absolute()


def test_resolve_folder_path_outside_root_falls_back_to_absolute(tmp_path: pathlib.Path):
    """Behavioral: a folder NOT under the root → absolute path (never a wrong relative)."""
    folder = tmp_path / "somewhere" / "else"
    folder.mkdir(parents=True)
    import os

    prev = os.getcwd()
    try:
        os.chdir(tmp_path)
        out = ingest.resolve_folder_path_for_backend(
            folder, pathlib.Path("data/transcripts"), "relative"
        )
    finally:
        os.chdir(prev)
    assert out == str(folder.resolve())


def test_parse_args_defaults_to_relative_mode():
    """Behavioral: path-mode defaults to 'relative'; transcripts-root default set."""
    args = ingest._parse_args(
        [
            "--folder",
            "data/transcripts/x",
            "--creator-name",
            "X",
            "--creator-slug",
            "x",
        ]
    )
    assert args.path_mode == "relative"
    assert args.transcripts_root == pathlib.Path("data/transcripts")
