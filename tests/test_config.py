"""Tests for src/config.py helpers."""

import os
from unittest import mock


def test_bool_returns_default_when_env_var_missing():
    """Exercise the `raw is None → return default` branch of _env_bool."""
    from src.config import _env_bool

    if "TT_TEST_BOOL_FLAG" in os.environ:
        del os.environ["TT_TEST_BOOL_FLAG"]
    assert _env_bool("TT_TEST_BOOL_FLAG", True) is True
    assert _env_bool("TT_TEST_BOOL_FLAG", False) is False


def test_bool_accepts_truthy_strings():
    """Behavioral test: bool accepts truthy strings."""
    from src.config import _env_bool

    for v in ["1", "true", "True", "yes", "YES", "on"]:
        with mock.patch.dict(os.environ, {"TT_TEST_BOOL_FLAG": v}):
            assert _env_bool("TT_TEST_BOOL_FLAG", False) is True


def test_bool_rejects_other_strings():
    """Behavioral test: bool rejects other strings."""
    from src.config import _env_bool

    for v in ["0", "false", "no", "off", "anything-else"]:
        with mock.patch.dict(os.environ, {"TT_TEST_BOOL_FLAG": v}):
            assert _env_bool("TT_TEST_BOOL_FLAG", True) is False


def test_path_accepts_relative_and_absolute(tmp_path):
    """Behavioral test: path accepts relative and absolute."""
    from src.config import _env_path

    # Relative paths resolve against PROJECT_ROOT.
    with mock.patch.dict(os.environ, {"TT_TEST_PATH": "data/processed"}):
        out = _env_path("TT_TEST_PATH", tmp_path / "default")
        assert out.is_absolute()
        assert out.parts[-2:] == ("data", "processed")

    # Absolute paths are used as-is.
    with mock.patch.dict(os.environ, {"TT_TEST_PATH": str(tmp_path)}):
        out = _env_path("TT_TEST_PATH", tmp_path / "default")
        assert out == tmp_path

    # Empty / missing → default.
    for v in ["", None]:
        env = {} if v is None else {"TT_TEST_PATH": v}
        with mock.patch.dict(os.environ, env, clear=False):
            if "TT_TEST_PATH" in os.environ and v is None:
                del os.environ["TT_TEST_PATH"]
            out = _env_path("TT_TEST_PATH", tmp_path / "default")
            assert out == tmp_path / "default"
