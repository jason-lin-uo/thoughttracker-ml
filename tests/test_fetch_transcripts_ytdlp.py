"""
Tests for scripts/fetch_transcripts_ytdlp.py.

Focus is the newly-configurable creator roster (audit §7 "Hardcoded
CREATORS") and the resume-aware ``--limit`` window. yt-dlp itself is
never invoked: ``run_ytdlp`` builds and inspects the argv it WOULD pass
to the subprocess, so these tests are hermetic and fast.
"""

import argparse
import importlib.util
import json
import pathlib
import sys
from unittest import mock


def _load_ytdlp():
    """Load scripts/fetch_transcripts_ytdlp.py as an importable module.

    Registered in ``sys.modules`` before execution so the module-level
    ``@dataclass`` can resolve its ``Optional[...]`` forward references.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    src_path = repo_root / "scripts" / "fetch_transcripts_ytdlp.py"
    spec = importlib.util.spec_from_file_location("ytdlp_under_test", src_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ytdlp_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


ytdlp = _load_ytdlp()


def _args(**overrides) -> argparse.Namespace:
    """Build an argparse.Namespace with sensible defaults for load_creators."""
    base = {
        "channel": None,
        "config": None,
        "creator_slug": None,
        "creator_name": None,
        "limit": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# ----------------------------------------------------------------------------
# Roster resolution (the "no longer hardcoded" fix)
# ----------------------------------------------------------------------------


def test_load_creators_defaults_to_builtin_roster():
    """Behavioral: with no --config/--channel we get the built-in roster."""
    creators = ytdlp.load_creators(_args())
    assert creators is ytdlp.DEFAULT_CREATORS
    assert len(creators) == 5
    # Back-compat alias preserved.
    assert ytdlp.CREATORS is ytdlp.DEFAULT_CREATORS


def test_load_creators_inline_channel_builds_single_creator():
    """Behavioral: --channel makes a one-entry roster; slug derived from URL."""
    creators = ytdlp.load_creators(
        _args(channel="https://www.youtube.com/@SomeCreator/videos", limit=7)
    )
    assert creators == [
        {
            "slug": "somecreator",
            "name": "somecreator",
            "channel": "https://www.youtube.com/@SomeCreator/videos",
            "limit": 7,
        }
    ]


def test_load_creators_inline_channel_respects_explicit_name_and_slug():
    """Behavioral: explicit --creator-name/--creator-slug override derivation."""
    creators = ytdlp.load_creators(
        _args(
            channel="https://www.youtube.com/@x",
            creator_slug="custom",
            creator_name="Custom Name",
        )
    )
    assert creators[0]["slug"] == "custom"
    assert creators[0]["name"] == "Custom Name"
    assert creators[0]["limit"] == 0  # limit=None → 0 (no cap)


def test_load_creators_from_config_list(tmp_path: pathlib.Path):
    """Behavioral: a bare JSON list config loads + normalizes entries."""
    cfg = tmp_path / "creators.json"
    cfg.write_text(
        json.dumps(
            [
                {"slug": "a", "name": "A", "channel": "https://yt/@a"},
                {"slug": "b", "name": "B", "channel": "https://yt/@b", "limit": "12"},
            ]
        )
    )
    creators = ytdlp.load_creators(_args(config=cfg))
    assert [c["slug"] for c in creators] == ["a", "b"]
    assert creators[0]["limit"] == 0
    assert creators[1]["limit"] == 12  # string coerced to int


def test_load_creators_from_config_creators_key(tmp_path: pathlib.Path):
    """Behavioral: a {"creators": [...]} wrapper object is also accepted."""
    cfg = tmp_path / "creators.json"
    cfg.write_text(
        json.dumps({"creators": [{"slug": "a", "name": "A", "channel": "https://yt/@a"}]})
    )
    creators = ytdlp.load_creators(_args(config=cfg))
    assert creators[0]["slug"] == "a"


def test_load_creators_from_config_rejects_missing_keys(tmp_path: pathlib.Path):
    """Behavioral: a config entry missing required keys fails fast."""
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps([{"slug": "a"}]))  # no name/channel
    with mock.patch.object(sys, "argv", ["x"]):
        try:
            ytdlp.load_creators(_args(config=cfg))
        except SystemExit as exc:
            assert "missing required key" in str(exc)
        else:  # pragma: no cover - the call must raise
            raise AssertionError("expected SystemExit")


def test_load_creators_from_config_rejects_empty(tmp_path: pathlib.Path):
    """Behavioral: an empty roster is rejected."""
    cfg = tmp_path / "empty.json"
    cfg.write_text(json.dumps([]))
    try:
        ytdlp.load_creators(_args(config=cfg))
    except SystemExit as exc:
        assert "non-empty list" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")


def test_load_creators_from_config_rejects_unreadable(tmp_path: pathlib.Path):
    """Behavioral: a non-existent / unparseable config file fails fast."""
    try:
        ytdlp.load_creators(_args(config=tmp_path / "does-not-exist.json"))
    except SystemExit as exc:
        assert "Could not read creators config" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")


def test_slug_from_channel_strips_tab_suffix():
    """Behavioral: trailing /videos etc. is skipped so we slug the handle."""
    assert ytdlp._slug_from_channel("https://www.youtube.com/@MyHandle/videos") == "myhandle"
    assert ytdlp._slug_from_channel("https://www.youtube.com/@MyHandle") == "myhandle"
    assert ytdlp._slug_from_channel("@BareHandle/streams") == "barehandle"
    # All-symbols tail can't form a slug → falls back to the literal "creator".
    assert ytdlp._slug_from_channel("https://example.com/---") == "creator"


# ----------------------------------------------------------------------------
# creator_set filtering against a roster
# ----------------------------------------------------------------------------


def test_creator_set_filters_roster_by_slug():
    """Behavioral: creator_set selects from the passed roster, not the global."""
    roster = [
        {"slug": "a", "name": "A", "channel": "u", "limit": 0},
        {"slug": "b", "name": "B", "channel": "u", "limit": 0},
    ]
    assert ytdlp.creator_set("all", roster) == roster
    assert [c["slug"] for c in ytdlp.creator_set("b", roster)] == ["b"]
    assert {c["slug"] for c in ytdlp.creator_set("a,b", roster)} == {"a", "b"}


def test_creator_set_unknown_slug_exits():
    """Behavioral: an unknown slug fails fast with a helpful message."""
    roster = [{"slug": "a", "name": "A", "channel": "u", "limit": 0}]
    for selector in ("nope", "a,nope"):
        try:
            ytdlp.creator_set(selector, roster)
        except SystemExit as exc:
            assert "nope" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected SystemExit")


# ----------------------------------------------------------------------------
# Resume-aware --limit window (alignment with fetch_transcripts.py)
# ----------------------------------------------------------------------------


def _captured_cmd(monkeypatch, tmp_path, creator, limit, seen_info_files=0):
    """Run run_ytdlp with the subprocess + conversion stubbed, returning argv.

    We fake ``next_unseen_playlist_item`` via real info.json files on disk
    and stub Popen so no subprocess actually launches; the command list
    handed to Popen is what we assert on.
    """
    captured = {}

    class FakeProc:
        returncode = 0

        def poll(self):
            return 0  # already exited → loop body is skipped

    def fake_popen(cmd, **_kw):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(ytdlp.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ytdlp, "convert_creator", lambda *a, **k: None)

    raw_dir = tmp_path / creator["slug"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    # next_unseen_playlist_item counts files matching YYYY-MM-DD_*.info.json,
    # so the names must carry a valid-looking date prefix to be counted.
    for i in range(seen_info_files):
        (raw_dir / f"2026-01-01_vid{i:04d}.info.json").write_text("{}")

    args = argparse.Namespace(
        caption_dir=tmp_path,
        transcript_dir=tmp_path / "transcripts",
        sleep_subtitles=0.0,
        sleep_requests=0.0,
        retries=1,
        min_duration=180,
        subprocess_timeout=0.0,
        limit=limit,
        rescan_existing=False,
    )
    ytdlp.run_ytdlp(creator, args)
    return captured["cmd"]


def test_limit_fresh_start_uses_plain_playlist_end(monkeypatch, tmp_path: pathlib.Path):
    """Behavioral: with no prior progress, --limit N → --playlist-end N."""
    creator = {"slug": "fresh", "name": "F", "channel": "https://yt/@f", "limit": 0}
    cmd = _captured_cmd(monkeypatch, tmp_path, creator, limit=30, seen_info_files=0)
    assert "--playlist-end" in cmd
    assert cmd[cmd.index("--playlist-end") + 1] == "30"
    assert "--playlist-start" not in cmd


def test_limit_on_resume_extends_window_past_seen(monkeypatch, tmp_path: pathlib.Path):
    """Behavioral: resuming after 50 seen + --limit 30 → start 51, end 80.

    The old behavior (`--playlist-end 30`) would fall before the resume
    point and fetch zero new videos; the fix makes --limit mean "30 MORE".
    """
    creator = {"slug": "resume", "name": "R", "channel": "https://yt/@r", "limit": 0}
    cmd = _captured_cmd(monkeypatch, tmp_path, creator, limit=30, seen_info_files=50)
    assert cmd[cmd.index("--playlist-start") + 1] == "51"
    assert cmd[cmd.index("--playlist-end") + 1] == "80"  # 51 - 1 + 30
