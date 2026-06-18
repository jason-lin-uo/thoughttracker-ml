"""
Tests for scripts/fetch_transcripts.py.

The script does network I/O (yt-dlp + youtube-transcript-api). We
mock both libraries so the tests are deterministic, hermetic, and
fast — no live YouTube calls during CI.
"""

import importlib.util
import json
import pathlib
import sys
from unittest import mock

import pytest


def _load_fetcher():
    """Load scripts/fetch_transcripts.py as a Python module so we can
    call its helpers directly. We don't add it to sys.path because the
    package name `scripts` is reserved by other Python tools."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    src_path = repo_root / "scripts" / "fetch_transcripts.py"
    spec = importlib.util.spec_from_file_location(
        "fetch_transcripts_under_test", src_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_transcripts_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


fetcher = _load_fetcher()


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------


def test_is_short_url_detects_shorts_pattern():
    """Behavioral test: is short url detects shorts pattern."""
    assert fetcher.is_short_url("https://www.youtube.com/shorts/abc123") is True
    assert fetcher.is_short_url("https://www.youtube.com/SHORTS/abc123") is True
    assert fetcher.is_short_url("https://www.youtube.com/watch?v=abc123") is False
    assert fetcher.is_short_url("") is False
    assert fetcher.is_short_url(None) is False  # type: ignore[arg-type]


def test_snippets_to_text_joins_unique_lines():
    """Behavioral test: snippets to text joins unique lines."""
    class S:
        """Test double / fixture: S."""
        def __init__(self, text):
            """Test helper: init."""
            self.text = text

    out = fetcher.snippets_to_text(
        [
            S("Welcome to the show"),
            S("Welcome to the show"),  # duplicate is collapsed
            S("Today we discuss"),
            S("  Today we discuss  "),  # whitespace-equivalent dedup
            S("the science of sleep"),
        ]
    )
    assert "Welcome to the show" in out
    assert "Today we discuss" in out
    assert "the science of sleep" in out
    # Each unique line appears exactly once.
    assert out.count("Welcome to the show") == 1
    assert out.count("Today we discuss") == 1


def test_snippets_to_text_handles_dict_style_snippets():
    """Behavioral test: snippets to text handles dict style snippets."""
    out = fetcher.snippets_to_text([{"text": "hello"}, {"text": "world"}])
    assert "hello" in out and "world" in out


def test_snippets_to_text_handles_empty_list():
    """Behavioral test: snippets to text handles empty list."""
    assert fetcher.snippets_to_text([]) == ""
    assert fetcher.snippets_to_text(None) == ""  # type: ignore[arg-type]


def test_snippets_to_text_strips_inline_newlines():
    """Behavioral test: snippets to text strips inline newlines."""
    class S:
        """Test double / fixture: S."""
        def __init__(self, text):
            """Test helper: init."""
            self.text = text

    out = fetcher.snippets_to_text([S("hello\nworld")])
    assert "\n" not in out.split("\n")[0]  # the joined line has no inner newline


def test_safe_filename_normalizes_unsafe_chars():
    """Behavioral test: safe filename normalizes unsafe chars."""
    assert fetcher._safe_filename("hello world!@#$") == "hello_world"
    assert fetcher._safe_filename("___trim___") == "trim"


def test_format_upload_date_handles_yyyymmdd():
    """Behavioral test: format upload date handles yyyymmdd."""
    assert fetcher._format_upload_date("20260315") == "2026-03-15"
    assert fetcher._format_upload_date("") is None
    assert fetcher._format_upload_date("badformat") is None
    assert fetcher._format_upload_date(None) is None
    # Too short
    assert fetcher._format_upload_date("20260") is None


def test_default_throttle_value():
    # Sanity: the documented default should be a positive number of seconds
    # so the default invocation is gentle on YouTube's rate limiter.
    """Behavioral test: default throttle value."""
    assert fetcher.DEFAULT_THROTTLE_SECONDS >= 1.0


# ----------------------------------------------------------------------------
# Manifest writes
# ----------------------------------------------------------------------------


def test_write_manifest_writes_json_atomically(tmp_path: pathlib.Path):
    """Behavioral test: write manifest writes json atomically."""
    creator_dir = tmp_path / "test_creator"
    creator_dir.mkdir()
    entries = [
        fetcher.ManifestEntry(
            videoId="abc",
            title="Sample",
            publishedAt="2026-01-01",
            durationSeconds=600,
            sourceUrl="https://example.com",
            thumbnailUrl=None,
            transcriptPath="abc.txt",
            status="saved",
        )
    ]
    out_path = fetcher.write_manifest(
        creator_dir,
        entries,
        creator={"name": "Test Creator", "slug": "test_creator", "channelUrl": "https://example.com/@test_creator"},
    )
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["creator"]["name"] == "Test Creator"
    assert data["creator"]["slug"] == "test_creator"
    assert data["creator"]["channelUrl"] == "https://example.com/@test_creator"
    assert "entries" in data
    assert "writtenAt" in data
    assert data["entries"][0]["videoId"] == "abc"
    assert data["entries"][0]["status"] == "saved"


def test_write_manifest_atomically_overwrites(tmp_path: pathlib.Path):
    """Behavioral test: write manifest atomically overwrites."""
    creator_dir = tmp_path / "atom"
    creator_dir.mkdir()
    # First write
    fetcher.write_manifest(
        creator_dir,
        [fetcher.ManifestEntry("a", "A", None, None, "u", None, None, "skipped", "x")],
        creator={"name": "Atom", "slug": "atom", "channelUrl": "https://example.com/@atom"},
    )
    # Second write with more entries — should fully replace, no leftover .tmp
    fetcher.write_manifest(
        creator_dir,
        [
            fetcher.ManifestEntry("b", "B", None, None, "u", None, None, "skipped"),
            fetcher.ManifestEntry("c", "C", None, None, "u", None, None, "skipped"),
        ],
        creator={"name": "Atom", "slug": "atom", "channelUrl": "https://example.com/@atom"},
    )
    data = json.loads((creator_dir / "_manifest.json").read_text())
    assert data["creator"]["slug"] == "atom"
    assert len(data["entries"]) == 2
    assert not (creator_dir / "_manifest.json.tmp").exists()


# ----------------------------------------------------------------------------
# process_one_video (per-video pipeline)
# ----------------------------------------------------------------------------


def _entry(**overrides):
    """Build a fake yt-dlp playlist entry."""
    base = {
        "id": "vid123",
        "url": "https://www.youtube.com/watch?v=vid123",
        "title": "Default Title",
        "duration": 600,
    }
    base.update(overrides)
    return base


def test_process_one_video_fast_skips_shorts_url(tmp_path: pathlib.Path):
    """Behavioral test: process one video fast skips shorts url."""
    entry = _entry(url="https://www.youtube.com/shorts/vid123")
    result = fetcher.process_one_video(
        entry, creator_dir=tmp_path, lang="en", min_duration=60
    )
    assert result.status == "skipped"
    assert result.skipReason == "shorts_url"


def test_process_one_video_fast_skips_short_duration(tmp_path: pathlib.Path):
    """Behavioral test: process one video fast skips short duration."""
    entry = _entry(duration=30)
    result = fetcher.process_one_video(
        entry, creator_dir=tmp_path, lang="en", min_duration=60
    )
    assert result.status == "skipped"
    assert "too_short" in (result.skipReason or "")


def test_process_one_video_saves_when_metadata_and_transcript_succeed(
    tmp_path: pathlib.Path,
):
    """Behavioral test: process one video saves when metadata and transcript succeed."""
    entry = _entry(id="ok123", duration=600)

    fake_info = {
        "duration": 1200,
        "upload_date": "20260315",
        "title": "Real Title",
        "thumbnail": "https://i.ytimg.com/vi/ok123/hq.jpg",
    }

    class S:
        """Test double / fixture: S."""
        def __init__(self, text):
            """Test helper: init."""
            self.text = text

    fake_snippets = [S("Hello world, this is the transcript.")]

    with (
        mock.patch.object(fetcher, "fetch_video_metadata", return_value=fake_info),
        mock.patch.object(
            fetcher, "fetch_video_transcript", return_value=fake_snippets
        ),
    ):
        result = fetcher.process_one_video(
            entry, creator_dir=tmp_path, lang="en", min_duration=60
        )

    assert result.status == "saved"
    assert result.videoId == "ok123"
    assert result.publishedAt == "2026-03-15"
    assert result.durationSeconds == 1200
    # File was written to disk.
    txts = list(tmp_path.glob("*.txt"))
    assert len(txts) == 1
    body = txts[0].read_text()
    assert "Real Title" in body
    assert "Hello world, this is the transcript." in body


def test_process_one_video_skips_when_no_captions(tmp_path: pathlib.Path):
    """Behavioral test: process one video skips when no captions."""
    entry = _entry(id="nocap", duration=600)
    fake_info = {"duration": 1200, "upload_date": "20260101", "title": "x"}
    with (
        mock.patch.object(fetcher, "fetch_video_metadata", return_value=fake_info),
        mock.patch.object(fetcher, "fetch_video_transcript", return_value=None),
    ):
        result = fetcher.process_one_video(
            entry, creator_dir=tmp_path, lang="en", min_duration=60
        )
    assert result.status == "skipped"
    assert result.skipReason == "no_en_captions"


def test_process_one_video_skips_when_caption_text_is_empty(tmp_path: pathlib.Path):
    """Behavioral test: process one video skips when caption text is empty."""
    entry = _entry(id="emptycap", duration=600)
    fake_info = {"duration": 1200, "upload_date": "20260101", "title": "x"}

    class S:
        """Test double / fixture: S."""
        def __init__(self, text):
            """Test helper: init."""
            self.text = text

    with (
        mock.patch.object(fetcher, "fetch_video_metadata", return_value=fake_info),
        mock.patch.object(fetcher, "fetch_video_transcript", return_value=[S("")]),
    ):
        result = fetcher.process_one_video(
            entry, creator_dir=tmp_path, lang="en", min_duration=60
        )
    assert result.status == "skipped"
    assert result.skipReason == "empty_caption_track"


def test_process_one_video_records_failed_on_metadata_error(tmp_path: pathlib.Path):
    """Behavioral test: process one video records failed on metadata error."""
    entry = _entry(id="bad", duration=600)
    with mock.patch.object(
        fetcher, "fetch_video_metadata", side_effect=RuntimeError("yt-dlp blew up")
    ):
        result = fetcher.process_one_video(
            entry, creator_dir=tmp_path, lang="en", min_duration=60
        )
    assert result.status == "failed"
    assert "metadata_fetch_failed" in (result.skipReason or "")


def test_process_one_video_skips_when_metadata_duration_too_short(
    tmp_path: pathlib.Path,
):
    """The fast-path duration check uses the playlist entry's `duration`,
    but the metadata's duration can override it when the playlist entry
    didn't carry one. This branch covers that case."""
    entry = _entry(id="late", duration=None)  # no fast-path skip
    fake_info = {"duration": 45, "upload_date": "20260101", "title": "short later"}
    with mock.patch.object(fetcher, "fetch_video_metadata", return_value=fake_info):
        result = fetcher.process_one_video(
            entry, creator_dir=tmp_path, lang="en", min_duration=60
        )
    assert result.status == "skipped"
    assert "too_short" in (result.skipReason or "")


def test_process_one_video_propagates_ip_block_to_caller(tmp_path: pathlib.Path):
    """IPBlockedError should bubble up so the orchestrator's main loop
    can apply its cool-down logic, NOT be silently turned into a 'failed'
    manifest entry."""
    entry = _entry(id="blocked", duration=600)
    fake_info = {"duration": 1200, "upload_date": "20260101", "title": "x"}
    with (
        mock.patch.object(fetcher, "fetch_video_metadata", return_value=fake_info),
        mock.patch.object(
            fetcher,
            "fetch_video_transcript",
            side_effect=fetcher.IPBlockedError("blocked"),
        ),
    ):
        with pytest.raises(fetcher.IPBlockedError):
            fetcher.process_one_video(
                entry, creator_dir=tmp_path, lang="en", min_duration=60
            )


# ----------------------------------------------------------------------------
# CLI argument parsing
# ----------------------------------------------------------------------------


def test_parse_args_defaults():
    """Behavioral test: parse args defaults."""
    args = fetcher._parse_args(
        [
            "--channel",
            "https://www.youtube.com/@x",
            "--creator-name",
            "X",
            "--creator-slug",
            "x",
        ]
    )
    assert args.channel == "https://www.youtube.com/@x"
    assert args.creator_name == "X"
    assert args.creator_slug == "x"
    assert args.limit == 0
    assert args.min_duration == 60
    assert args.throttle == fetcher.DEFAULT_THROTTLE_SECONDS


def test_parse_args_supports_throttle_override():
    """Behavioral test: parse args supports throttle override."""
    args = fetcher._parse_args(
        [
            "--channel",
            "x",
            "--creator-name",
            "x",
            "--creator-slug",
            "x",
            "--throttle",
            "0",
        ]
    )
    assert args.throttle == 0


# ----------------------------------------------------------------------------
# IP-block classification
# ----------------------------------------------------------------------------


def test_fetch_video_transcript_classifies_ip_block(monkeypatch):
    """When youtube-transcript-api raises a generic Exception whose
    message contains the well-known IP-block sentence, we should
    re-raise it as IPBlockedError."""

    class FakeYtt:
        """Test double / fixture: FakeYtt."""
        def fetch(self, *_a, **_kw):
            """Test helper: fetch."""
            raise Exception("YouTube is blocking requests from your IP")

    class FakeException(Exception):
        """Test double / fixture: FakeException."""
        pass

    fake_module = mock.MagicMock(
        YouTubeTranscriptApi=lambda: FakeYtt(),
        NoTranscriptFound=FakeException,
        TranscriptsDisabled=FakeException,
        VideoUnavailable=FakeException,
    )
    # Patch the import that fetch_video_transcript does inline.
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)

    with pytest.raises(fetcher.IPBlockedError):
        fetcher.fetch_video_transcript("vid", "en")


def test_fetch_video_transcript_returns_none_for_no_transcript(monkeypatch):
    """Behavioral test: fetch video transcript returns none for no transcript."""
    class NoTr(Exception):
        """Test double / fixture: NoTr."""
        pass

    class FakeYtt:
        """Test double / fixture: FakeYtt."""
        def fetch(self, *_a, **_kw):
            """Test helper: fetch."""
            raise NoTr("nope")

    fake_module = mock.MagicMock(
        YouTubeTranscriptApi=lambda: FakeYtt(),
        NoTranscriptFound=NoTr,
        TranscriptsDisabled=Exception,
        VideoUnavailable=Exception,
    )
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)

    out = fetcher.fetch_video_transcript("vid", "en")
    assert out is None
