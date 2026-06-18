#!/usr/bin/env python
"""
Fallback transcript fetcher that uses yt-dlp's subtitle downloader.

This exists for local setup moments where youtube-transcript-api is
IP-blocked but yt-dlp can still download the advertised auto-caption
files. It writes raw VTT files to a temporary caption cache and converts them
into ThoughtTracker-compatible .txt files under data/transcripts/<creator>/.
The cache can be deleted after conversion; the cleaned transcript .txt files
are the durable product artifact.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# Built-in real creator roster. This is only a default; the creator set is
# configurable at runtime, mirroring fetch_transcripts.py's CLI-driven
# approach, so the script is not hardcoded to one person's channels:
#   * ``--config creators.json`` loads an arbitrary roster from disk.
#   * ``--channel/--creator-name/--creator-slug`` defines a single ad-hoc
#     creator inline (no file needed), with an optional ``--limit``.
# The hardcoded list below is the fallback when neither is supplied.
DEFAULT_CREATORS = [
    {
        "slug": "huberman",
        "name": "Andrew Huberman",
        "channel": "https://www.youtube.com/@hubermanlab",
        "limit": 0,
    },
    {
        "slug": "allin",
        "name": "All In Podcast",
        "channel": "https://www.youtube.com/@allin",
        "limit": 0,
    },
    {
        "slug": "mkbhd",
        "name": "Marques Brownlee",
        "channel": "https://www.youtube.com/@mkbhd",
        "limit": 0,
    },
    {
        "slug": "delauer",
        "name": "Thomas DeLauer",
        "channel": "https://www.youtube.com/@ThomasDeLauerOfficial",
        "limit": 1500,
    },
    {
        "slug": "campea",
        "name": "John Campea",
        "channel": "https://www.youtube.com/playlist?list=PL6628E7149D3A7D56",
        "limit": 500,
    },
]

#: Backwards-compatible alias. Earlier callers/tests referenced ``CREATORS``
#: directly; it now points at the built-in default roster.
CREATORS = DEFAULT_CREATORS

#: Keys every creator entry must carry once normalized.
_REQUIRED_CREATOR_KEYS = ("slug", "name", "channel")


def load_creators(args: argparse.Namespace) -> list[dict]:
    """Resolve the roster of creators to fetch from the CLI args.

    Resolution order (first match wins):
      1. ``--channel`` given -> a single ad-hoc creator built from
         ``--channel`` / ``--creator-name`` / ``--creator-slug``
         (slug/name default off the channel handle when omitted).
      2. ``--config FILE`` given -> parse a JSON roster from disk. The file
         may be a bare list of creator objects, or an object with a
         top-level ``"creators"`` list. Each entry needs ``slug``, ``name``,
         and ``channel``; ``limit`` defaults to 0 (no cap).
      3. Otherwise -> the built-in :data:`DEFAULT_CREATORS` real creator roster.

    Raises ``SystemExit`` with a clear message on a malformed config so a
    typo fails fast instead of silently fetching the wrong channels.
    """
    if getattr(args, "channel", None):
        slug = args.creator_slug or _slug_from_channel(args.channel)
        return [
            {
                "slug": slug,
                "name": args.creator_name or slug,
                "channel": args.channel,
                "limit": args.limit if args.limit is not None else 0,
            }
        ]
    if getattr(args, "config", None):
        return _load_creators_from_file(args.config)
    return DEFAULT_CREATORS


def _slug_from_channel(channel: str) -> str:
    """Derive a filesystem-safe slug from a channel URL/handle.

    Pulls the ``@handle`` (or last meaningful path segment) and lowercases
    it, stripping anything that isn't a-z/0-9 so it's safe as a folder name.
    Trailing tab suffixes (``/videos``, ``/shorts``, ``/streams``) are
    skipped so ``.../@handle/videos`` slugs to ``handle``, not ``videos``.
    """
    parts = [seg for seg in channel.rstrip("/").split("/") if seg]
    tab_suffixes = {"videos", "shorts", "streams", "featured", "playlists"}
    while parts and parts[-1].lower() in tab_suffixes:
        parts.pop()
    tail = parts[-1].lstrip("@") if parts else ""
    cleaned = re.sub(r"[^a-z0-9]+", "", tail.lower())
    return cleaned or "creator"


def _load_creators_from_file(path: Path) -> list[dict]:
    """Parse + validate a creators roster JSON file.

    Accepts either a bare ``[ {...}, ... ]`` list or ``{"creators": [...]}``.
    Normalizes each entry (defaulting ``limit`` to 0) and validates the
    required keys, raising ``SystemExit`` on any structural problem.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read creators config {path}: {exc}") from exc
    entries = raw.get("creators") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise SystemExit(
            f"Creators config {path} must be a non-empty list (or a "
            f'{{"creators": [...]}} object).'
        )
    creators: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit(f"Creator entry must be an object, got: {entry!r}")
        missing = [key for key in _REQUIRED_CREATOR_KEYS if not entry.get(key)]
        if missing:
            raise SystemExit(
                f"Creator entry {entry!r} is missing required key(s): "
                f"{', '.join(missing)}"
            )
        creators.append(
            {
                "slug": entry["slug"],
                "name": entry["name"],
                "channel": entry["channel"],
                "limit": int(entry.get("limit") or 0),
            }
        )
    return creators


@dataclass
class ManifestEntry:
    """One per-video row in a creator's ``_manifest.json``.

    Mirrors the shape the backend's bulk-import endpoint expects, so the
    manifest this script writes can be POSTed verbatim.
    """

    videoId: str
    title: str
    publishedAt: Optional[str]
    durationSeconds: Optional[int]
    sourceUrl: str
    thumbnailUrl: Optional[str]
    transcriptPath: Optional[str]
    status: str
    skipReason: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the yt-dlp caption fetcher."""
    parser = argparse.ArgumentParser(description="Fetch YouTube captions with yt-dlp.")
    parser.add_argument(
        "--creator",
        default="all",
        help="Slug, comma-separated slugs, or 'all' to select from the roster.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to a JSON creators roster (a list of {slug,name,channel,"
            "limit} objects, or a {'creators': [...]} object). Overrides the "
            "built-in DEFAULT_CREATORS so the script isn't hardcoded."
        ),
    )
    parser.add_argument(
        "--channel",
        default=None,
        help=(
            "Fetch a single ad-hoc creator's channel/playlist URL without a "
            "config file; pairs with --creator-name / --creator-slug."
        ),
    )
    parser.add_argument(
        "--creator-name",
        default=None,
        help="Display name for the --channel creator (defaults to the slug).",
    )
    parser.add_argument(
        "--creator-slug",
        default=None,
        help="Folder slug for the --channel creator (derived from the URL if omitted).",
    )
    parser.add_argument("--caption-dir", type=Path, default=Path("data/yt-dlp-captions"))
    parser.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--sleep-subtitles", type=float, default=5.0)
    parser.add_argument("--sleep-requests", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument(
        "--min-duration",
        type=int,
        default=180,
        help="Skip videos at or below this many seconds. 180 excludes modern Shorts.",
    )
    parser.add_argument(
        "--subprocess-timeout",
        type=float,
        default=21600.0,
        help=(
            "Wall-clock cap (seconds) for a single yt-dlp subprocess. A hung "
            "yt-dlp (network stall, captcha wall) is killed once this elapses "
            "so the run can't block forever. Default 6 hours; 0 disables."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Override creator limit.")
    parser.add_argument("--no-download", action="store_true", help="Only convert existing VTT files.")
    parser.add_argument(
        "--rescan-existing",
        action="store_true",
        help="Revisit the playlist from the start without overwriting files; useful after changing subtitle options.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 ``...Z`` string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def creator_set(slug: str, roster: Optional[list[dict]] = None) -> list[dict]:
    """Filter ``roster`` by a ``--creator`` value to a list of config dicts.

    Accepts ``"all"`` (every entry in ``roster``), a single slug, or a
    comma-separated list of slugs. Raises ``SystemExit`` with a helpful
    message if any requested slug is unknown. ``roster`` defaults to the
    built-in :data:`DEFAULT_CREATORS` so older callers keep working; the
    CLI passes the runtime-resolved roster from :func:`load_creators`.
    """
    roster = DEFAULT_CREATORS if roster is None else roster
    if slug == "all":
        return roster
    if "," in slug:
        wanted = {part.strip() for part in slug.split(",") if part.strip()}
        selected = [creator for creator in roster if creator["slug"] in wanted]
        missing = wanted - {creator["slug"] for creator in selected}
        if missing:
            raise SystemExit(f"Unknown creator(s): {', '.join(sorted(missing))}")
        return selected
    selected = [creator for creator in roster if creator["slug"] == slug]
    if not selected:
        raise SystemExit(f"Unknown creator: {slug}")
    return selected


def channel_videos_url(channel: str) -> str:
    """Normalize a channel/playlist URL to the form yt-dlp should crawl.

    Playlist URLs are returned untouched; a channel URL is suffixed with
    ``/videos`` so yt-dlp walks the Videos tab (not Shorts / Live).
    """
    channel = channel.rstrip("/")
    if "youtube.com/playlist" in channel or "list=" in channel:
        return channel
    return channel if channel.endswith("/videos") else f"{channel}/videos"


def run_ytdlp(creator: dict, args: argparse.Namespace) -> int:
    """Run yt-dlp for one creator, converting captions as they land.

    Spawns yt-dlp as a subprocess writing VTT + info-JSON into the
    creator's caption cache, and periodically converts already-downloaded
    captions into ``.txt`` so progress is visible mid-run. A wall-clock
    ``--subprocess-timeout`` guards against a hung yt-dlp (the process is
    terminated, then killed, if it overruns). Returns the subprocess exit
    code (0 on a clean / timed-out-but-terminated run).
    """
    raw_dir = args.caption_dir / creator["slug"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "_archive.txt"
    log_path = raw_dir / "_yt-dlp.log"
    limit = creator["limit"] if args.limit is None else args.limit
    playlist_start = 1 if args.rescan_existing else next_unseen_playlist_item(raw_dir)

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--force-ipv4",
        "--ignore-errors",
        "--no-abort-on-error",
        "--skip-download",
        "--no-overwrites",
        "--write-subs",
        "--write-auto-subs",
        "--write-info-json",
        "--sub-langs",
        "en-orig,en",
        "--sub-format",
        "vtt",
        "--match-filter",
        f"duration > {args.min_duration} & !is_live",
        "--download-archive",
        str(archive),
        "--sleep-subtitles",
        str(args.sleep_subtitles),
        "--sleep-requests",
        str(args.sleep_requests),
        "--retries",
        str(args.retries),
        "--fragment-retries",
        str(args.retries),
        "--paths",
        str(raw_dir),
        "--output",
        "%(upload_date>%Y-%m-%d)s_%(id)s.%(ext)s",
    ]
    if playlist_start > 1:
        cmd.extend(["--playlist-start", str(playlist_start)])
    if limit and limit > 0:
        # Make --limit mean "this many MORE videos this run", aligning with
        # fetch_transcripts.py's newly-saved-count semantics. yt-dlp's
        # --playlist-end is an ABSOLUTE 1-based index, so on a resume from
        # item N we must extend the window to N + limit - 1; otherwise a
        # plain `--playlist-end {limit}` would fall *before* the resume
        # point and fetch nothing (e.g. resume at 50 with limit 30 →
        # end=30 < start=50 → zero new videos). When starting fresh
        # (playlist_start == 1) this collapses to the original behavior.
        playlist_end = playlist_start - 1 + limit
        cmd.extend(["--playlist-end", str(playlist_end)])
    cmd.append(channel_videos_url(creator["channel"]))

    timeout = getattr(args, "subprocess_timeout", 0.0) or 0.0
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] starting {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if timeout > 0 and (time.monotonic() - started) > timeout:
                # Hung yt-dlp: terminate, then escalate to kill if it
                # ignores SIGTERM, so the run can't block indefinitely.
                log.write(
                    f"[{utc_now()}] wall-clock timeout after {timeout}s; terminating\n"
                )
                log.flush()
                _terminate(proc)
                break
            convert_creator(creator, args.caption_dir, args.transcript_dir, args.min_duration)
            time.sleep(60)
        log.write(f"[{utc_now()}] exited {proc.returncode}\n")
    convert_creator(creator, args.caption_dir, args.transcript_dir, args.min_duration)
    # A None returncode means the child never reaped (e.g. it survived SIGKILL
    # plus the bounded waits in _terminate). Report a FAILURE, not exit 0 —
    # `proc.returncode or 0` previously masked that wedged/killed case as a
    # success. A real negative signal value (e.g. -9) is preserved as-is.
    rc = proc.returncode
    return rc if rc is not None else 1


def _terminate(proc: "subprocess.Popen") -> None:
    """Stop a subprocess gracefully, escalating to a hard kill.

    Sends SIGTERM and waits briefly; if the process is still alive it
    sends SIGKILL. Both waits are bounded so a wedged child can never
    hang the caller — the whole point of the wall-clock timeout.
    """
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def next_unseen_playlist_item(raw_dir: Path) -> int:
    """Best-effort resume hint for yt-dlp playlists.

    Subtitle-only runs do not reliably populate the download archive, so
    a restart otherwise walks through hundreds of already-fetched videos.
    We use per-video info JSON files as the durable progress marker and
    start after the count already seen.
    """
    seen = 0
    for path in raw_dir.glob("*.info.json"):
        if re.match(r"\d{4}-\d{2}-\d{2}_.+\.info\.json$", path.name):
            seen += 1
    return seen + 1


def strip_vtt_tags(line: str) -> str:
    """Strip VTT inline markup (timestamps, ``<c>`` cue tags, any ``<...>``)
    from one caption line and collapse runs of whitespace."""
    line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", line)
    line = re.sub(r"</?c(?:\.[^>]*)?>", "", line)
    line = re.sub(r"<[^>]+>", "", line)
    return re.sub(r"\s+", " ", line).strip()


def vtt_to_text(path: Path) -> str:
    """Convert a WebVTT caption file to de-duplicated plain text.

    Drops the header, ``-->`` cue-timing lines, and consecutive
    duplicate lines (auto-captions repeat heavily), returning the
    remaining cue text joined by newlines.
    """
    lines: list[str] = []
    previous = ""
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT" or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        clean = strip_vtt_tags(line)
        if not clean or clean == previous:
            continue
        lines.append(clean)
        previous = clean
    return "\n".join(lines)


def parse_caption_name(path: Path) -> tuple[str, str, str]:
    """Parse a caption filename into ``(date, video_id, base_stem)``.

    Example: ``2026-03-26_hnzrPKvRBD8.en-orig.vtt`` →
    ``("2026-03-26", "hnzrPKvRBD8", "2026-03-26_hnzrPKvRBD8")``. Raises
    ``ValueError`` on an unrecognized name so the caller can skip it.
    """
    match = re.match(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<id>[^.]+)\.(?:en-orig|en)\.vtt$", path.name)
    if not match:
        raise ValueError(f"Unexpected caption filename: {path.name}")
    base = f"{match.group('date')}_{match.group('id')}"
    return match.group("date"), match.group("id"), base


def load_info(raw_dir: Path, base: str) -> dict:
    """Load the ``<base>.info.json`` metadata sidecar, or ``{}`` if absent."""
    info_path = raw_dir / f"{base}.info.json"
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8", errors="ignore"))
    return {}


def fetch_title_via_oembed(url: str) -> Optional[str]:
    """Best-effort YouTube title via the public oEmbed endpoint (no API key).

    Fallback for when the yt-dlp info-json lacks a title (some fetch batches
    don't capture it) so a video never ends up with its bare id as the title.
    Returns ``None`` on any error.
    """
    try:
        from urllib.parse import urlencode
        from urllib.request import urlopen

        endpoint = "https://www.youtube.com/oembed?" + urlencode({"url": url, "format": "json"})
        with urlopen(endpoint, timeout=20) as resp:  # noqa: S310 - fixed youtube.com host
            data = json.loads(resp.read().decode("utf-8"))
        title = (data.get("title") or "").strip()
        return title or None
    except Exception:
        return None


def is_short_form(info: dict, min_duration: int) -> bool:
    """Heuristically decide whether a video is a Short / too-short clip.

    True when the duration is at/below ``min_duration``, the URL contains
    ``/shorts/``, or the title carries a ``#shorts`` hashtag.
    """
    duration = info.get("duration")
    if isinstance(duration, (int, float)) and duration <= min_duration:
        return True
    for key in ("webpage_url", "original_url", "url"):
        value = str(info.get(key) or "").lower()
        if "/shorts/" in value:
            return True
    title = str(info.get("title") or "").lower()
    return "#shorts" in title or "#short" in title


def move_excluded_txt(out_path: Path, transcript_root: Path, creator_slug: str) -> None:
    """Quarantine a short-form ``.txt`` into ``_excluded_short_form/<slug>/``.

    No-op if the file doesn't exist. If a same-named file already sits in
    the excluded folder, the source is simply deleted (the quarantine
    copy is authoritative).
    """
    if not out_path.exists():
        return
    excluded_dir = transcript_root / "_excluded_short_form" / creator_slug
    excluded_dir.mkdir(parents=True, exist_ok=True)
    destination = excluded_dir / out_path.name
    if destination.exists():
        out_path.unlink()
    else:
        out_path.replace(destination)


def cleanup_short_form_txt(raw_dir: Path, out_dir: Path, transcript_root: Path, creator_slug: str, min_duration: int) -> None:
    """Sweep already-written ``.txt`` files and quarantine any short-form ones.

    Re-checks each transcript against its info sidecar so videos that were
    only later revealed to be Shorts (or that predate the duration filter)
    get moved out of the active transcript folder.
    """
    for out_path in out_dir.glob("*.txt"):
        match = re.match(r"(?P<base>\d{4}-\d{2}-\d{2}_[^.]+)\.txt$", out_path.name)
        if not match:
            continue
        info = load_info(raw_dir, match.group("base"))
        if info and is_short_form(info, min_duration):
            move_excluded_txt(out_path, transcript_root, creator_slug)


def convert_creator(creator: dict, caption_root: Path, transcript_root: Path, min_duration: int) -> None:
    """Convert a creator's downloaded VTT captions into clean ``.txt`` files.

    Picks the best caption variant per video (preferring ``en-orig``),
    skips already-converted non-empty outputs, drops short-form videos,
    prepends a title + URL header, and rewrites the ``_manifest.json``.
    Idempotent: safe to call repeatedly while yt-dlp is still running.
    """
    raw_dir = caption_root / creator["slug"]
    out_dir = transcript_root / creator["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not raw_dir.exists():
        write_manifest(creator, out_dir)
        return
    cleanup_short_form_txt(raw_dir, out_dir, transcript_root, creator["slug"], min_duration)

    candidates: dict[str, Path] = {}
    for path in raw_dir.glob("*.vtt"):
        try:
            _, _, base = parse_caption_name(path)
        except ValueError:
            continue
        current = candidates.get(base)
        if current is None or path.name.endswith(".en-orig.vtt"):
            candidates[base] = path

    for base, caption_path in candidates.items():
        date, video_id, _ = parse_caption_name(caption_path)
        out_path = out_dir / f"{date}_{video_id}.txt"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        info = load_info(raw_dir, base)
        if info and is_short_form(info, min_duration):
            move_excluded_txt(out_path, transcript_root, creator["slug"])
            continue
        url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
        # Prefer the yt-dlp metadata title; if it's missing, recover it from the
        # oEmbed endpoint BEFORE falling back to the bare video id (an id-as-
        # title renders as gibberish in the UI — see the 545-row backfill).
        title = info.get("title") or fetch_title_via_oembed(url) or video_id
        text = vtt_to_text(caption_path)
        if text.strip():
            out_path.write_text(f"# {title}\n# {url}\n\n{text}\n", encoding="utf-8")

    write_manifest(creator, out_dir)


def manifest_entry_from_txt(path: Path, transcript_root: Path) -> ManifestEntry:
    """Build a ``ManifestEntry`` from a saved transcript ``.txt``.

    Recovers the video id + date from the filename and reads the title /
    URL from the first two ``# ...`` header lines, falling back to the id
    and a watch URL when those are missing or unreadable.
    """
    match = re.match(r"(?P<date>\d{4}-\d{2}-\d{2}|unknown)_(?P<id>[^.]+)\.txt$", path.name)
    video_id = match.group("id") if match else path.stem
    published = match.group("date") if match and match.group("date") != "unknown" else None
    title = video_id
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        first_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:2]
        if first_lines and first_lines[0].startswith("# "):
            title = first_lines[0][2:].strip() or title
        if len(first_lines) > 1 and first_lines[1].startswith("# "):
            url = first_lines[1][2:].strip() or url
    except OSError:
        pass
    return ManifestEntry(
        videoId=video_id,
        title=title,
        publishedAt=published,
        durationSeconds=None,
        sourceUrl=url,
        thumbnailUrl=None,
        transcriptPath=str(path.relative_to(transcript_root.parent)),
        status="saved",
        skipReason=None,
    )


def write_manifest(creator: dict, out_dir: Path) -> None:
    """Atomically (write-temp + rename) write the creator's ``_manifest.json``.

    Enumerates the non-underscore ``.txt`` files newest-first, building
    one entry each plus a creator metadata block.
    """
    transcript_root = out_dir.parent
    entries = [
        manifest_entry_from_txt(path, transcript_root)
        for path in sorted(out_dir.glob("*.txt"), reverse=True)
        if not path.name.startswith("_")
    ]
    payload = {
        "creator": {
            "name": creator["name"],
            "slug": creator["slug"],
            "channelUrl": creator["channel"],
        },
        "entries": [asdict(entry) for entry in entries],
        "writtenAt": utc_now(),
        "source": "yt-dlp-auto-captions",
    }
    tmp = out_dir / "_manifest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "_manifest.json")


def main() -> int:
    """CLI entry point: fetch + convert captions for the selected creators.

    For each creator, runs yt-dlp (unless ``--no-download``), converts the
    resulting captions to ``.txt``, and logs a per-creator txt/vtt count.
    The roster comes from --config / --channel / the built-in default (see
    :func:`load_creators`) rather than a single hardcoded list.
    Returns the max subprocess exit code seen across creators.
    """
    args = parse_args()
    args.caption_dir.mkdir(parents=True, exist_ok=True)
    args.transcript_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the roster first (config file / ad-hoc channel / default),
    # then narrow it by --creator. An inline --channel produces a one-entry
    # roster, so --creator stays a no-op ("all") in that mode.
    roster = load_creators(args)
    exit_code = 0
    for creator in creator_set(args.creator, roster):
        print(f"[{utc_now()}] {creator['slug']}: starting")
        if not args.no_download:
            result = run_ytdlp(creator, args)
            exit_code = max(exit_code, result)
        else:
            convert_creator(creator, args.caption_dir, args.transcript_dir, args.min_duration)
        txt_count = len(list((args.transcript_dir / creator["slug"]).glob("*.txt")))
        vtt_count = len(list((args.caption_dir / creator["slug"]).glob("*.vtt")))
        print(f"[{utc_now()}] {creator['slug']}: txt={txt_count} vtt={vtt_count}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
