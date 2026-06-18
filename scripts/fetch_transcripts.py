#!/usr/bin/env python
"""
fetch_transcripts.py — bulk-download YouTube auto-caption transcripts
for a set of creators, filter out shorts, and write one .txt file per
video into a structured `data/transcripts/<creator>/` tree.

Why this exists
---------------
The ThoughtTracker app analyzes long-form YouTube content. Pulling
transcripts manually for hundreds of videos isn't viable, and the
official YouTube Data API doesn't expose captions for non-owned channels.
yt-dlp's auto-caption extraction is the pragmatic path: it parses the
same caption tracks the YouTube UI serves to viewers.

What it does
------------
1. Resolve a channel URL or @handle to its full uploads playlist.
2. Iterate every video in the uploads list.
3. For each video:
   - Skip if duration < `MIN_DURATION_SECONDS` (defaults to 60s — kills shorts).
   - Skip if the canonical URL contains `/shorts/` (belt-and-suspenders).
   - Skip if no English auto-caption is available.
   - Otherwise extract the auto-caption text, clean it (drop timestamps,
     collapse blank lines), and write to
     `<out_dir>/<creator_slug>/<YYYY-MM-DD>_<videoId>.txt`.
4. Maintain a `_manifest.json` per creator with one entry per video
   (videoId, title, publishedAt, durationSeconds, sourceUrl, status,
   skipReason) so the downstream bulk-import endpoint has structured
   metadata without re-querying YouTube.

The script is designed to be re-runnable: it skips any video whose
output file already exists, so a partial run can be resumed by running
the same command again.

Usage
-----
    python scripts/fetch_transcripts.py \\
        --channel https://www.youtube.com/@hubermanlab \\
        --creator-name "Andrew Huberman" \\
        --creator-slug huberman \\
        --out-dir data/transcripts \\
        --limit 0   # 0 = no cap; otherwise N most-recent uploads

To fetch all 5 demo creators, see `fetch_demo_creators.sh` (or just call
this script in a loop).

Implementation notes
--------------------
- We use TWO libraries, each for what it does best:
   * **yt-dlp** for resolving a channel URL to the list of video IDs +
     per-video metadata (title, upload date, duration, thumbnail).
   * **youtube-transcript-api** for the actual transcript text. We tried
     yt-dlp's auto-caption extraction first but YouTube now blocks
     anonymous caption access without a "PO token" (yt-dlp issue
     #12482). `youtube-transcript-api` uses the same caption endpoint
     the YouTube web UI uses for its transcript panel and works without
     a PO token.
- Rate limiting: both libraries handle retries internally. For
  persistent failures we sleep + re-try at the per-video level (up to
  `MAX_RETRIES_PER_VIDEO` attempts).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# yt-dlp is imported lazily inside main() so this module can be imported
# for unit tests without the dependency present.

logger = logging.getLogger("fetch_transcripts")
MIN_DURATION_SECONDS = 60
DEFAULT_LANG = "en"
MAX_RETRIES_PER_VIDEO = 3
# Throttle between successful per-video fetches. YouTube's anonymous
# caption endpoint will IP-ban after ~30 sustained requests without a
# pause; spacing requests by ~3-5s gets us through full channel pulls
# without hitting the rate limiter. Configurable via --throttle.
DEFAULT_THROTTLE_SECONDS = 3.0
# On a confirmed IP-block error, sleep this long before resuming. The
# block usually clears in 5-15 minutes for hobby-scale request volume.
IP_BLOCK_COOLDOWN_SECONDS = 600.0


@dataclass
class ManifestEntry:
    """One row in the per-creator manifest. Mirrors the shape the bulk-
    import endpoint expects so we can POST the manifest verbatim."""

    videoId: str
    title: str
    publishedAt: Optional[str]
    durationSeconds: Optional[int]
    sourceUrl: str
    thumbnailUrl: Optional[str]
    transcriptPath: Optional[str]
    status: str  # "saved" | "skipped" | "failed"
    skipReason: Optional[str] = None


def _parse_args(argv=None):
    """Parse the CLI flags. Kept narrow on purpose: one channel per
    invocation so a single failure doesn't cascade across creators."""
    p = argparse.ArgumentParser(description="Fetch YouTube transcripts for one channel.")
    p.add_argument("--channel", required=True, help="Channel URL or @handle.")
    p.add_argument("--creator-name", required=True, help="Display name (e.g. 'Andrew Huberman').")
    p.add_argument(
        "--creator-slug",
        required=True,
        help="URL-safe slug used as the per-creator output folder.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/transcripts"),
        help="Root output directory; per-creator subfolders are created under here.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only fetch the N most-recent uploads. 0 means everything.",
    )
    p.add_argument(
        "--min-duration",
        type=int,
        default=MIN_DURATION_SECONDS,
        help=f"Skip videos shorter than this many seconds (default {MIN_DURATION_SECONDS}).",
    )
    p.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help=f"Subtitle language code (default '{DEFAULT_LANG}').",
    )
    p.add_argument(
        "--throttle",
        type=float,
        default=DEFAULT_THROTTLE_SECONDS,
        help=(
            f"Seconds to sleep between successful video fetches "
            f"(default {DEFAULT_THROTTLE_SECONDS}). Set to 0 to disable; "
            f"raise to 5-10s if YouTube IP-blocks you mid-run."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def is_short_url(url: str) -> bool:
    """Belt-and-suspenders check that the URL isn't a /shorts/ link.
    Some videos have duration > 60s but are still classified as shorts
    by YouTube via the dedicated /shorts/ URL pattern."""
    return "/shorts/" in (url or "").lower()


def snippets_to_text(snippets) -> str:
    """Convert a list of `FetchedTranscriptSnippet`s (or dicts) from
    youtube-transcript-api into plain text.

    YouTube's auto-caption snippets are short overlapping fragments —
    e.g. one snippet for "Welcome back to the Huberman" and another
    for "back to the Huberman Lab Podcast". We join all snippet texts
    with single spaces, then de-duplicate consecutive duplicates so
    the output reads as a continuous transcript rather than a stutter.
    """
    if not snippets:
        return ""
    unique_lines: list[str] = []
    last = ""
    for snippet in snippets:
        # Support either dataclass-style snippets (`.text`) or dict-style.
        text = snippet.text if hasattr(snippet, "text") else snippet.get("text", "")
        clean = re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()
        if not clean or clean == last:
            continue
        unique_lines.append(clean)
        last = clean
    return "\n".join(unique_lines)


def _safe_filename(s: str) -> str:
    """Make a string safe for use as a filename component. Allow letters,
    digits, dashes, underscores; replace everything else with '_'.
    Kept ASCII to avoid filesystem encoding surprises."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def write_manifest(
    creator_dir: Path,
    entries: list[ManifestEntry],
    creator: Optional[dict] = None,
) -> Path:
    """Write the per-creator manifest to `_manifest.json`. Atomic via
    write-to-temp + rename so a crash mid-write doesn't leave the file
    half-written."""
    manifest_path = creator_dir / "_manifest.json"
    tmp = creator_dir / "_manifest.json.tmp"
    creator_payload = dict(creator or {})
    creator_payload.setdefault("name", creator_dir.name)
    creator_payload.setdefault("slug", creator_dir.name)
    if creator_payload.get("channelUrl") is None:
        creator_payload.pop("channelUrl", None)
    payload = {
        "creator": creator_payload,
        "entries": [asdict(e) for e in entries],
        "writtenAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    return manifest_path


def fetch_channel_video_list(channel_url: str, limit: int = 0):
    """Resolve a channel URL/@handle to a list of `{id, url}` dicts.

    yt-dlp's `extract_flat=True` returns the uploads playlist without
    actually downloading each video — much faster than the default
    behavior which would resolve full metadata for every entry.
    """
    import yt_dlp

    # The `/videos` suffix forces yt-dlp to use the channel's Videos tab
    # specifically, skipping Live and Shorts tabs. A playlist URL
    # (`playlist?list=...`) is already an explicit, ordered video list, so it
    # must be left untouched: appending `/videos` to a playlist URL yields a
    # malformed URL that yt-dlp rejects with HTTP 400. fetch_all_creators.sh
    # uses a playlist URL for the John Campea Show, so this path is load-bearing.
    is_playlist = "list=" in channel_url
    if not is_playlist and not channel_url.rstrip("/").endswith("/videos"):
        channel_url = channel_url.rstrip("/") + "/videos"

    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "playlistreverse": False,
        # Cap the playlist fetch at `limit * 2` to account for shorts we
        # haven't filtered yet. 0 means unlimited.
        "playlistend": limit * 2 if limit > 0 else None,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    entries = info.get("entries") or []
    # The flat extraction returns shallow rows: id, url, title, duration.
    # We don't yet have a publishedAt; that comes from the per-video
    # extraction in the main loop.
    return entries


def fetch_video_metadata(video_url: str) -> dict:
    """Pull metadata (title, duration, upload_date, thumbnail) for one
    video via yt-dlp. Does NOT attempt caption extraction — captions
    come from youtube-transcript-api separately, which sidesteps the
    PO-token issue (yt-dlp #12482)."""
    import yt_dlp

    opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(video_url, download=False)


class IPBlockedError(Exception):
    """Raised when YouTube confirms an IP-ban (RequestBlocked /
    IpBlocked). Distinct from a regular transient error so the caller
    can sleep aggressively before resuming."""


def fetch_video_transcript(video_id: str, lang: str) -> Optional[list]:
    """Fetch the auto-caption transcript for one video via
    youtube-transcript-api. Returns a list of snippet objects (each
    with `.text`, `.start`, `.duration`) or None if no `lang` track
    is available.

    Raises:
      - IPBlockedError on confirmed YouTube IP bans (so the orchestrator
        can sleep long enough for the block to clear).
      - Other exceptions on transient network failures (caller retries).
    """
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    # youtube-transcript-api defines these only in newer versions; import
    # defensively so this works against older installs.
    try:
        from youtube_transcript_api import RequestBlocked, IpBlocked  # type: ignore[attr-defined]

        ip_block_exception_types: tuple = (RequestBlocked, IpBlocked)
    except ImportError:
        ip_block_exception_types = ()

    transcript_api = YouTubeTranscriptApi()
    try:
        result = transcript_api.fetch(video_id, languages=[lang])
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        return None
    except Exception as exc:  # noqa: BLE001
        # Newer library: typed RequestBlocked. Older library: bare
        # `Exception` with the helpful text. Detect both.
        msg = str(exc).lower()
        # `isinstance(exc, ())` raises TypeError, so guard. Also guard
        # against entries in the tuple that aren't real classes (can
        # happen when an upstream import returns sentinel/mock objects).
        typed_match = False
        if ip_block_exception_types:
            try:
                typed_match = isinstance(exc, ip_block_exception_types)
            except TypeError:
                typed_match = False
        is_ip_block = (
            typed_match
            or "youtube is blocking requests from your ip" in msg
            or "requestblocked" in msg
            or "ipblocked" in msg
        )
        if is_ip_block:
            raise IPBlockedError(str(exc)) from exc
        raise
    # Newer versions return a FetchedTranscript object with `.snippets`;
    # older versions return a list directly. Handle both.
    return result.snippets if hasattr(result, "snippets") else result


def process_one_video(
    entry: dict,
    creator_dir: Path,
    lang: str,
    min_duration: int,
) -> ManifestEntry:  # noqa: C901  (the per-status branches keep this honest)
    """Process a single playlist entry: skip / fetch / save. Returns
    the ManifestEntry for the per-creator manifest. Errors are caught
    and recorded as `status="failed"` so a single bad video doesn't
    halt the entire fetch."""
    video_id = entry.get("id") or ""
    url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
    title = entry.get("title") or video_id
    duration = entry.get("duration")

    # Fast-path: shorts URL pattern check (cheap, before any network).
    if is_short_url(url):
        return ManifestEntry(
            videoId=video_id,
            title=title,
            publishedAt=None,
            durationSeconds=duration,
            sourceUrl=url,
            thumbnailUrl=None,
            transcriptPath=None,
            status="skipped",
            skipReason="shorts_url",
        )

    # Fast-path: known-short duration. Some entries from extract_flat
    # already carry duration — skip without fetching full metadata.
    if isinstance(duration, (int, float)) and duration < min_duration:
        return ManifestEntry(
            videoId=video_id,
            title=title,
            publishedAt=None,
            durationSeconds=int(duration),
            sourceUrl=url,
            thumbnailUrl=None,
            transcriptPath=None,
            status="skipped",
            skipReason=f"too_short_{int(duration)}s",
        )

    # Full pipeline per video: metadata via yt-dlp → transcript via
    # youtube-transcript-api. Both wrapped in retries because both can
    # transiently 429 / 500.
    info: Optional[dict] = None
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES_PER_VIDEO):
        try:
            info = fetch_video_metadata(url)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("metadata fetch failed (attempt %d) for %s: %s", attempt + 1, video_id, exc)
            time.sleep(2 ** attempt)
    if info is None:
        return ManifestEntry(
            videoId=video_id,
            title=title,
            publishedAt=None,
            durationSeconds=None,
            sourceUrl=url,
            thumbnailUrl=None,
            transcriptPath=None,
            status="failed",
            skipReason=f"metadata_fetch_failed: {str(last_err)[:160]}" if last_err else "metadata_fetch_failed",
        )

    duration_secs = info.get("duration")
    if isinstance(duration_secs, (int, float)) and duration_secs < min_duration:
        return ManifestEntry(
            videoId=video_id,
            title=info.get("title") or title,
            publishedAt=_format_upload_date(info.get("upload_date")),
            durationSeconds=int(duration_secs),
            sourceUrl=url,
            thumbnailUrl=info.get("thumbnail"),
            transcriptPath=None,
            status="skipped",
            skipReason=f"too_short_{int(duration_secs)}s",
        )

    # Fetch the actual transcript via youtube-transcript-api.
    snippets: Optional[list] = None
    last_err = None
    for attempt in range(MAX_RETRIES_PER_VIDEO):
        try:
            snippets = fetch_video_transcript(video_id, lang=lang)
            break
        except IPBlockedError:
            # Distinct from a transient error — YouTube has confirmed
            # an IP ban. Re-raise so the caller can cool down for
            # several minutes before resuming.
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("transcript fetch failed (attempt %d) for %s: %s", attempt + 1, video_id, exc)
            time.sleep(2 ** attempt)

    if snippets is None and last_err is not None:
        # The library raised something we didn't classify as "no
        # transcript". Record as failure rather than skip.
        return ManifestEntry(
            videoId=video_id,
            title=info.get("title") or title,
            publishedAt=_format_upload_date(info.get("upload_date")),
            durationSeconds=int(duration_secs) if isinstance(duration_secs, (int, float)) else None,
            sourceUrl=url,
            thumbnailUrl=info.get("thumbnail"),
            transcriptPath=None,
            status="failed",
            skipReason=f"transcript_fetch_failed: {str(last_err)[:160]}",
        )

    if not snippets:
        return ManifestEntry(
            videoId=video_id,
            title=info.get("title") or title,
            publishedAt=_format_upload_date(info.get("upload_date")),
            durationSeconds=int(duration_secs) if isinstance(duration_secs, (int, float)) else None,
            sourceUrl=url,
            thumbnailUrl=info.get("thumbnail"),
            transcriptPath=None,
            status="skipped",
            skipReason=f"no_{lang}_captions",
        )

    text = snippets_to_text(snippets)
    if not text.strip():
        return ManifestEntry(
            videoId=video_id,
            title=info.get("title") or title,
            publishedAt=_format_upload_date(info.get("upload_date")),
            durationSeconds=int(duration_secs) if isinstance(duration_secs, (int, float)) else None,
            sourceUrl=url,
            thumbnailUrl=info.get("thumbnail"),
            transcriptPath=None,
            status="skipped",
            skipReason="empty_caption_track",
        )

    # Save the transcript file. Format: YYYY-MM-DD_videoId.txt
    upload_date = _format_upload_date(info.get("upload_date")) or "unknown"
    out_name = f"{upload_date}_{_safe_filename(video_id)}.txt"
    out_path = creator_dir / out_name
    # First two lines are title + URL — the bulk-import endpoint reads
    # them so a folder-only ingest still has structured per-video data.
    out_path.write_text(f"# {info.get('title') or title}\n# {url}\n\n{text}\n", encoding="utf-8")

    return ManifestEntry(
        videoId=video_id,
        title=info.get("title") or title,
        publishedAt=upload_date if upload_date != "unknown" else None,
        durationSeconds=int(duration_secs) if isinstance(duration_secs, (int, float)) else None,
        sourceUrl=url,
        thumbnailUrl=info.get("thumbnail"),
        transcriptPath=str(out_path.relative_to(creator_dir.parent.parent)),
        status="saved",
        skipReason=None,
    )


def _format_upload_date(yt_upload_date: Optional[str]) -> Optional[str]:
    """yt-dlp returns upload_date as 'YYYYMMDD'. Convert to ISO 'YYYY-MM-DD'."""
    if not yt_upload_date or len(yt_upload_date) != 8 or not yt_upload_date.isdigit():
        return None
    return f"{yt_upload_date[0:4]}-{yt_upload_date[4:6]}-{yt_upload_date[6:8]}"


def main(argv=None) -> int:
    """CLI entry point: fetch one channel's transcripts end to end.

    Resolves the channel to a playlist, then iterates videos —
    skipping shorts and already-on-disk files, retrying transient
    errors, and cooling down (bounded) on confirmed IP blocks. The
    ``--limit`` budget counts only NEW fetches this run, so a resume
    against a partially-populated folder keeps making progress instead
    of stopping at the first N already-downloaded videos. Writes a
    ``_manifest.json`` periodically and at the end.

    Returns 0 on completion, 2 if it bailed out on a persistent IP block.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    creator_dir: Path = args.out_dir / args.creator_slug
    creator_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing transcripts to %s", creator_dir)
    creator_meta = {
        "name": args.creator_name,
        "slug": args.creator_slug,
        "channelUrl": args.channel,
    }

    logger.info("Resolving channel: %s", args.channel)
    playlist_entries = fetch_channel_video_list(args.channel, limit=args.limit)
    logger.info("Channel returned %d candidate videos", len(playlist_entries))

    # Apply hard limit AFTER fetching the broader playlist so the limit
    # corresponds to "N newly-fetched videos", not "N raw videos including
    # shorts". We filter shorts during processing and stop once we've
    # newly fetched `--limit` videos this run.

    entries: list[ManifestEntry] = []
    # `saved_count` is total recorded (incl. already-on-disk) and drives
    # manifest-cadence + the final summary. `newly_saved_count` counts ONLY
    # videos we actually fetched this run, and is what `--limit` is checked
    # against. Counting already-on-disk files toward the limit (the old
    # behavior) made resume stop early: 30 files on disk + `--limit 30`
    # would exit having fetched zero new videos.
    saved_count = 0
    newly_saved_count = 0
    for playlist_entry in playlist_entries:
        if args.limit > 0 and newly_saved_count >= args.limit:
            break
        # Skip if transcript file already exists for this video — makes
        # the script resumable on interruption.
        video_id = playlist_entry.get("id") or ""
        existing_transcripts = list(creator_dir.glob(f"*_{_safe_filename(video_id)}.txt"))
        if existing_transcripts:
            logger.info("skip (already saved): %s", video_id)
            entries.append(
                ManifestEntry(
                    videoId=video_id,
                    title=playlist_entry.get("title") or video_id,
                    publishedAt=None,
                    durationSeconds=playlist_entry.get("duration"),
                    sourceUrl=playlist_entry.get("url") or "",
                    thumbnailUrl=None,
                    transcriptPath=str(existing_transcripts[0].relative_to(creator_dir.parent.parent)),
                    status="saved",
                    skipReason="already_on_disk",
                )
            )
            saved_count += 1
            # Intentionally NOT incrementing newly_saved_count: an
            # already-on-disk file is not a fresh fetch, so it must not
            # consume the run's --limit budget.
            continue

        # Process with IP-block retry: on a confirmed block, cool down
        # for IP_BLOCK_COOLDOWN_SECONDS then retry the SAME video. We
        # cap cool-downs per session so a permanent block doesn't loop
        # forever.
        ip_block_cooldowns_used = 0
        MAX_IP_BLOCK_COOLDOWNS = 3
        while True:
            try:
                entry = process_one_video(
                    playlist_entry,
                    creator_dir=creator_dir,
                    lang=args.lang,
                    min_duration=args.min_duration,
                )
                break
            except IPBlockedError as exc:
                ip_block_cooldowns_used += 1
                if ip_block_cooldowns_used > MAX_IP_BLOCK_COOLDOWNS:
                    logger.error(
                        "Persistent IP block; bailing after %d cooldowns. Last error: %s",
                        MAX_IP_BLOCK_COOLDOWNS,
                        exc,
                    )
                    # Record the failure and bail out of the whole loop.
                    entries.append(
                        ManifestEntry(
                            videoId=playlist_entry.get("id") or "",
                            title=playlist_entry.get("title") or "",
                            publishedAt=None,
                            durationSeconds=playlist_entry.get("duration"),
                            sourceUrl=playlist_entry.get("url") or "",
                            thumbnailUrl=None,
                            transcriptPath=None,
                            status="failed",
                            skipReason="ip_blocked_persistent",
                        )
                    )
                    write_manifest(creator_dir, entries, creator_meta)
                    logger.error("Bailed out at %d saved.", saved_count)
                    return 2
                cooldown = IP_BLOCK_COOLDOWN_SECONDS * ip_block_cooldowns_used
                logger.warning(
                    "IP-blocked (cooldown %d/%d). Sleeping %ss before retry.",
                    ip_block_cooldowns_used,
                    MAX_IP_BLOCK_COOLDOWNS,
                    cooldown,
                )
                time.sleep(cooldown)

        entries.append(entry)
        if entry.status == "saved":
            saved_count += 1
            newly_saved_count += 1
            logger.info(
                "saved %d/%s: %s (%ss)",
                newly_saved_count,
                args.limit if args.limit > 0 else "?",
                entry.title[:60],
                entry.durationSeconds,
            )
            # Throttle ONLY on success. Skipped / failed videos didn't
            # hammer the API, so no need to sleep after them.
            if args.throttle > 0:
                time.sleep(args.throttle)
        elif entry.status == "skipped":
            logger.info("skip (%s): %s", entry.skipReason, entry.title[:60])
        else:
            logger.warning("FAIL %s: %s", entry.videoId, entry.skipReason)

        # Write the manifest every 10 videos so a crash mid-run still
        # leaves us with a usable manifest of progress so far.
        if saved_count > 0 and saved_count % 10 == 0:
            write_manifest(creator_dir, entries, creator_meta)

    write_manifest(creator_dir, entries, creator_meta)
    logger.info("Done. Saved %d transcripts; manifest at %s/_manifest.json", saved_count, creator_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
