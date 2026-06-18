#!/usr/bin/env python
"""
ingest_transcripts.py — POST a fetched transcript folder to the
ThoughtTracker backend's bulk-import endpoint.

This is the Stage 3 glue between the Python fetcher
(`fetch_transcripts.py`) and the TypeScript backend's bulk-import job
(`backend/src/jobs/bulkImport.job.ts`).

Why a separate script (vs running curl)?
   - Computes the `folderPath` the backend should read. By default it
     sends a path RELATIVE to the configured transcripts root
     (`data/transcripts/<creator>`) rather than this host's absolute path:
     the backend resolves it against its own cwd and enforces a
     `BULK_IMPORT_ROOT` allowlist, so a relative path is both portable and
     stays inside the allowlist. Pass `--path-mode absolute` for the legacy
     absolute-path behavior (see `resolve_folder_path_for_backend`).
   - Patches the `_manifest.json` with the resolved creator metadata
     (name + slug come from CLI args, not the fetcher's manifest, so
     the same fetched folder can be ingested under different display
     names without re-fetching).
   - Polls the job status after enqueueing so the caller sees progress
     instead of just a job-id and silence.

Usage
-----
    python scripts/ingest_transcripts.py \\
        --folder data/transcripts/huberman \\
        --creator-name "Andrew Huberman" \\
        --creator-slug huberman \\
        --channel-url "https://www.youtube.com/@hubermanlab" \\
        --api-base http://localhost:4000/api

The script ensures the `_manifest.json` has a `creator` block (the
fetcher writes one but only with `name + slug + channelUrl`). If the
manifest already has matching fields, they're left alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

# `requests` ships in most Python environments; gracefully degrade to
# `urllib` if it's missing so this script has zero install dependencies.
try:
    import requests  # type: ignore[import-not-found]

    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False
    import urllib.request
    import urllib.error

LOGGER = logging.getLogger("ingest_transcripts")


def _parse_args(argv=None):
    """Parse CLI flags for the transcript-folder ingest script."""
    p = argparse.ArgumentParser(description="POST a transcripts folder to ThoughtTracker.")
    p.add_argument("--folder", required=True, type=Path)
    p.add_argument("--creator-name", required=True)
    p.add_argument("--creator-slug", required=True)
    p.add_argument("--channel-url", default=None)
    p.add_argument("--description", default=None)
    p.add_argument(
        "--transcripts-root",
        type=Path,
        default=Path("data/transcripts"),
        help=(
            "The backend's bulk-import allowlist root (its BULK_IMPORT_ROOT, "
            "defaulting to <backend cwd>/data/transcripts). Used to compute "
            "the folderPath we send the backend. See --path-mode."
        ),
    )
    p.add_argument(
        "--path-mode",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "How to express folderPath to the backend. 'relative' (default) "
            "sends a path the backend resolves against its OWN cwd "
            "(data/transcripts/<creator>), which works when the backend runs "
            "from the project root sharing this filesystem and keeps the "
            "request portable + inside the allowlist. 'absolute' sends the "
            "resolved absolute path (the old behavior) for when the script "
            "and backend share a filesystem but not a working directory."
        ),
    )
    p.add_argument(
        "--api-base",
        default="http://localhost:4000/api",
        help="ThoughtTracker backend base URL.",
    )
    p.add_argument(
        "--admin-pin",
        default=None,
        help="Optional admin PIN for creator-onboarding protected backends.",
    )
    p.add_argument(
        "--no-poll",
        action="store_true",
        help="Return immediately after enqueueing instead of polling for completion.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=3.0,
        help="Seconds between poll attempts while waiting for job completion.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Stop polling after this many seconds (default 10 min).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _admin_headers(admin_pin: Optional[str]) -> dict:
    """Build request headers, adding ``X-Admin-Pin`` only when a PIN is given."""
    if not admin_pin:
        return {"Content-Type": "application/json"}
    return {"Content-Type": "application/json", "X-Admin-Pin": admin_pin}


def _http_post_json(url: str, body: dict, admin_pin: Optional[str] = None) -> dict:
    """POST a JSON body and return the parsed JSON response.

    Uses ``requests`` when available, else falls back to stdlib
    ``urllib`` so the script has zero hard dependencies. A 30s timeout
    guards against a hung backend.
    """
    if HAVE_REQUESTS:
        resp = requests.post(url, json=body, timeout=30,
                             headers=_admin_headers(admin_pin))
        resp.raise_for_status()
        return resp.json()
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=_admin_headers(admin_pin),
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url: str) -> dict:
    """GET a URL and return the parsed JSON response (30s timeout)."""
    if HAVE_REQUESTS:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def patch_manifest(folder: Path, creator_name: str, creator_slug: str,
                   channel_url: Optional[str], description: Optional[str]) -> None:
    """Ensure the manifest has a top-level `creator` block matching the
    caller's intent. The fetcher writes one with the slug + name we
    asked for, but a hand-prepared folder might not, so we be defensive.
    """
    manifest_path = folder / "_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No _manifest.json at {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    creator = data.get("creator") or {}
    creator.setdefault("name", creator_name)
    creator.setdefault("slug", creator_slug)
    if channel_url:
        creator["channelUrl"] = channel_url
    if description:
        creator["description"] = description
    data["creator"] = creator
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_folder_path_for_backend(
    folder: Path, transcripts_root: Path, path_mode: str
) -> str:
    """Compute the ``folderPath`` string to POST to the backend.

    Coupling note (audit §7 "POSTs absolute path"): the backend's
    bulk-import endpoint runs ``path.resolve(folderPath)`` and then verifies
    the result is inside its allowlist root (``BULK_IMPORT_ROOT``, default
    ``<backend cwd>/data/transcripts``). A *relative* path is therefore
    resolved against the BACKEND's cwd, not this script's — so the two
    processes must share a filesystem AND the backend must run from the
    project root for a relative path to land inside the allowlist.

    Modes:
      - ``relative`` (default): return ``<transcripts_root_name>/<folder
        name>`` (e.g. ``data/transcripts/huberman``). This is portable and
        stays inside the allowlist when the backend runs from the project
        root, instead of leaking this host's absolute layout.
      - ``absolute``: return the absolute resolved path (legacy behavior),
        for when the backend shares the filesystem but not the cwd.

    When the folder isn't actually under ``transcripts_root`` we fall back
    to the absolute path so we never silently send a wrong relative path.
    """
    folder = folder.resolve()
    if path_mode == "absolute":
        return str(folder)
    root = transcripts_root.resolve()
    try:
        relative = folder.relative_to(root)
    except ValueError:
        # Folder isn't under the configured root — a relative path would be
        # meaningless to the backend, so send the absolute path instead.
        return str(folder)
    # Re-attach the root's own (relative) name so the backend, resolving
    # against its cwd, lands on <cwd>/data/transcripts/<creator>.
    return str(transcripts_root / relative) if not transcripts_root.is_absolute() else str(folder)


def main(argv=None) -> int:
    """CLI entry point: patch the manifest, enqueue the bulk-import, poll.

    Resolves the folder, ensures its ``_manifest.json`` carries the
    caller's creator metadata, POSTs the bulk-import job with a
    ``folderPath`` computed per ``--path-mode`` (relative-to-transcripts-root
    by default; see :func:`resolve_folder_path_for_backend`), and (unless
    ``--no-poll``) polls until the job reaches a terminal state or
    ``--timeout`` elapses.

    Returns: 0 success/no-poll, 2 bad folder, 3 enqueue failure, 4 job
    failed, 5 poll timeout.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    folder = args.folder.resolve()
    if not folder.is_dir():
        LOGGER.error("Not a directory: %s", folder)
        return 2

    LOGGER.info("Patching manifest for %s", folder)
    patch_manifest(folder, args.creator_name, args.creator_slug,
                   args.channel_url, args.description)

    folder_path = resolve_folder_path_for_backend(
        folder, args.transcripts_root, args.path_mode
    )
    api = args.api_base.rstrip("/")
    LOGGER.info(
        "Enqueuing bulk-import job at %s/import-jobs/bulk-import (folderPath=%s)",
        api,
        folder_path,
    )
    body = {"folderPath": folder_path}
    try:
        result = _http_post_json(f"{api}/import-jobs/bulk-import", body,
                                 args.admin_pin)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("POST failed: %s", exc)
        return 3
    job_id = result.get("jobId")
    if not job_id:
        LOGGER.error("Unexpected response: %s", result)
        return 3
    LOGGER.info("Enqueued job %s (status %s)", job_id, result.get("status"))

    if args.no_poll:
        print(job_id)
        return 0

    # Poll until terminal.
    started = time.time()
    last_status = ""
    while time.time() - started < args.timeout:
        try:
            job = _http_get_json(f"{api}/import-jobs/{job_id}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("poll failed: %s", exc)
            time.sleep(args.poll_interval)
            continue
        status = job.get("status", "unknown")
        if status != last_status:
            LOGGER.info(
                "status=%s imported=%s transcripts=%s failed=%s",
                status,
                job.get("totalVideosImported"),
                job.get("totalTranscriptsImported"),
                job.get("totalFailed"),
            )
            last_status = status
        if status in {"completed", "completed_with_errors", "failed"}:
            LOGGER.info("Final: %s", json.dumps(job, indent=2))
            return 0 if status != "failed" else 4
        time.sleep(args.poll_interval)

    LOGGER.error("Timed out waiting for job %s after %ss", job_id, args.timeout)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
