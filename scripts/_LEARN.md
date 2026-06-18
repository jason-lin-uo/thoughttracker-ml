# \_LEARN.md — `thoughttracker-ml/scripts/`

> ~10 operational scripts (cross-platform Node `.mjs`, Python, one Bash).
> The jobs that fetch transcripts, ingest them into the main app, reanalyze
> against the frozen topic/stance policy, and recompute topic-selection
> metrics. See `scripts/README.md` for the owner-facing runbook.

---

## The story of this folder

If `src/` is the factory floor (where the same machines run the same
process every day), `scripts/` is the **shipping/receiving dock** —
where one-off jobs happen. Receiving raw materials (downloading
transcripts, ingesting public datasets), shipping finished work out
(POSTing transcripts to the main app's bulk-import endpoint), and the
occasional batch operation that's too quirky to live in `src/`.

These scripts aren't called from anywhere in `src/`. They're called by
**humans** (or cron jobs) when there's data to fetch, convert, or
load. They tend to be:

- Longer than they look (lots of edge cases — YouTube rate limits, TSV
  quoting quirks, partial failures).
- Idempotent (re-running shouldn't double-do anything).
- Loud (they log what they're doing because nobody's watching).

---

## File-by-file

### `fetch_transcripts.py`

**What it is:** the **YouTube transcript grabber**. About 350 lines.
Pulls auto-caption transcripts for a YouTube channel and saves one
`.txt` per video.

**Why it exists:** ThoughtTracker analyzes long-form YouTube content.
Pulling transcripts manually for hundreds of videos isn't viable, and
the official YouTube Data API doesn't expose captions for channels
you don't own. **`yt-dlp`** (the modern fork of youtube-dl — in plain terms, a well-known open-source command-line tool for grabbing YouTube videos and their caption tracks) parses the
same caption tracks YouTube serves to viewers in the browser, so we
use that.

**What it does:**

1. Resolves a channel URL or `@handle` to the full uploads playlist.
2. Iterates every video in the list.
3. For each video, skips if it's shorter than `MIN_DURATION_SECONDS`
   (default 60s — filters out YouTube Shorts, which don't have
   meaningful long-form analysis value).
4. Downloads the auto-caption transcript (`.vtt`). (VTT is the standard subtitle file format used on the web — a plain-text file with timestamps next to each caption line.)
5. Cleans the VTT (strips timestamps, removes duplicate adjacent
   lines, collapses to plain text).
6. Writes to `data/transcripts/<creator-slug>/<video-id>.txt`.
7. Writes a `_manifest.json` summarizing what was fetched (video
   IDs, titles, durations, success/skip/fail per video).

**Quirks it handles:**

- Auto-captions sometimes don't exist (private/disabled). Skip.
- Some videos are deleted between listing and downloading. Skip.
- YouTube rate-limits aggressive scraping. Built-in throttle.
- Network flakes get retried with backoff.

**Usage:**

```bash
python scripts/fetch_transcripts.py \
 --channel "@andrewhuberman" \
 --slug huberman \
 --limit 30
```

---

### `fetch_all_creators.sh`

**What it is:** a thin Bash orchestrator that calls
`fetch_transcripts.py` for 5 specific creators in sequence.

**Why a shell script:** the work is sequential (don't parallelize —
YouTube will rate-limit the IP). Each creator gets its own log file
under `data/transcripts/<slug>/_fetch.log` so partial progress is
debuggable. Bash is fine for this — it's just looping over a list and
calling a Python script.

**Why a separate script and not just args to the Python:** keeps the
Python script generic (any channel, any slug) and lets the shell
script encode the fixed 5-creator corpus used by the product.

---

### `fetch_transcripts_ytdlp.py`

**What it is:** the **cross-platform transcript fetcher** built on `yt-dlp`.
Writes clean `.txt` transcripts under `data/transcripts/<creator>/` plus a
`_manifest.json`. A `--subprocess-timeout` wall-clock cap guards against a
hung fetcher. This is the fetcher the onboarding pipeline shells out to.

### `add_creator_pipeline.mjs`

**What it is:** the **one-command owner workflow** for ADDING a new creator
to an already-calibrated install. End-to-end: fetch transcripts → ingest →
reanalyze against the frozen topic/stance policy → run a quality audit →
create a review packet if uncertain rows need owner review. Node built-ins
only, so it runs unchanged on macOS / Linux / Windows.

### `update_all_creators.mjs`

**What it is:** the **routine refresh** for creators ALREADY in the DB. It
enumerates creators via the backend API and, per creator, fetches only NEW
uploads (`fetch_transcripts.py`, which skips videos already on disk) and —
only if there are new uploads — ingests the folder. Skips the full-DB
reanalysis / audit / packet phases, is per-creator-resilient, and writes a
pollable `reports/metrics/update_all_creators_status.json`. Flags: `--api-base`,
`--admin-pin`, `--limit`, `--only`, `--dry-run`, `--skip-fetch`, `--skip-ingest`.

### `ingest_all_transcripts.mjs`

**What it is:** bulk-ingests the known local transcript folders into the
backend in one shot — the multi-folder sibling of `ingest_transcripts.py`.

### `run_reanalyze_latest_model.mjs`

**What it is:** reanalyzes existing videos against the current frozen
topic/stance policy. Called by `add_creator_pipeline.mjs`, or run directly
after the frozen policy changes.

### `build_manifest_from_transcripts.mjs`

**What it is:** reconstructs a bulk-import `_manifest.json` from a folder of
already-fetched `.txt` files — offline, from each file's header (`# Title`
line + watch URL) and its `YYYY-MM-DD_<videoId>.txt` filename. Needed when a
creator's transcripts were committed/copied WITHOUT the manifest (as the
gold-standard corpora were), since the backend bulk-import reads
`<folder>/_manifest.json`. No network, no YouTube API.

### `evaluate_hybrid_topic_pipeline.py`

**What it is:** recomputes topic-selection metrics for the frozen hybrid
topic pipeline and writes the final metrics reports under `reports/metrics/`.
This is the script behind the README's headline topic-selection numbers.

---

### `ingest_transcripts.py`

**What it is:** **the bridge between this repo and the main app**.
About 250 lines. After `fetch_transcripts.py` has produced a folder
of `.txt` transcripts, this script POSTs them to the main
ThoughtTracker backend's bulk-import endpoint
(`POST /api/import-jobs/bulk` with `inline=true`).

**Why a script and not just curl:**

- Resolves the transcripts folder to an absolute path the backend
  process can read (the backend might be in a Docker container or
  on a different machine).
- Patches the `_manifest.json` with the resolved creator metadata
  (name + slug) — comes from CLI args, not from the fetcher's
  manifest. This means the same fetched folder can be ingested under
  different display names without re-fetching.
- **Polls the job status** after enqueueing — so the caller sees
  progress ("imported 12/30 videos...") instead of just a job ID and
  silence.

**Usage:**

```bash
python scripts/ingest_transcripts.py \
 --folder data/transcripts/huberman \
 --creator-name "Andrew Huberman" \
 --creator-slug huberman \
 --backend-url http://localhost:4000
```

**This script is the only one that talks to the main repo.** Everything
else in this folder is self-contained.

---

## How scripts/ connects to everything else

```
[YouTube]
 │ fetch_transcripts_ytdlp.py / fetch_transcripts.py
 │ (fetch_all_creators.sh wraps the fetcher for the fixed creator roster)
 ▼
data/transcripts/<slug>/*.txt + _manifest.json
 │ (build_manifest_from_transcripts.mjs regenerates the manifest
 │ for corpora committed without one)
 ▼
ingest_transcripts.py / ingest_all_transcripts.mjs
 │ POST → backend /api/import-jobs/bulk (inline=true)
 ▼
main app DB: new Creator + Videos + Transcripts
 │ backend's bulkImport job chunks + enqueues analysis
 ▼
run_reanalyze_latest_model.mjs ── reanalyze against the frozen policy
 │
 ▼
evaluate_hybrid_topic_pipeline.py ── recompute topic-selection metrics

add_creator_pipeline.mjs orchestrates the whole top-to-bottom path;
update_all_creators.mjs runs the lighter fetch + ingest refresh.
```

---

## Why these are scripts and not part of `src/`

- They have **side effects on external systems** (YouTube, HTTP calls,
  file writes) that we don't want test code to exercise.
- They're **invoked rarely** — not part of the every-request hot path.
- They have **lots of CLI-specific concerns** (argparse, progress
  bars, logging configuration, sigint handling) that bloat library
  code.

Keeping them in `scripts/` lets the import-time cost of `src/` stay
low and keeps the unit-test scope tighter.

---

## "Where do I look when X happens"

| You want to fix...                       | Open...                                                                                                                      |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| YouTube fetch failing                    | `fetch_transcripts_ytdlp.py` / `fetch_transcripts.py` — rate-limit handling, `--subprocess-timeout`, then the yt-dlp version |
| Transcripts have weird formatting        | the fetcher's VTT-to-text cleanup section                                                                                    |
| Add a brand-new creator                  | `add_creator_pipeline.mjs` (fetch → ingest → reanalyze → audit → packet)                                                     |
| Refresh existing creators' new uploads   | `update_all_creators.mjs`                                                                                                    |
| Bulk-import polling stuck                | `ingest_transcripts.py` — the polling loop's timeout                                                                         |
| Backend at a different URL               | `ingest_transcripts.py` `--backend-url`, or the `.mjs` scripts' `--api-base`                                                 |
| Manifest missing for a committed corpus  | `build_manifest_from_transcripts.mjs`                                                                                        |
| Topic-selection metrics need recomputing | `evaluate_hybrid_topic_pipeline.py`                                                                                          |
| Stance dataset CSV column mismatch       | `src/data/load_dataset.py` (the validation logic)                                                                            |
