# Scripts

These scripts are the intentional local automation layer for the ML companion
repo. They are kept here because they operate on local transcript/model files
and call the TypeScript app over HTTP.

## Creator Onboarding

Use this path when adding a new YouTube creator to an already calibrated
ThoughtTracker install.

All orchestration scripts are cross-platform Node (`.mjs`, Node built-ins
only) so they run on macOS, Linux, and Windows with a stock Node install —
no Windows-only shell required. Run them with `node scripts/<name>.mjs`.

| Script | Purpose |
| --- | --- |
| `add_creator_pipeline.mjs` | One-command owner workflow: fetch transcripts, ingest them, reanalyze with the frozen policy, run a quality audit, and create a review packet if uncertain rows need owner review. |
| `fetch_transcripts_ytdlp.py` | Cross-platform transcript fetcher using `yt-dlp`; writes clean `.txt` transcripts under `data/transcripts/<creator>/`. A `--subprocess-timeout` wall-clock cap guards against a hung fetcher. |
| `ingest_all_transcripts.mjs` | Bulk-ingests the known local transcript folders into the backend. |
| `run_reanalyze_latest_model.mjs` | Reanalyzes existing videos against the current frozen topic/stance policy. Called by the onboarding wrapper. |

## Keeping Existing Creators Up To Date

Use this path to refresh creators that are ALREADY in the database — it only
fetches NEW uploads, not the whole back catalog.

| Script | Purpose |
| --- | --- |
| `update_all_creators.mjs` | Refresh EVERY existing creator: enumerates creators via the backend API, then for each one fetches new transcripts (`fetch_transcripts.py`, which skips videos already on disk) and — **only if that creator has new uploads** — ingests the folder (`ingest_transcripts.py` → bulk-import, which upserts videos and enqueues analysis). Creators with no new uploads are skipped entirely. Does **not** run the full-DB reanalysis / quality-audit / packet phases. Per-creator-resilient (one bad channel doesn't abort the rest). Requires the backend reachable at `--api-base`; it does **not** auto-start servers. Writes a pollable `reports/metrics/update_all_creators_status.json`, so it is shaped to be spawned by a future admin endpoint exactly like `add_creator_pipeline.mjs` is today. Flags: `--api-base`, `--admin-pin` (or env `THOUGHTTRACKER_ADMIN_PIN`), `--limit` (cap new videos/creator, 0 = all), `--only <slug>` (repeatable), `--dry-run`, `--skip-fetch`, `--skip-ingest`. Run: `node scripts/update_all_creators.mjs`. |

`add_creator_pipeline.mjs` (onboarding) is for ADDING a new creator and runs the
full reanalysis + quality audit + review packet; `update_all_creators.mjs`
(routine refresh) is the lighter path for keeping known creators current.

**Known cost (not yet fully incremental):** when a creator has new uploads, the
whole folder is re-ingested, and bulk-import currently re-chunks + re-enqueues
analysis for *every* video in that folder — including the creator's already-
analyzed back-catalog (on-disk transcripts are re-listed with status `saved` +
`skipReason="already_on_disk"`, which bulk-import still re-processes). New videos
are analyzed correctly; the waste is re-analyzing unchanged ones. Making it fully
incremental needs a small bulk-import change: skip re-chunk/re-analysis for a
manifest entry whose video already has `analysisStatus="completed"` and is flagged
`already_on_disk`. Until then, creators with no new uploads cost nothing, and a
full reanalysis should be run explicitly only when the frozen topic/stance policy
changes.

## Evaluation

| Script | Purpose |
| --- | --- |
| `evaluate_hybrid_topic_pipeline.py` | Recomputes topic-selection metrics for the frozen hybrid topic pipeline and writes final metrics reports. |

## Historical Scripts Removed

Old ChatGPT packet exports, failed calibration rounds, VTT caches, and one-off
overnight helper scripts are not part of the clean repo state. The remaining
scripts are either reusable onboarding tools or metric-verification tools.
