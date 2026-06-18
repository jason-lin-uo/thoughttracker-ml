#!/usr/bin/env bash
# fetch_all_creators.sh — orchestrate the 5-creator fetch.
#
# Runs each creator's fetch sequentially so we don't hammer YouTube
# with 5 parallel scrapes from the same IP (high probability of
# rate-limiting). Each creator gets its own log file under
# data/transcripts/<slug>/_fetch.log so partial progress is debuggable.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
FETCHER="$ROOT/scripts/fetch_transcripts.py"
OUT="$ROOT/data/transcripts"
mkdir -p "$OUT"

# run: fetch one creator's transcripts into its own log file.
# Args: <slug> <channel-url> <display-name> <limit>. Failures are logged
# but don't abort the orchestrator (best-effort per creator).
run() {
  local slug="$1" channel="$2" name="$3" limit="$4"
  local logdir="$OUT/$slug"
  mkdir -p "$logdir"
  echo "[orchestrator] starting $slug (limit=$limit) at $(date -u +%FT%TZ)"
  "$PYTHON" "$FETCHER" \
    --channel "$channel" \
    --creator-name "$name" \
    --creator-slug "$slug" \
    --out-dir "$OUT" \
    --limit "$limit" \
    >>"$logdir/_fetch.log" 2>&1 || echo "[orchestrator] $slug exited non-zero" >&2
  echo "[orchestrator] finished $slug at $(date -u +%FT%TZ)"
}

# Order: shortest catalogs first so we get fast feedback that the
# pipeline is working before committing to the long runs.
run "huberman"  "https://www.youtube.com/@hubermanlab"     "Andrew Huberman"  0
run "allin"     "https://www.youtube.com/@allin"           "All In Podcast"   0
run "mkbhd"     "https://www.youtube.com/@mkbhd"           "Marques Brownlee" 0
run "delauer"   "https://www.youtube.com/@ThomasDeLauerOfficial" "Thomas DeLauer" 1500
run "campea"    "https://www.youtube.com/playlist?list=PL6628E7149D3A7D56" "John Campea" 500

echo "[orchestrator] all done at $(date -u +%FT%TZ)"
