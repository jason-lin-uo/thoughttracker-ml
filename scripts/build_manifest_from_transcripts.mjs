#!/usr/bin/env node
// @ts-check
/**
 * build_manifest_from_transcripts.mjs — reconstruct a bulk-import
 * `_manifest.json` from a folder of already-fetched transcript `.txt` files.
 *
 * Why this exists
 * ---------------
 * `fetch_transcripts.py` normally writes `_manifest.json` next to the `.txt`
 * files it downloads. When a creator's transcripts are committed or copied
 * WITHOUT that manifest (as the gold-standard creator corpora were), the
 * backend bulk-import job — which reads `<folder>/_manifest.json` — has
 * nothing to ingest. This script regenerates the manifest deterministically
 * and offline from each file's header (`# Title` line + `# https://…v=<id>`
 * line) and its `YYYY-MM-DD_<videoId>.txt` filename. No network, no YouTube
 * API: it only reads files already on disk.
 *
 * The output shape matches `BulkImportManifest` in
 * `thoughttracker/backend/src/jobs/bulkImport.job.ts`:
 * { creator: { name, slug, channelUrl? }, entries: [ {…} ], writtenAt }
 *
 * Node built-ins only — runs unchanged on macOS / Linux / Windows.
 *
 * Usage:
 * node scripts/build_manifest_from_transcripts.mjs \
 * --dir data/transcripts/huberman \
 * --name "Andrew Huberman" --slug huberman \
 * [--channel-url https://www.youtube.com/@hubermanlab] \
 * [--limit 0] # 0 = every transcript; N = the N most-recent by date
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";

/** Transcript filenames are `YYYY-MM-DD_<videoId>.txt`. */
const FILENAME_RE = /^(\d{4}-\d{2}-\d{2})_(.+)\.txt$/;
/** Extract a YouTube id from a watch URL (`?v=<id>`); ids are [A-Za-z0-9_-]. */
const URL_VIDEO_RE = /[?&]v=([A-Za-z0-9_-]{6,})/;

/**
 * Parse `--flag value` / `--flag=value` pairs into a plain object. Unknown
 * flags are kept so a typo surfaces as an unused key rather than a crash.
 */
function parseArgs(argv) {
  /** @type {Record<string,string>} */
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const tok = argv[i];
    if (!tok.startsWith("--")) continue;
    const eq = tok.indexOf("=");
    if (eq !== -1) {
      out[tok.slice(2, eq)] = tok.slice(eq + 1);
    } else {
      const next = argv[i + 1];
      // A bare `--flag` with no following value is treated as boolean "true".
      out[tok.slice(2)] =
        next && !next.startsWith("--") ? ((i += 1), next) : "true";
    }
  }
  return out;
}

/**
 * Build one manifest entry from a single transcript file, or return null if
 * the filename isn't a recognised `<date>_<id>.txt` transcript. The video id
 * is taken from the header URL when present (authoritative) and otherwise
 * falls back to the id embedded in the filename.
 */
function parseEntry(dir, file) {
  const m = FILENAME_RE.exec(file);
  if (!m) return null;
  const [, date, idFromName] = m;
  // Only the first three lines matter (title, url, blank) — read them cheaply.
  const head = readFileSync(path.join(dir, file), "utf-8").split("\n", 3);
  const title = (head[0] || "").replace(/^#\s?/, "").trim() || idFromName;
  const urlLine = (head[1] || "").replace(/^#\s?/, "").trim();
  const urlMatch = URL_VIDEO_RE.exec(urlLine);
  const videoId = urlMatch ? urlMatch[1] : idFromName;
  const sourceUrl = urlLine || `https://www.youtube.com/watch?v=${videoId}`;
  return {
    videoId,
    title,
    // Filename carries only a date; midnight UTC is a stable, sortable stamp
    // for the timeline x-axis (the real upload time-of-day is not preserved).
    publishedAt: `${date}T00:00:00.000Z`,
    durationSeconds: null,
    sourceUrl,
    transcriptPath: file, // bulk-import resolves this by basename within --dir
    status: "saved",
  };
}

/** Read the dir, build entries, sort newest-first, apply --limit, write JSON. */
function main() {
  const args = parseArgs(process.argv.slice(2));
  const dir = args.dir;
  const name = args.name;
  const slug = args.slug;
  if (!dir || !name || !slug) {
    process.stderr.write(
      "Usage: --dir <folder> --name <creator> --slug <slug> [--channel-url url] [--limit N]\n",
    );
    process.exit(2);
  }
  const limit = Number.parseInt(args.limit ?? "0", 10) || 0;

  const entries = readdirSync(dir)
    .map((file) => parseEntry(dir, file))
    .filter((e) => e !== null)
    // Newest first so a positive --limit keeps the most recent videos, which
    // give the densest, most current end of the stance timeline.
    .sort((a, b) => b.publishedAt.localeCompare(a.publishedAt));

  const kept = limit > 0 ? entries.slice(0, limit) : entries;

  const manifest = {
    creator: {
      name,
      slug,
      channelUrl: args["channel-url"] ?? null,
    },
    entries: kept,
    writtenAt: new Date().toISOString(),
  };

  const outPath = path.join(dir, "_manifest.json");
  writeFileSync(outPath, JSON.stringify(manifest, null, 2), "utf-8");
  process.stdout.write(
    `Wrote ${outPath} — ${kept.length} of ${entries.length} transcripts (limit=${limit || "all"}).\n`,
  );
}

main();
