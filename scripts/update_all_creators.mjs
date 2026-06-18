#!/usr/bin/env node
// @ts-check
/**
 * update_all_creators.mjs — incrementally refresh transcripts for EVERY
 * existing creator.
 *
 * Unlike `add_creator_pipeline.mjs` (the heavyweight ONBOARDING workflow:
 * full-DB reanalysis + quality audit + owner-review packet), this is the
 * ROUTINE-REFRESH path: for each creator already in the database it fetches any
 * NEW uploads and, only when there are new uploads, ingests that creator's
 * transcript folder. It reuses the two working, tested sub-scripts that do the
 * real work —
 * - `fetch_transcripts.py` (skips transcripts already on disk → only new
 * uploads are downloaded), and
 * - `ingest_transcripts.py` (POSTs the folder to /import-jobs/bulk-import,
 * which upserts videos and enqueues analysis) —
 * and does NOT run the full-database reanalysis, quality audit, or review
 * packet phases.
 *
 * Cost note — read this before assuming "incremental":
 * - Creators with NO new uploads are skipped entirely (this script reads the
 * per-creator manifest after fetch and skips the ingest), so they cost
 * nothing beyond the channel listing.
 * - Creators WITH new uploads have their whole folder re-ingested. New videos
 * are stance/topic-analyzed automatically at ingest — but bulk-import
 * currently re-chunks and re-enqueues analysis for that creator's EXISTING
 * videos too (it processes every "saved" manifest entry, and the on-disk
 * ones carry skipReason="already_on_disk" but still status="saved"). So a
 * creator who posted 1 new video pays to re-analyze that creator's full
 * back-catalog. Making this fully incremental needs a bulk-import change
 * (skip re-analysis of already-completed videos); see scripts/README.md.
 * - Run the dedicated reanalysis separately if you change the frozen
 * topic/stance policy and want every existing video recomputed under it.
 *
 * Why not just call `add_creator_pipeline.mjs` per creator: its always-on
 * quality-audit phase shells to a backend script that isn't present in the
 * current repo, its reanalysis phase wipes-and-recomputes the WHOLE DB (wrong
 * for a refresh and very slow), and per-creator invocations would clobber its
 * single hard-coded status file. This script keeps the same orchestration
 * *shell* (status JSON, logging, admin-PIN-from-env) so a future admin
 * endpoint can spawn it exactly the way `creatorOnboardingPipeline.service.ts`
 * spawns the onboarding pipeline today — i.e. this is "#2" built in the
 * "#3 (button)-ready" shape.
 *
 * The backend must already be reachable at `--api-base` (this script lists
 * creators via the API and does NOT auto-start servers). When spawned by the
 * backend itself, that is true by construction; from a terminal, start the dev
 * backend first.
 *
 * Node built-ins only — no npm dependencies — so a detached child runs under a
 * stock Node (`process.execPath`) on macOS/Linux/Windows.
 *
 * Usage:
 * node scripts/update_all_creators.mjs \
 * [--api-base http://localhost:4000/api] [--admin-pin ... | env THOUGHTTRACKER_ADMIN_PIN] \
 * [--limit 0] [--min-duration-seconds 60] [--throttle-seconds 3] \
 * [--import-timeout-seconds 1800] [--health-timeout-seconds 10] \
 * [--only <slug> ...] [--dry-run] [--skip-fetch] [--skip-ingest]
 *
 * `--only` is repeatable and restricts the run to the listed creator slugs.
 * `--dry-run` lists what would be refreshed without fetching/ingesting.
 */

import { spawnSync } from "node:child_process";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import http from "node:http";
import path from "node:path";
import process from "node:process";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

/** Absolute path to this script's directory. */
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
/** ML repo root (the spawner sets cwd here; we also resolve all paths off it). */
const ML_DIR = path.resolve(SCRIPT_DIR, "..");
/** Filesystem-safe run timestamp (no colons/dots) for per-run output files. */
const STAMP = new Date().toISOString().replace(/[:.]/g, "-");
/** Output directories + files (mirrors the add_creator_pipeline layout). */
const LOG_DIR = path.join(ML_DIR, "logs");
const REPORTS_DIR = path.join(ML_DIR, "reports", "metrics");
const TRANSCRIPT_ROOT = path.join(ML_DIR, "data", "transcripts");
/**
 * The pollable status file. A FUTURE backend endpoint returns this path in its
 * 202 body so a UI can read progress — distinct from add_creator's status file
 * so the two never collide.
 */
const STATUS_PATH = path.join(REPORTS_DIR, "update_all_creators_status.json");
const SUMMARY_PATH = path.join(
  REPORTS_DIR,
  `update_all_creators_summary_${STAMP}.md`,
);
const LOG_PATH = path.join(LOG_DIR, `update-all-creators-${STAMP}.log`);
/** Wall-clock start, used in status payloads + the summary. */
const STARTED_AT = new Date();

/**
 * Resolve the venv Python interpreter cross-platform (prefers `.venv`, falls
 * back to `.venv311`), matching add_creator_pipeline's resolution.
 * @returns {string}
 */
function resolvePython() {
  const candidates =
    process.platform === "win32"
      ? [path.join(ML_DIR, ".venv", "Scripts", "python.exe")]
      : [
          path.join(ML_DIR, ".venv", "bin", "python"),
          path.join(ML_DIR, ".venv311", "bin", "python"),
        ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

/**
 * Append a timestamped line to both the console and the run log.
 * @param {string} message
 * @returns {void}
 */
function writeLog(message) {
  const stamp = new Date().toISOString().replace("T", " ").slice(0, 19);
  const line = `${stamp} ${message}\n`;
  process.stdout.write(line);
  appendFileSync(LOG_PATH, line);
}

/** Default CLI options. */
const DEFAULTS = {
  apiBase: "http://localhost:4000/api",
  adminPin: "",
  limit: 0,
  minDurationSeconds: 60,
  throttleSeconds: 3.0,
  importTimeoutSeconds: 1800,
  healthTimeoutSeconds: 10,
  /** Repeatable creator-slug filter; empty = all creators. */
  only: /** @type {string[]} */ ([]),
  dryRun: false,
  skipFetch: false,
  skipIngest: false,
  /**
   * Accepted no-op for spawn-family compatibility: this script NEVER starts
   * servers (the backend must already be up), so the flag is implicitly always
   * on. Declared so a future endpoint reusing the onboarding spawn args won't
   * trip the unknown-flag guard.
   */
  noStartServers: false,
};

/** Flags that are boolean switches (consume no value). */
const SWITCHES = new Set([
  "dryRun",
  "skipFetch",
  "skipIngest",
  "noStartServers",
]);

/**
 * Map a `--kebab-case` flag to its camelCase option key.
 * @param {string} flag
 * @returns {string}
 */
function flagToKey(flag) {
  return flag
    .replace(/^--/, "")
    .replace(/-([a-z])/g, (_m, c) => c.toUpperCase());
}

/**
 * Parse argv into an options object. `--only` accumulates; switches consume no
 * value; numeric fields are coerced. Unknown flags throw so typos fail loudly.
 * @param {string[]} argv
 * @returns {typeof DEFAULTS}
 */
function parseArgs(argv) {
  const opts = { ...DEFAULTS, only: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const key = flagToKey(argv[i]);
    if (!(key in opts)) {
      throw new Error(`Unknown argument: ${argv[i]}`);
    }
    if (SWITCHES.has(key)) {
      /** @type {Record<string, unknown>} */ (opts)[key] = true;
      continue;
    }
    const raw = argv[(i += 1)];
    if (Array.isArray(/** @type {Record<string, unknown>} */ (opts)[key])) {
      /** @type {string[]} */ (
        /** @type {Record<string, unknown>} */ (opts)[key]
      ).push(raw);
    } else if (
      typeof (/** @type {Record<string, unknown>} */ (opts)[key]) === "number"
    ) {
      /** @type {Record<string, unknown>} */ (opts)[key] = Number(raw);
    } else {
      /** @type {Record<string, unknown>} */ (opts)[key] = raw;
    }
  }
  return opts;
}

/**
 * GET a URL and parse the JSON body. Rejects on non-2xx, connection error,
 * timeout, or invalid JSON. Node built-ins only.
 * @param {string} urlStr
 * @returns {Promise<any>}
 */
function httpGetJson(urlStr) {
  return new Promise((resolve, reject) => {
    const u = new URL(urlStr);
    const req = http.get(
      {
        host: u.hostname,
        port: u.port || 80,
        path: `${u.pathname}${u.search}`,
        timeout: 15000,
      },
      (res) => {
        const status = res.statusCode ?? 0;
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          if (status < 200 || status >= 300) {
            reject(new Error(`GET ${u.pathname} -> HTTP ${status}`));
            return;
          }
          try {
            resolve(JSON.parse(body));
          } catch (err) {
            reject(
              new Error(
                `GET ${u.pathname} -> invalid JSON: ${err instanceof Error ? err.message : String(err)}`,
              ),
            );
          }
        });
      },
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy(new Error(`GET ${u.pathname} timed out`));
    });
  });
}

/**
 * Probe the backend `/health` endpoint. Resolves true for any 2xx–4xx
 * response, false on connection error/timeout.
 * @param {string} apiBase
 * @returns {Promise<boolean>}
 */
function testBackendHealth(apiBase) {
  return new Promise((resolve) => {
    const url = new URL(`${apiBase.replace(/\/$/, "")}/health`);
    const req = http.get(
      {
        host: url.hostname,
        port: url.port || 80,
        path: url.pathname,
        timeout: 5000,
      },
      (res) => {
        res.resume();
        resolve(
          (res.statusCode ?? 500) >= 200 && (res.statusCode ?? 500) < 500,
        );
      },
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

/**
 * Wait up to `seconds` for the backend `/health` to come up, polling every 3s.
 * @param {string} apiBase
 * @param {number} seconds
 * @returns {Promise<boolean>}
 */
async function waitBackendHealth(apiBase, seconds) {
  const deadline = Date.now() + Math.max(0, seconds) * 1000;
  // Always try at least once even when seconds is 0.
  do {
    if (await testBackendHealth(apiBase)) {
      return true;
    }
    if (Date.now() >= deadline) {
      break;
    }
    await sleep(3000);
  } while (Date.now() < deadline);
  return false;
}

/**
 * @typedef {{ channelUrl: string, creatorName: string, creatorSlug: string, transcriptFolder: string }} CreatorSpec
 */

/**
 * Enumerate every existing creator via the backend API and resolve each one's
 * YouTube channel URL (from the creator-detail endpoint's `sourceChannels`).
 * GET endpoints need no auth. Returns the refreshable specs plus a list of
 * creators skipped (with a reason) so the summary can report them.
 * @param {string} apiBase
 * @param {string[]} onlySlugs
 * @returns {Promise<{ specs: CreatorSpec[], skipped: { slug: string, name: string, reason: string }[] }>}
 */
async function enumerateSpecs(apiBase, onlySlugs) {
  const base = apiBase.replace(/\/$/, "");
  const list = await httpGetJson(`${base}/creators`);
  const items = Array.isArray(list?.items) ? list.items : [];
  const only = new Set(
    onlySlugs.map((s) => s.trim().toLowerCase()).filter(Boolean),
  );
  /** @type {CreatorSpec[]} */
  const specs = [];
  /** @type {{ slug: string, name: string, reason: string }[]} */
  const skipped = [];

  for (const item of items) {
    const slug = String(item?.slug ?? "");
    const name = String(item?.name ?? slug);
    if (!slug) {
      continue;
    }
    if (only.size && !only.has(slug.toLowerCase())) {
      continue;
    }
    let detail;
    try {
      detail = await httpGetJson(
        `${base}/creators/${encodeURIComponent(slug)}`,
      );
    } catch (err) {
      skipped.push({
        slug,
        name,
        reason: `detail lookup failed: ${err instanceof Error ? err.message : String(err)}`,
      });
      continue;
    }
    const channels = Array.isArray(detail?.sourceChannels)
      ? detail.sourceChannels.filter(
          (sc) =>
            sc && typeof sc.channelUrl === "string" && sc.channelUrl.trim(),
        )
      : [];
    if (channels.length === 0) {
      skipped.push({
        slug,
        name,
        reason: "no source channel with a channelUrl (cannot re-fetch)",
      });
      continue;
    }
    if (channels.length > 1) {
      writeLog(
        `NOTE ${slug} has ${channels.length} source channels; refreshing only the first (${channels[0].channelUrl}).`,
      );
    }
    specs.push({
      channelUrl: channels[0].channelUrl.trim(),
      creatorName: name,
      creatorSlug: slug,
      // Mirror add_creator_pipeline: <ML_DIR>/data/transcripts/<slug>. The slug
      // matches the original ingest, so fetch finds the existing folder and
      // skips already-downloaded videos.
      transcriptFolder: path.join(TRANSCRIPT_ROOT, slug),
    });
  }

  // Surface any --only slugs that matched no creator.
  if (only.size) {
    const found = new Set(specs.map((s) => s.creatorSlug.toLowerCase()));
    for (const wanted of only) {
      if (
        !found.has(wanted) &&
        !skipped.some((s) => s.slug.toLowerCase() === wanted)
      ) {
        skipped.push({
          slug: wanted,
          name: wanted,
          reason: "--only slug not found among existing creators",
        });
      }
    }
  }

  return { specs, skipped };
}

/**
 * Write the pollable status JSON and log a one-line status update. The field
 * set mirrors add_creator_pipeline's status so a UI poller can be shared.
 * @param {object} fields
 * @param {string} fields.stage
 * @param {string} fields.message
 * @param {CreatorSpec[]} fields.specs
 * @param {string} [fields.eta]
 * @param {boolean} [fields.done]
 * @param {string} [fields.activeOutput]
 * @param {string} [fields.needsUser]
 * @returns {void}
 */
function writeStatus({
  stage,
  message,
  specs,
  eta = "",
  done = false,
  activeOutput = "",
  needsUser = "",
}) {
  const status = {
    stage,
    message,
    eta,
    done,
    needsUser,
    startedAt: STARTED_AT.toISOString(),
    updatedAt: new Date().toISOString(),
    logPath: LOG_PATH,
    summaryPath: SUMMARY_PATH,
    statusPath: STATUS_PATH,
    activeOutput,
    creators: specs.map((s) => ({
      channelUrl: s.channelUrl,
      creatorName: s.creatorName,
      creatorSlug: s.creatorSlug,
      transcriptFolder: s.transcriptFolder,
    })),
  };
  writeFileSync(STATUS_PATH, JSON.stringify(status, null, 2));
  writeLog(`STATUS [${stage}] ${message}${eta ? ` ETA=${eta}` : ""}`);
}

/**
 * Run a native command, redacting any `--admin-pin` value in the logged
 * command line, streaming output, and throwing on failure.
 * @param {string} command
 * @param {string[]} args
 * @param {string} cwd
 * @returns {void}
 */
function invokeNative(command, args, cwd) {
  const display = args.slice();
  for (let i = 0; i < display.length; i += 1) {
    if (display[i] === "--admin-pin" && i + 1 < display.length) {
      display[i + 1] = "<redacted>";
    }
  }
  writeLog(`RUN ${command} ${display.join(" ")}`);
  const result = spawnSync(command, args, {
    cwd,
    stdio: "inherit",
    env: process.env,
  });
  if (result.error) {
    throw new Error(
      `Command failed to launch: ${command} (${result.error.message})`,
    );
  }
  if (result.status !== 0) {
    throw new Error(
      `Command failed with exit code ${result.status}: ${command} ${display.join(" ")}`,
    );
  }
}

/**
 * Build the `fetch_transcripts.py` argv for one creator (cwd = ML_DIR).
 * @param {CreatorSpec} spec
 * @param {typeof DEFAULTS} opts
 * @returns {string[]}
 */
function fetchArgs(spec, opts) {
  return [
    path.join("scripts", "fetch_transcripts.py"),
    "--channel",
    spec.channelUrl,
    "--creator-name",
    spec.creatorName,
    "--creator-slug",
    spec.creatorSlug,
    "--out-dir",
    path.join("data", "transcripts"),
    "--limit",
    String(opts.limit),
    "--min-duration",
    String(opts.minDurationSeconds),
    "--throttle",
    String(opts.throttleSeconds),
  ];
}

/**
 * Build the `ingest_transcripts.py` argv for one creator (cwd = ML_DIR).
 * @param {CreatorSpec} spec
 * @param {typeof DEFAULTS} opts
 * @returns {string[]}
 */
function ingestArgs(spec, opts) {
  return [
    path.join("scripts", "ingest_transcripts.py"),
    "--folder",
    spec.transcriptFolder,
    "--creator-name",
    spec.creatorName,
    "--creator-slug",
    spec.creatorSlug,
    "--channel-url",
    spec.channelUrl,
    "--api-base",
    opts.apiBase,
    "--admin-pin",
    opts.adminPin,
    "--timeout",
    String(opts.importTimeoutSeconds),
    "--poll-interval",
    "5",
  ];
}

/**
 * Count NEW transcripts a fetch produced for a creator by reading the
 * per-creator `_manifest.json`. New = a "saved" entry that is NOT flagged
 * `skipReason="already_on_disk"` (those are pre-existing files re-listed on a
 * resumable run). Returns the count, or `null` if the manifest is missing or
 * unreadable (caller treats null as "unknown → ingest to be safe").
 * @param {string} transcriptFolder
 * @returns {number | null}
 */
function countNewVideos(transcriptFolder) {
  const manifestPath = path.join(transcriptFolder, "_manifest.json");
  if (!existsSync(manifestPath)) {
    return null;
  }
  try {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    const entries = Array.isArray(manifest?.entries) ? manifest.entries : [];
    return entries.filter(
      (e) => e && e.status === "saved" && e.skipReason !== "already_on_disk",
    ).length;
  } catch {
    return null;
  }
}

/**
 * Write the run summary markdown.
 * @param {CreatorSpec[]} specs
 * @param {{ creatorSlug: string, creatorName: string, status: string, detail?: string, error?: string }[]} results
 * @param {{ slug: string, name: string, reason: string }[]} skipped
 * @param {typeof DEFAULTS} opts
 * @returns {void}
 */
function writeSummary(specs, results, skipped, opts) {
  const failed = results.filter((r) => r.status === "failed");
  const refreshed = results.filter((r) => r.status === "refreshed");
  const upToDate = results.filter((r) => r.status === "up-to-date");
  const lines = [
    "# Update All Creators — Refresh Summary",
    "",
    `- Started: ${STARTED_AT.toISOString()}`,
    `- Finished: ${new Date().toISOString()}`,
    `- API base: ${opts.apiBase}`,
    `- Mode: ${opts.dryRun ? "dry run (no fetch/ingest)" : "routine refresh (fetch new uploads; ingest only creators that have new uploads)"}`,
    `- Creators considered: ${specs.length}`,
    `- Refreshed: ${refreshed.length} · Up-to-date: ${upToDate.length} · Failed: ${failed.length} · Skipped: ${skipped.length}`,
    `- Log: ${LOG_PATH}`,
    `- Status: ${STATUS_PATH}`,
    "",
    "## Results",
    "",
    ...results.map((r) => {
      if (r.status === "failed") {
        return `- ❌ ${r.creatorName} \`${r.creatorSlug}\`: ${r.error ?? "failed"}`;
      }
      const icon =
        r.status === "up-to-date" ? "◦" : r.status === "dry-run" ? "•" : "✅";
      return `- ${icon} ${r.creatorName} \`${r.creatorSlug}\`: ${r.status}${r.detail ? ` (${r.detail})` : ""}`;
    }),
  ];
  if (skipped.length) {
    lines.push(
      "",
      "## Skipped",
      "",
      ...skipped.map((s) => `- ${s.name} \`${s.slug}\`: ${s.reason}`),
    );
  }
  lines.push(
    "",
    "## Notes",
    "",
    "- Creators with NO new uploads are skipped entirely (no re-import, no re-analysis).",
    "- Creators WITH new uploads have their whole folder re-ingested; new videos are analyzed automatically, but bulk-import currently also re-chunks + re-analyzes that creator's existing videos (a known cost — making it fully incremental needs a bulk-import change to skip already-completed videos).",
    "- This does NOT run a full-database reanalysis. If you change the frozen topic/stance policy and want every existing video recomputed, run the dedicated reanalysis separately.",
  );
  writeFileSync(SUMMARY_PATH, lines.join("\n"));
}

/**
 * Orchestrate the incremental refresh of all creators. Returns an exit code
 * (0 = all refreshed / nothing to do / dry run; 1 = a fatal setup error or at
 * least one creator failed). Failures are written to the status JSON before
 * exit so a fire-and-forget spawner can surface them.
 * @returns {Promise<number>}
 */
async function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(
      `${err instanceof Error ? err.message : String(err)}\n`,
    );
    return 1;
  }
  if (!opts.adminPin && process.env.THOUGHTTRACKER_ADMIN_PIN) {
    opts.adminPin = process.env.THOUGHTTRACKER_ADMIN_PIN;
  }

  for (const dir of [LOG_DIR, REPORTS_DIR, TRANSCRIPT_ROOT]) {
    mkdirSync(dir, { recursive: true });
  }

  writeStatus({
    stage: "starting",
    message: "Preparing incremental refresh of all creators.",
    eta: "minutes to hours depending on how many new uploads exist",
    activeOutput: LOG_PATH,
    specs: [],
  });

  const python = resolvePython();
  if (process.platform !== "win32" && !existsSync(python) && !opts.skipFetch) {
    writeStatus({
      stage: "starting",
      message: `ML Python venv not found at ${python}`,
      needsUser:
        "Create the ML virtualenv (.venv or .venv311) before running, or pass --skip-fetch to ingest existing transcripts only.",
      done: true,
      specs: [],
    });
    return 1;
  }

  // The backend must already be reachable — we enumerate creators via its API
  // and deliberately do NOT auto-start servers (a future endpoint runs us from
  // inside the live backend; a terminal user starts it first).
  if (!(await waitBackendHealth(opts.apiBase, opts.healthTimeoutSeconds))) {
    writeStatus({
      stage: "enumerate",
      message: `Backend is not reachable at ${opts.apiBase}.`,
      needsUser:
        "Start the backend (npm run dev:backend) and retry, or pass a correct --api-base.",
      done: true,
      specs: [],
    });
    return 1;
  }

  writeStatus({
    stage: "enumerate",
    message: "Listing creators and resolving channel URLs.",
    activeOutput: LOG_PATH,
    specs: [],
  });
  let specs;
  let skipped;
  try {
    ({ specs, skipped } = await enumerateSpecs(opts.apiBase, opts.only));
  } catch (err) {
    writeStatus({
      stage: "enumerate",
      message: `Failed to enumerate creators: ${err instanceof Error ? err.message : String(err)}`,
      needsUser: "Check that GET <api-base>/creators responds.",
      done: true,
      specs: [],
    });
    return 1;
  }
  for (const s of skipped) {
    writeLog(`SKIP ${s.slug}: ${s.reason}`);
  }

  if (specs.length === 0) {
    writeStatus({
      stage: "complete",
      message: "No creators with a YouTube channel URL to refresh.",
      eta: "complete",
      done: true,
      specs: [],
    });
    writeSummary([], [], skipped, opts);
    return 0;
  }

  if (opts.dryRun) {
    writeStatus({
      stage: "complete",
      message: `Dry run: ${specs.length} creator(s) would be refreshed.`,
      eta: "complete",
      done: true,
      specs,
    });
    writeSummary(
      specs,
      specs.map((s) => ({
        creatorSlug: s.creatorSlug,
        creatorName: s.creatorName,
        status: "dry-run",
      })),
      skipped,
      opts,
    );
    writeLog(
      `DRY RUN — would refresh ${specs.length} creator(s): ${specs.map((s) => s.creatorSlug).join(", ")}`,
    );
    return 0;
  }

  /** @type {{ creatorSlug: string, creatorName: string, status: string, error?: string }[]} */
  const results = [];
  for (let i = 0; i < specs.length; i += 1) {
    const spec = specs[i];
    writeStatus({
      stage: "refresh",
      message: `Refreshing ${spec.creatorName} (${i + 1}/${specs.length}).`,
      eta: `${specs.length - i} creator(s) remaining`,
      activeOutput: LOG_PATH,
      specs,
    });
    try {
      let newCount = /** @type {number | null} */ (null);
      if (!opts.skipFetch) {
        if (!existsSync(spec.transcriptFolder)) {
          // No local folder for this slug yet → fetch will download the WHOLE
          // channel (not just new uploads). Happens for a creator first
          // refreshed on this machine, or one whose DB slug doesn't match the
          // ML transcript-folder slug. Surface it rather than silently churning.
          writeLog(
            `NOTE ${spec.creatorSlug}: no local transcript folder — fetching the full channel.`,
          );
        }
        invokeNative(python, fetchArgs(spec, opts), ML_DIR);
        newCount = countNewVideos(spec.transcriptFolder);
      }
      if (opts.skipIngest) {
        results.push({
          creatorSlug: spec.creatorSlug,
          creatorName: spec.creatorName,
          status: "fetched-only",
        });
        writeLog(`FETCH-ONLY ${spec.creatorSlug} (ingest skipped)`);
      } else if (newCount === 0) {
        // Fetched and nothing new → skip the ingest so we don't re-import (and
        // thereby re-analyze) this creator's unchanged back-catalog.
        results.push({
          creatorSlug: spec.creatorSlug,
          creatorName: spec.creatorName,
          status: "up-to-date",
        });
        writeLog(`UP-TO-DATE ${spec.creatorSlug} (no new uploads)`);
      } else {
        invokeNative(python, ingestArgs(spec, opts), ML_DIR);
        const detail =
          newCount === null
            ? "ingested (new-count unknown)"
            : `${newCount} new video(s)`;
        results.push({
          creatorSlug: spec.creatorSlug,
          creatorName: spec.creatorName,
          status: "refreshed",
          detail,
        });
        writeLog(`REFRESHED ${spec.creatorSlug} (${detail})`);
      }
    } catch (err) {
      // Per-creator resilience: one bad channel must not abort the rest.
      const msg = err instanceof Error ? err.message : String(err);
      results.push({
        creatorSlug: spec.creatorSlug,
        creatorName: spec.creatorName,
        status: "failed",
        error: msg,
      });
      writeLog(`ERROR ${spec.creatorSlug}: ${msg}`);
    }
  }

  const failed = results.filter((r) => r.status === "failed");
  writeSummary(specs, results, skipped, opts);
  writeStatus({
    stage: "complete",
    message: failed.length
      ? `Refreshed ${results.length - failed.length}/${results.length} creators; ${failed.length} failed.`
      : `Refreshed all ${results.length} creators.`,
    eta: "complete",
    done: true,
    needsUser: failed.length
      ? `Failed: ${failed.map((f) => f.creatorSlug).join(", ")}. Review the log and rerun (already-fetched videos are skipped).`
      : "",
    activeOutput: LOG_PATH,
    specs,
  });
  return failed.length ? 1 : 0;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    // Last-resort guard: record the failure to the status file before exiting
    // so a fire-and-forget spawner sees a terminal, diagnosable state.
    try {
      writeStatus({
        stage: "complete",
        message: `Fatal error: ${err instanceof Error ? err.message : String(err)}`,
        needsUser: "Review the log; this is an unexpected failure.",
        done: true,
        specs: [],
      });
    } catch {
      // Status write itself failed — fall through to stderr + nonzero exit.
    }
    process.stderr.write(`${err instanceof Error ? err.stack : String(err)}\n`);
    process.exit(1);
  });
