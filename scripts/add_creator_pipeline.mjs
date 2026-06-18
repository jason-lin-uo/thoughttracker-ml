#!/usr/bin/env node
// @ts-check
/**
 * add_creator_pipeline.mjs — one-command creator onboarding workflow.
 *
 * Cross-platform Node port of the legacy Windows-only shell wrapper.
 * Given one or more channel URLs it: ensures the backend is up, fetches
 * transcripts, ingests them, reanalyzes against the frozen policy, runs a
 * quality audit, and (optionally) builds an owner-review packet. Progress
 * is mirrored to a JSON status file the UI can poll, and to a run log.
 *
 * Node built-ins only — no npm dependencies — so the backend (and a human
 * on macOS/Linux/Windows) can spawn it with a stock Node install.
 *
 * Usage:
 * node scripts/add_creator_pipeline.mjs \
 * --channel-url https://www.youtube.com/@somechannel \
 * [--creator-name "Some Channel"] [--creator-slug somechannel] \
 * [--limit 0] [--api-base http://localhost:4000/api] [--admin-pin ...] \
 * [--concurrency 5] [--skip-fetch] [--skip-ingest] [--skip-reanalysis] \
 * [--skip-packet] [--no-start-servers]
 *
 * `--channel-url` is repeatable; `--creator-name` / `--creator-slug`
 * positionally align with the channel URLs in order.
 */

import { spawn, spawnSync } from "node:child_process";
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
/** ML repo root. */
const ML_DIR = path.resolve(SCRIPT_DIR, "..");
/** Parent projects dir + main app + backend. */
const PROJECTS_DIR = path.resolve(ML_DIR, "..");
const APP_DIR = path.join(PROJECTS_DIR, "thoughttracker");
const BACKEND_DIR = path.join(APP_DIR, "backend");
/** Filesystem-safe run timestamp. */
const STAMP = new Date().toISOString().replace(/[:.]/g, "-");
/** Output directories + files (mirrors the PowerShell layout). */
const LOG_DIR = path.join(ML_DIR, "logs");
const REPORTS_DIR = path.join(ML_DIR, "reports", "metrics");
const TRANSCRIPT_ROOT = path.join(ML_DIR, "data", "transcripts");
const STATUS_PATH = path.join(REPORTS_DIR, "add_creator_pipeline_status.json");
const SUMMARY_PATH = path.join(
  REPORTS_DIR,
  `add_creator_pipeline_summary_${STAMP}.md`,
);
const LOG_PATH = path.join(LOG_DIR, `add-creator-pipeline-${STAMP}.log`);
const QUALITY_AUDIT_PATH = path.join(
  LOG_DIR,
  `add-creator-quality-audit-${STAMP}.md`,
);
const PACKET_DIR = path.join(
  ML_DIR,
  "data",
  "labeling",
  `creator_onboarding_review_${STAMP}`,
);
/** Wall-clock start, used in status payloads + the summary. */
const STARTED_AT = new Date();

/**
 * Resolve the venv Python interpreter cross-platform.
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
 * Platform-appropriate npm-style binary name (`.cmd` suffix on Windows).
 * @param {string} name
 * @returns {string}
 */
function platformBin(name) {
  return process.platform === "win32" ? `${name}.cmd` : name;
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

/**
 * Default CLI options. `channelUrl`/`creatorName`/`creatorSlug` are
 * parallel arrays aligned by index, matching the PowerShell `[string[]]`
 * parameters.
 */
const DEFAULTS = {
  channelUrl: /** @type {string[]} */ ([]),
  creatorName: /** @type {string[]} */ ([]),
  creatorSlug: /** @type {string[]} */ ([]),
  limit: 0,
  minDurationSeconds: 60,
  throttleSeconds: 3.0,
  apiBase: "http://localhost:4000/api",
  adminPin: "",
  concurrency: 5,
  importTimeoutSeconds: 1800,
  packetMaxRows: 500,
  packetMaxConfidence: 0.72,
  packetMinRelevance: 0.35,
  skipFetch: false,
  skipIngest: false,
  skipReanalysis: false,
  skipPacket: false,
  noStartServers: false,
};

/** Flags that are boolean switches (consume no value). */
const SWITCHES = new Set([
  "skipFetch",
  "skipIngest",
  "skipReanalysis",
  "skipPacket",
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
 * Parse argv into an options object. Array-valued flags
 * (`--channel-url`, `--creator-name`, `--creator-slug`) accumulate;
 * switches consume no value; numeric fields are coerced. Unknown flags
 * throw so typos fail loudly.
 * @param {string[]} argv
 * @returns {typeof DEFAULTS}
 */
function parseArgs(argv) {
  const opts = {
    ...DEFAULTS,
    channelUrl: [],
    creatorName: [],
    creatorSlug: [],
  };
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
 * Derive a URL-safe slug from a creator name or channel handle.
 * Lowercases, strips protocol/host/`@`, and collapses non-alphanumerics
 * to single dashes. Falls back to ``creator-<n>`` for empty input.
 * @param {string} value
 * @param {number} fallbackIndex 1-based index for the fallback name.
 * @returns {string}
 */
function toSlug(value, fallbackIndex) {
  let candidate = (value ?? "").trim();
  if (!candidate) {
    return `creator-${fallbackIndex}`;
  }
  candidate = candidate
    .toLowerCase()
    .replace(/https?:\/\//g, "")
    .replace(/www\./g, "")
    .replace(/^youtube\.com\//, "")
    .replace(/^@/, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return candidate || `creator-${fallbackIndex}`;
}

/**
 * Extract a handle from a channel URL: the `@handle` if present, else the
 * last path segment. Falls back to ``creator-<n>``.
 * @param {string} url
 * @param {number} fallbackIndex
 * @returns {string}
 */
function urlHandle(url, fallbackIndex) {
  const at = url.match(/@([^/?#]+)/);
  if (at) {
    return at[1];
  }
  try {
    const parsed = new URL(url);
    const segments = parsed.pathname.split("/").filter(Boolean);
    if (segments.length) {
      return segments[segments.length - 1];
    }
  } catch {
    // Not a parseable URL (e.g. a bare @handle) — use the fallback below.
  }
  return `creator-${fallbackIndex}`;
}

/**
 * Title-case a handle-derived name (dashes/underscores → spaces).
 * @param {string} value
 * @returns {string}
 */
function toDisplayName(value) {
  const words = value
    .replace(/[-_]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (words.length === 0) {
    return "Creator";
  }
  return words
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

/**
 * Build the per-creator spec list (URL, name, slug, transcript folder)
 * from the parallel CLI arrays. Throws if no non-empty channel URL given.
 * @param {typeof DEFAULTS} opts
 * @returns {{ channelUrl: string, creatorName: string, creatorSlug: string, transcriptFolder: string }[]}
 */
function getCreatorSpecs(opts) {
  const specs = [];
  for (let i = 0; i < opts.channelUrl.length; i += 1) {
    const url = opts.channelUrl[i].trim();
    if (!url) {
      continue;
    }
    const handle = urlHandle(url, i + 1);
    const name = opts.creatorName[i]?.trim() || toDisplayName(handle);
    const slugSource = opts.creatorSlug[i] || handle;
    const slug = toSlug(slugSource, i + 1);
    specs.push({
      channelUrl: url,
      creatorName: name,
      creatorSlug: slug,
      transcriptFolder: path.join(TRANSCRIPT_ROOT, slug),
    });
  }
  if (specs.length === 0) {
    throw new Error("At least one non-empty --channel-url value is required.");
  }
  return specs;
}

/**
 * Write the pollable status JSON and log a one-line status update.
 * @param {object} fields
 * @param {string} fields.stage Short stage id.
 * @param {string} fields.message Human-readable message.
 * @param {{ channelUrl: string, creatorName: string, creatorSlug: string, transcriptFolder: string }[]} fields.specs
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
    packetDir: PACKET_DIR,
    activeOutput,
    creators: specs.map((s) => ({
      channelUrl: s.channelUrl,
      creatorName: s.creatorName,
      creatorSlug: s.creatorSlug,
      transcriptFolder: s.transcriptFolder,
    })),
  };
  writeFileSync(STATUS_PATH, JSON.stringify(status, null, 2));
  writeLog(`STATUS [${stage}] ${message} ETA=${eta}`);
}

/**
 * Run a named native command, redacting any `--admin-pin` value in the
 * logged command line, streaming output, and throwing on failure.
 * @param {string} command
 * @param {string[]} args
 * @param {string} cwd
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {void}
 */
function invokeNative(command, args, cwd, env) {
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
    env: env ?? process.env,
    shell: process.platform === "win32",
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
 * Probe the backend `/health` endpoint via the configured API base.
 * Resolves true for any 2xx–4xx response, false on connection error.
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
 * Wait up to `seconds` for the backend `/health` to come up, polling
 * every 3 seconds.
 * @param {string} apiBase
 * @param {number} seconds
 * @returns {Promise<boolean>}
 */
async function waitBackendHealth(apiBase, seconds) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (await testBackendHealth(apiBase)) {
      return true;
    }
    await sleep(3000);
  }
  return false;
}

/**
 * Ensure the backend is reachable, starting Docker + the dev server if
 * needed (unless `--no-start-servers`). Throws if it can't be made healthy.
 * @param {typeof DEFAULTS} opts
 * @returns {Promise<void>}
 */
async function startAppIfNeeded(opts) {
  if (await testBackendHealth(opts.apiBase)) {
    writeLog(`Backend already healthy at ${opts.apiBase}.`);
    return;
  }
  if (opts.noStartServers) {
    throw new Error(
      `Backend is not healthy at ${opts.apiBase} and --no-start-servers was provided.`,
    );
  }

  writeLog("Starting Docker services.");
  invokeNative("docker", ["compose", "up", "-d"], APP_DIR);
  if (await testBackendHealth(opts.apiBase)) {
    return;
  }

  writeLog("Starting backend dev server in the background.");
  const child = spawn(platformBin("npm"), ["run", "dev:backend"], {
    cwd: APP_DIR,
    detached: true,
    stdio: "ignore",
    shell: process.platform === "win32",
  });
  child.unref();

  if (!(await waitBackendHealth(opts.apiBase, 120))) {
    throw new Error("Backend did not become healthy within 120 seconds.");
  }
  writeLog(`Backend is healthy at ${opts.apiBase}.`);
}

/**
 * Run a pipeline step with status bookkeeping. Emits a "Starting" status,
 * runs `fn`, then emits "Finished"; on error it records a needs-user
 * status and re-throws so the run halts.
 * @param {string} name
 * @param {string} eta
 * @param {() => void | Promise<void>} fn
 * @param {{ channelUrl: string, creatorName: string, creatorSlug: string, transcriptFolder: string }[]} specs
 * @returns {Promise<void>}
 */
async function invokeStep(name, eta, fn, specs) {
  writeStatus({
    stage: name,
    message: `Starting ${name}.`,
    eta,
    activeOutput: LOG_PATH,
    specs,
  });
  try {
    await fn();
  } catch (err) {
    writeStatus({
      stage: name,
      message: `Failed during ${name}: ${err instanceof Error ? err.message : String(err)}`,
      activeOutput: LOG_PATH,
      needsUser: "Review the log and rerun this script after fixing the error.",
      specs,
    });
    throw err;
  }
  writeStatus({
    stage: name,
    message: `Finished ${name}.`,
    eta: "complete",
    activeOutput: LOG_PATH,
    specs,
  });
}

/**
 * Path to the backend's local `tsx` launcher.
 * @returns {string}
 */
function tsxBin() {
  return path.join(APP_DIR, "node_modules", ".bin", platformBin("tsx"));
}

/**
 * Write the run summary markdown. Notes the decision, key paths, the
 * creator roster, and the recommended next step (packet review).
 * @param {{ channelUrl: string, creatorName: string, creatorSlug: string, transcriptFolder: string }[]} specs
 * @param {string} packetRows
 * @param {boolean} skipPacket
 * @returns {void}
 */
function writeSummary(specs, packetRows, skipPacket) {
  const lines = [
    "# Add Creator Pipeline Summary",
    "",
    `- Started: ${STARTED_AT.toISOString()}`,
    `- Finished: ${new Date().toISOString()}`,
    "- Decision: Pipeline finished through current-model analysis and owner review handoff.",
    `- Log: ${LOG_PATH}`,
    `- Status: ${STATUS_PATH}`,
    `- Quality audit: ${QUALITY_AUDIT_PATH}`,
    `- Packet folder: ${PACKET_DIR}`,
    `- Packet rows: ${packetRows}`,
    "",
    "## Creators",
    "",
    ...specs.map(
      (s) => `- ${s.creatorName} \`${s.creatorSlug}\`: ${s.channelUrl}`,
    ),
    "",
    "## Next Step",
    "",
  ];
  if (skipPacket) {
    lines.push(
      "Review-packet generation was skipped. Run the backend packet builder later if owner review is needed.",
    );
  } else if (
    existsSync(path.join(PACKET_DIR, "creator_onboarding_review_input.jsonl"))
  ) {
    lines.push(
      "Review the packet folder before promoting new labels or recalibration changes.",
    );
  } else {
    lines.push("No packet file was produced. Check the log for details.");
  }
  writeFileSync(SUMMARY_PATH, lines.join("\n"));
}

/**
 * Orchestrate the full onboarding pipeline. Returns a process exit code.
 * @returns {Promise<number>}
 */
async function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
    if (!opts.adminPin && process.env.THOUGHTTRACKER_ADMIN_PIN) {
      opts.adminPin = process.env.THOUGHTTRACKER_ADMIN_PIN;
    }
  } catch (err) {
    process.stderr.write(
      `${err instanceof Error ? err.message : String(err)}\n`,
    );
    return 1;
  }

  for (const dir of [LOG_DIR, REPORTS_DIR, TRANSCRIPT_ROOT]) {
    mkdirSync(dir, { recursive: true });
  }

  let specs;
  try {
    specs = getCreatorSpecs(opts);
  } catch (err) {
    process.stderr.write(
      `${err instanceof Error ? err.message : String(err)}\n`,
    );
    return 1;
  }

  const python = resolvePython();
  let packetRows = "skipped";

  writeStatus({
    stage: "starting",
    message: "Preparing creator onboarding pipeline.",
    eta: "about 30-120 minutes depending on transcript volume and reanalysis settings",
    activeOutput: LOG_PATH,
    specs,
  });

  if (process.platform !== "win32" && !existsSync(python)) {
    process.stderr.write(`ML Python venv not found at ${python}\n`);
    return 1;
  }

  try {
    await invokeStep(
      "ensure_services",
      "1-3 minutes",
      () => startAppIfNeeded(opts),
      specs,
    );

    if (!opts.skipFetch) {
      await invokeStep(
        "fetch_transcripts",
        "depends on creator size; usually minutes to hours for full channels",
        () => {
          for (const spec of specs) {
            invokeNative(
              python,
              [
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
              ],
              ML_DIR,
            );
          }
        },
        specs,
      );
    }

    if (!opts.skipIngest) {
      await invokeStep(
        "ingest_transcripts",
        "5-30 minutes depending on transcript count",
        () => {
          for (const spec of specs) {
            invokeNative(
              python,
              [
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
              ],
              ML_DIR,
            );
          }
        },
        specs,
      );
    }

    if (!opts.skipReanalysis) {
      await invokeStep(
        "reanalyze_current_model",
        "often 30-120+ minutes on the full database",
        () => {
          invokeNative(
            process.execPath,
            [
              path.join("scripts", "run_reanalyze_latest_model.mjs"),
              "--concurrency",
              String(opts.concurrency),
              "--topic-assignment-provider",
              "final_policy",
              "--topic-selection-policy-path",
              path.join(
                "models",
                "topic-selection-policy-gold-standard",
                "policy.json",
              ),
              "--topic-relevance-model-dir",
              path.join(
                "models",
                "topic-relevance-classifier-supervalidation-hardneg2x-l512",
              ),
              "--topic-relevance-model-version",
              "topic-relevance-supervalidation-hardneg2x-l512",
              "--topic-relevance-max-length",
              "512",
              "--topic-reranker-model-dir",
              path.join("models", "topic-reranker-tfidf-sgd-supervalidation"),
              "--topic-reranker-limit",
              "12",
              "--topic-reranker-min-score",
              "0",
            ],
            ML_DIR,
          );
        },
        specs,
      );
    }

    await invokeStep(
      "quality_audit",
      "1-3 minutes",
      () => {
        invokeNative(
          tsxBin(),
          [
            path.join("scripts", "report-quality-audit.ts"),
            "--ml-dir",
            ML_DIR,
            "--out",
            QUALITY_AUDIT_PATH,
          ],
          BACKEND_DIR,
        );
      },
      specs,
    );

    if (!opts.skipPacket) {
      await invokeStep(
        "build_review_packet",
        "1-5 minutes",
        () => {
          // Comma-joined creator slugs for the packet builder's --creator-slugs.
          const slugs = specs.map((s) => s.creatorSlug).join(",");
          invokeNative(
            tsxBin(),
            [
              path.join("scripts", "build-creator-onboarding-packet.ts"),
              "--creator-slugs",
              slugs,
              "--out-dir",
              PACKET_DIR,
              "--max-rows",
              String(opts.packetMaxRows),
              "--max-confidence",
              String(opts.packetMaxConfidence),
              "--min-relevance",
              String(opts.packetMinRelevance),
            ],
            BACKEND_DIR,
          );
          const summaryFile = path.join(PACKET_DIR, "packet_summary.json");
          if (existsSync(summaryFile)) {
            const summary = JSON.parse(readFileSync(summaryFile, "utf8"));
            packetRows = String(summary.rows);
          }
        },
        specs,
      );
    }

    writeSummary(specs, packetRows, opts.skipPacket);
    writeStatus({
      stage: "complete",
      message:
        "Creator onboarding pipeline finished through current-model analysis and owner review handoff.",
      eta: "complete",
      done: true,
      activeOutput: SUMMARY_PATH,
      needsUser:
        "If packet rows are greater than 0, review the packet before promoting new labels or recalibration changes.",
      specs,
    });
    return 0;
  } catch (err) {
    writeLog(`[FATAL] ${err instanceof Error ? err.message : String(err)}`);
    return 1;
  }
}

main().then((code) => process.exit(code));
