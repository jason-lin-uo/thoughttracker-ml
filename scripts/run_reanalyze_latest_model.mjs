#!/usr/bin/env node
// @ts-check
/**
 * run_reanalyze_latest_model.mjs — start the ML API with the latest saved
 * model and reanalyze every video in the backend against the current
 * frozen topic/stance policy.
 *
 * Cross-platform Node port of the legacy Windows-only shell wrapper.
 * Steps, in order:
 * 1. Stop any ML API already listening on port 8000, then start a fresh
 * uvicorn process and wait for `/health`.
 * 2. `docker compose up -d` (Postgres) in the app repo.
 * 3. `npm run typecheck` in the backend.
 * 4. Reset derived analysis data (tsx script).
 * 5. Reanalyze all videos with `STANCE_ANALYSIS_PROVIDER=custom_ml`,
 * pointing the backend at the local ML API and the chosen models.
 *
 * Node built-ins only (no npm deps). Logs are written under
 * `<ml>/logs/` exactly as the legacy wrapper did.
 *
 * Usage:
 * node scripts/run_reanalyze_latest_model.mjs \
 * [--concurrency 5] [--topic-relevance-threshold 0.8] \
 * [--topic-assignment-provider final_policy] \
 * [--topic-relevance-model-dir <dir>] [--topic-reranker-model-dir <dir>] ...
 */

import { spawn, spawnSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, openSync } from "node:fs";
import http from "node:http";
import path from "node:path";
import process from "node:process";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

/** Absolute path to this script's directory. */
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
/** The ML repo root (one level above `scripts/`). */
const ML_DIR = path.resolve(SCRIPT_DIR, "..");
/** The parent "projects" directory containing both repos. */
const PROJECTS_DIR = path.resolve(ML_DIR, "..");
/** The main ThoughtTracker app + its backend. */
const APP_DIR = path.join(PROJECTS_DIR, "thoughttracker");
const BACKEND_DIR = path.join(APP_DIR, "backend");
/** A filesystem-safe timestamp for log file names. */
const STAMP = new Date().toISOString().replace(/[:.]/g, "-");
/** Where run logs are written. */
const LOG_DIR = path.join(ML_DIR, "logs");

/**
 * Resolve the venv Python interpreter cross-platform (see the matching
 * helper in `ingest_all_transcripts.mjs`).
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
 * Resolve the platform-appropriate command name for an npm-style binary.
 * On Windows the launcher is `<name>.cmd`; elsewhere it's `<name>`.
 * @param {string} name e.g. "npm".
 * @returns {string}
 */
function platformBin(name) {
  return process.platform === "win32" ? `${name}.cmd` : name;
}

/**
 * Default option set, overridden by CLI flags. Empty strings mean "leave
 * the corresponding env var unset" (matching the PowerShell defaults).
 */
const DEFAULTS = {
  concurrency: 5,
  topicRelevanceThreshold: 0.8,
  minStanceConfidence: 0.5,
  topicAssignmentProvider: "",
  topicSelectionPolicyPath: "",
  topicRelevanceModelDir: "",
  topicRelevanceModelVersion: "",
  topicRelevanceMaxLength: 512,
  topicRerankerLabelsPath: "",
  topicRerankerDisplayTiers: "",
  topicRerankerModelDir: "",
  topicRerankerLimit: 12,
  topicRerankerMinScore: 0.2,
};

/**
 * Map a `--kebab-case` flag to its camelCase option key.
 * @param {string} flag e.g. "--topic-relevance-threshold".
 * @returns {string} e.g. "topicRelevanceThreshold".
 */
function flagToKey(flag) {
  return flag
    .replace(/^--/, "")
    .replace(/-([a-z])/g, (_m, c) => c.toUpperCase());
}

/**
 * Parse argv into an options object, coercing numeric fields. Unknown
 * flags throw. Each known flag consumes the following token as its value.
 * @param {string[]} argv
 * @returns {typeof DEFAULTS}
 */
function parseArgs(argv) {
  const opts = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i += 1) {
    const key = flagToKey(argv[i]);
    if (!(key in opts)) {
      throw new Error(`Unknown argument: ${argv[i]}`);
    }
    const raw = argv[(i += 1)];
    const current = /** @type {Record<string, unknown>} */ (opts)[key];
    /** @type {Record<string, unknown>} */ (opts)[key] =
      typeof current === "number" ? Number(raw) : raw;
  }
  return opts;
}

/**
 * Resolve a possibly-relative path against the ML repo root.
 * @param {string} value Absolute or ML-relative path.
 * @returns {string} Absolute path.
 */
function resolveMlPath(value) {
  return path.isAbsolute(value) ? value : path.resolve(ML_DIR, value);
}

/**
 * Timestamped console log line.
 * @param {string} message
 * @returns {void}
 */
function logStep(message) {
  const stamp = new Date().toISOString().replace("T", " ").slice(0, 19);
  process.stdout.write(`\n[${stamp}] ${message}\n`);
}

/**
 * Probe the local ML API `/health` endpoint.
 * Resolves true for any 2xx–4xx status (the service is up), false on any
 * connection/timeout error. Uses `node:http` so there's no dependency.
 * @returns {Promise<boolean>}
 */
function testMlHealth() {
  return new Promise((resolve) => {
    const req = http.get(
      { host: "127.0.0.1", port: 8000, path: "/health", timeout: 5000 },
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
 * Best-effort: kill any process listening on port 8000 so we can start a
 * fresh ML API. Uses `lsof` on POSIX and `netstat`+`taskkill` on Windows.
 * Failures are swallowed — if nothing is listening, there's nothing to do.
 * @returns {void}
 */
function stopMlApiIfRunning() {
  if (process.platform === "win32") {
    const out = spawnSync("netstat", ["-ano"], { encoding: "utf8" });
    const pids = new Set();
    for (const line of (out.stdout || "").split(/\r?\n/)) {
      const match = line.match(/:8000\s+.*LISTENING\s+(\d+)/);
      if (match) {
        pids.add(match[1]);
      }
    }
    for (const pid of pids) {
      spawnSync("taskkill", ["/PID", pid, "/F"], { stdio: "ignore" });
    }
    return;
  }
  const out = spawnSync("lsof", ["-ti", "tcp:8000", "-sTCP:LISTEN"], {
    encoding: "utf8",
  });
  for (const pid of (out.stdout || "").split(/\s+/).filter(Boolean)) {
    if (pid !== String(process.pid)) {
      spawnSync("kill", ["-9", pid], { stdio: "ignore" });
    }
  }
}

/**
 * Start the ML API (uvicorn) as a detached background process and wait
 * for `/health` to come up (polls for up to ~120s).
 *
 * Topic-model env vars from `opts` are exported into the child's
 * environment so the running model matches the requested configuration.
 * Throws if the service doesn't become healthy in time.
 * @param {typeof DEFAULTS} opts
 * @returns {Promise<void>}
 */
async function startMlApi(opts) {
  const python = resolvePython();
  if (process.platform !== "win32" && !existsSync(python)) {
    throw new Error(`ML venv python not found at ${python}`);
  }

  const env = { ...process.env };
  if (opts.topicRelevanceModelDir) {
    env.TOPIC_RELEVANCE_MODEL_DIR = resolveMlPath(opts.topicRelevanceModelDir);
    logStep(`Topic relevance model dir: ${env.TOPIC_RELEVANCE_MODEL_DIR}`);
  }
  if (opts.topicRelevanceModelVersion) {
    env.TOPIC_RELEVANCE_MODEL_VERSION = opts.topicRelevanceModelVersion;
  }
  env.TOPIC_RELEVANCE_MAX_LENGTH = String(opts.topicRelevanceMaxLength);
  if (opts.topicRerankerModelDir) {
    env.TOPIC_RERANKER_MODEL_DIR = resolveMlPath(opts.topicRerankerModelDir);
    logStep(`Topic reranker model dir: ${env.TOPIC_RERANKER_MODEL_DIR}`);
  }

  const stdout = openSync(
    path.join(LOG_DIR, `ml-api-reanalyze-${STAMP}.out.log`),
    "a",
  );
  const stderr = openSync(
    path.join(LOG_DIR, `ml-api-reanalyze-${STAMP}.err.log`),
    "a",
  );
  logStep("Starting ML API with latest saved model");
  const child = spawn(
    python,
    [
      "-m",
      "uvicorn",
      "src.api.main:app",
      "--host",
      "127.0.0.1",
      "--port",
      "8000",
    ],
    { cwd: ML_DIR, env, detached: true, stdio: ["ignore", stdout, stderr] },
  );
  child.unref();

  for (let i = 0; i < 60; i += 1) {
    await sleep(2000);
    if (await testMlHealth()) {
      logStep("ML API is healthy");
      return;
    }
  }
  throw new Error("ML API did not become healthy within 120 seconds");
}

/**
 * Run one named step (a command + args in a working dir), streaming its
 * output. Throws if the command can't launch or exits non-zero.
 * @param {string} name Human-readable step label.
 * @param {string} cwd Working directory.
 * @param {string} command Executable.
 * @param {string[]} args Arguments.
 * @param {NodeJS.ProcessEnv} [env] Optional environment overrides.
 * @returns {void}
 */
function invokeStep(name, cwd, command, args, env) {
  logStep(`START ${name}`);
  const result = spawnSync(command, args, {
    cwd,
    stdio: "inherit",
    env: env ?? process.env,
    shell: process.platform === "win32",
  });
  if (result.error) {
    throw new Error(`${name} failed to launch: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(`${name} failed with exit code ${result.status}`);
  }
  logStep(`DONE ${name}`);
}

/**
 * Build the backend environment for the reanalyze step from `opts`,
 * exporting only the variables whose options are set (mirroring the
 * PowerShell conditionals) plus the fixed real-provider switches.
 * @param {typeof DEFAULTS} opts
 * @returns {NodeJS.ProcessEnv}
 */
function buildReanalyzeEnv(opts) {
  const env = { ...process.env };
  env.STANCE_ANALYSIS_PROVIDER = "custom_ml";
  if (opts.topicAssignmentProvider) {
    env.TOPIC_ASSIGNMENT_PROVIDER = opts.topicAssignmentProvider;
  }
  if (opts.topicSelectionPolicyPath) {
    env.TOPIC_SELECTION_POLICY_PATH = resolveMlPath(
      opts.topicSelectionPolicyPath,
    );
  }
  if (opts.topicRerankerLabelsPath) {
    env.TOPIC_RERANKER_LABELS_PATH = resolveMlPath(
      opts.topicRerankerLabelsPath,
    );
  }
  if (opts.topicRerankerDisplayTiers) {
    env.TOPIC_RERANKER_DISPLAY_TIERS = opts.topicRerankerDisplayTiers;
  }
  env.TOPIC_RELEVANCE_PROVIDER = "custom_ml";
  env.TOPIC_RELEVANCE_THRESHOLD = String(opts.topicRelevanceThreshold);
  env.MIN_STANCE_CONFIDENCE = String(opts.minStanceConfidence);
  if (opts.topicRelevanceModelDir) {
    env.TOPIC_RELEVANCE_MODEL_DIR = resolveMlPath(opts.topicRelevanceModelDir);
  }
  if (opts.topicRelevanceModelVersion) {
    env.TOPIC_RELEVANCE_MODEL_VERSION = opts.topicRelevanceModelVersion;
  }
  env.TOPIC_RELEVANCE_MAX_LENGTH = String(opts.topicRelevanceMaxLength);
  if (opts.topicRerankerModelDir) {
    env.TOPIC_RERANKER_MODEL_DIR = resolveMlPath(opts.topicRerankerModelDir);
  }
  env.TOPIC_RERANKER_LIMIT = String(opts.topicRerankerLimit);
  env.TOPIC_RERANKER_MIN_SCORE = String(opts.topicRerankerMinScore);
  env.ML_CLASSIFIER_URL = "http://127.0.0.1:8000";
  env.AI_PROVIDER = "local";
  env.EMBEDDING_PROVIDER = "ml";
  return env;
}

/**
 * Path to the backend's local `tsx` launcher, used to run the .ts helper
 * scripts (reset + reanalyze).
 * @returns {string}
 */
function tsxBin() {
  return path.join(APP_DIR, "node_modules", ".bin", platformBin("tsx"));
}

/**
 * Orchestrate the full reanalysis run. Returns a process exit code.
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

  mkdirSync(LOG_DIR, { recursive: true });
  const transcript = createWriteStream(
    path.join(LOG_DIR, `reanalyze-latest-model-${STAMP}.log`),
    { flags: "a" },
  );

  try {
    logStep("ThoughtTracker latest-model reanalysis runner");
    logStep(`Concurrency: ${opts.concurrency}`);
    logStep(`Topic relevance threshold: ${opts.topicRelevanceThreshold}`);

    stopMlApiIfRunning();
    await startMlApi(opts);

    invokeStep("Start Postgres with Docker Compose", APP_DIR, "docker", [
      "compose",
      "up",
      "-d",
    ]);
    invokeStep("Backend typecheck", BACKEND_DIR, platformBin("npm"), [
      "run",
      "typecheck",
    ]);
    invokeStep("Reset derived analysis only", BACKEND_DIR, tsxBin(), [
      path.join("scripts", "reset-analysis-derived-data.ts"),
    ]);
    invokeStep(
      "Reanalyze all videos with latest refined model",
      BACKEND_DIR,
      tsxBin(),
      [
        path.join("scripts", "reanalyze-all-videos.ts"),
        "--concurrency",
        String(opts.concurrency),
      ],
      buildReanalyzeEnv(opts),
    );

    logStep("Complete. Latest model reanalysis finished.");
    return 0;
  } catch (err) {
    logStep(`[FATAL] ${err instanceof Error ? err.message : String(err)}`);
    return 1;
  } finally {
    transcript.end();
  }
}

main().then((code) => process.exit(code));
