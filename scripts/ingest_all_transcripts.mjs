#!/usr/bin/env node
// @ts-check
/**
 * ingest_all_transcripts.mjs — bulk-ingest the known local transcript
 * folders into the ThoughtTracker backend.
 *
 * Cross-platform Node port of the legacy Windows-only shell wrapper.
 * It loops over the fixed real creator roster, verifies each one's
 * `_manifest.json` exists, and shells out to `scripts/ingest_transcripts.py`
 * (the single script that actually talks to the backend) for each.
 *
 * Why Node and not a shell wrapper: the rest of the toolchain (and the
 * backend that spawns this) runs on Node, and the old wrapper only ran
 * on Windows. This uses Node built-ins only — no npm dependencies — so
 * it runs anywhere Node does.
 *
 * Usage:
 * node scripts/ingest_all_transcripts.mjs \
 * [--api-base http://localhost:4000/api] \
 * [--poll-interval 5] [--timeout 3600] \
 * [--creator huberman --creator allin] [--no-poll]
 *
 * Exit codes: 0 on success; non-zero if any creator's ingest fails or a
 * manifest is missing.
 */

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

/** Absolute path to this script's directory (ESM has no `__dirname`). */
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
/** Repo root — one level above `scripts/`. */
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
/** Path to the Python ingest script we delegate to. */
const INGEST_SCRIPT = path.join(SCRIPT_DIR, "ingest_transcripts.py");

/**
 * The fixed real creator roster: slug → display name + channel URL.
 * Mirrors the roster the former PowerShell script hard-coded.
 * @type {Record<string, { name: string, url: string }>}
 */
const CREATORS = {
  huberman: {
    name: "Andrew Huberman",
    url: "https://www.youtube.com/@hubermanlab",
  },
  allin: { name: "All In Podcast", url: "https://www.youtube.com/@allin" },
  mkbhd: { name: "Marques Brownlee", url: "https://www.youtube.com/@mkbhd" },
  delauer: {
    name: "Thomas DeLauer",
    url: "https://www.youtube.com/@ThomasDeLauerOfficial",
  },
  campea: {
    name: "John Campea",
    url: "https://www.youtube.com/playlist?list=PL6628E7149D3A7D56",
  },
};

/**
 * Resolve the virtualenv Python interpreter cross-platform.
 *
 * Prefers the repo venv (`.venv/bin/python` on POSIX,
 * `.venv\Scripts\python.exe` on Windows) and falls back to whichever
 * `python`/`python3` is on PATH so the script still works without a
 * committed venv. Returns the first existing candidate.
 * @returns {string} The interpreter command or path.
 */
function resolvePython() {
  const candidates =
    process.platform === "win32"
      ? [path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")]
      : [
          path.join(REPO_ROOT, ".venv", "bin", "python"),
          path.join(REPO_ROOT, ".venv311", "bin", "python"),
        ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

/**
 * Minimal argv parser. Supports `--flag value`, repeatable `--creator`,
 * and the boolean `--no-poll`. Unknown flags throw so typos fail loudly
 * rather than being silently ignored.
 * @param {string[]} argv Raw arguments (excluding node + script path).
 * @returns {{ apiBase: string, pollInterval: number, timeout: number, creators: string[], noPoll: boolean }}
 */
function parseArgs(argv) {
  const opts = {
    apiBase: "http://localhost:4000/api",
    pollInterval: 5,
    timeout: 3600,
    creators: /** @type {string[]} */ ([]),
    noPoll: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--api-base":
        opts.apiBase = argv[(i += 1)];
        break;
      case "--poll-interval":
        opts.pollInterval = Number(argv[(i += 1)]);
        break;
      case "--timeout":
        opts.timeout = Number(argv[(i += 1)]);
        break;
      case "--creator":
        opts.creators.push(argv[(i += 1)]);
        break;
      case "--no-poll":
        opts.noPoll = true;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (opts.creators.length === 0) {
    opts.creators = Object.keys(CREATORS);
  }
  return opts;
}

/**
 * Ingest a single creator by shelling out to the Python ingest script.
 *
 * Verifies the creator slug is known and its `_manifest.json` exists,
 * then runs `ingest_transcripts.py` synchronously with stdio inherited
 * so progress streams to the console. Throws on a missing manifest,
 * unknown slug, or non-zero exit.
 * @param {string} slug The creator slug (must be a key of CREATORS).
 * @param {ReturnType<typeof parseArgs>} opts Parsed CLI options.
 * @param {string} python Resolved Python interpreter.
 * @returns {void}
 */
function ingestCreator(slug, opts, python) {
  const meta = CREATORS[slug];
  if (!meta) {
    throw new Error(
      `Unknown creator '${slug}'. Expected one of: ${Object.keys(CREATORS).join(", ")}`,
    );
  }
  const folder = path.join(REPO_ROOT, "data", "transcripts", slug);
  const manifest = path.join(folder, "_manifest.json");
  if (!existsSync(manifest)) {
    throw new Error(`Missing manifest for '${slug}': ${manifest}`);
  }

  process.stdout.write(`==> Ingesting ${slug} from ${folder}\n`);

  const ingestArgs = [
    INGEST_SCRIPT,
    "--folder",
    folder,
    "--creator-name",
    meta.name,
    "--creator-slug",
    slug,
    "--channel-url",
    meta.url,
    "--api-base",
    opts.apiBase,
    "--poll-interval",
    String(opts.pollInterval),
    "--timeout",
    String(opts.timeout),
  ];
  if (opts.noPoll) {
    ingestArgs.push("--no-poll");
  }

  const result = spawnSync(python, ingestArgs, {
    cwd: REPO_ROOT,
    stdio: "inherit",
  });
  if (result.error) {
    throw new Error(
      `Failed to launch ingest for '${slug}': ${result.error.message}`,
    );
  }
  if (result.status !== 0) {
    throw new Error(
      `Ingest failed for '${slug}' with exit code ${result.status}`,
    );
  }
}

/**
 * Entry point: parse args, ingest each requested creator in sequence.
 * Returns a process exit code (0 success, 1 on the first failure).
 * @returns {number}
 */
function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(
      `${err instanceof Error ? err.message : String(err)}\n`,
    );
    return 1;
  }

  const python = resolvePython();
  for (const slug of opts.creators) {
    try {
      ingestCreator(slug, opts, python);
    } catch (err) {
      process.stderr.write(
        `${err instanceof Error ? err.message : String(err)}\n`,
      );
      return 1;
    }
  }

  process.stdout.write("All requested transcript folders submitted.\n");
  return 0;
}

process.exit(main());
