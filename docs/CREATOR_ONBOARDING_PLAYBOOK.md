# Creator Onboarding Playbook

This document describes how ThoughtTracker scales from the current five-creator
demo corpus to additional YouTube creators without rebuilding the whole ML
system from scratch.

## Product Rule

Creator onboarding is owner-only.

The Add Creators button remains visible in the UI so reviewers can see that the
product has a scale-up path, but mutation requires the backend
`ADMIN_ONBOARDING_PIN`. Requests must include:

```text
X-Admin-Pin: <owner pin>
```

Recruiters should not receive the owner PIN or any LLM/API keys.

## Baseline

The current calibrated five-creator corpus is the baseline. New creators are
processed against the frozen policy first, then only uncertain or drift-prone
rows are reviewed.

Frozen topic-selection metrics:

| Metric      | Result |
| ----------- | -----: |
| Exact match | 95.44% |
| Micro F1    | 98.40% |
| Precision   | 97.82% |
| Recall      | 98.98% |
| Macro F1    | 75.35% |

Macro F1 remains the rare-topic polish metric. It is monitored, but it should
not block a recruiter-facing demo when exact match, micro F1, precision, recall,
evidence quality, and display gating are strong.

## Local One-Command Workflow

Run from the ML repo. The orchestrator is cross-platform Node (built-ins
only), so the same command works on macOS, Linux, and Windows:

```bash
cd thoughttracker-ml
node scripts/add_creator_pipeline.mjs \
 --channel-url "https://www.youtube.com/@newcreator" \
 --creator-name "New Creator" \
 --creator-slug "new-creator" \
 --admin-pin "your-owner-pin" \
 --concurrency 5
```

The wrapper performs:

1. transcript fetch via `fetch_transcripts_ytdlp.py`
2. transcript ingest into the TypeScript backend
3. reanalysis with the current frozen topic policy
4. quality audit
5. review-packet generation only for uncertain rows

Progress is written to:

```text
reports/metrics/add_creator_pipeline_status.json
```

## Review Packet Purpose

The packet is not a public artifact. It is an owner review aid for rows where
automation is most likely to drift:

- low-confidence topic assignments
- likely false positives
- likely false negatives
- weak or missing evidence quotes
- rare-topic language
- possible new taxonomy areas

Rows that are already high-confidence under the frozen policy do not need to be
manually relabeled.

## Promotion Rules

Promote a new creator only when:

- existing five-creator regression metrics still pass
- transcript coverage is adequate
- topic precision and recall remain above product threshold
- evidence quotes are inspectable and not generic filler
- no new domain language pollutes existing topic definitions
- any true new topics are added deliberately, with tests and docs updated

## When To Recalibrate

Do not recalibrate for every new creator automatically. Recalibrate when:

- the review packet shows repeated true false positives or false negatives
- the creator introduces a meaningful new domain
- rare-topic macro F1 worsens materially
- display-tier quality becomes visibly weaker

The intended future loop is active learning: run the current policy, review only
the hard rows, add high-value labels, retrain or adjust thresholds, then rerun
the regression suite.

## Recruiter Explanation

The short version:

> ThoughtTracker has a controlled owner-only onboarding path. A new creator URL
> goes through transcript download, ingestion, frozen-policy analysis, quality
> audit, and selective review. The app does not expose that workflow publicly,
> and the existing calibrated corpus remains the regression baseline so scaling
> does not erase prior accuracy.
