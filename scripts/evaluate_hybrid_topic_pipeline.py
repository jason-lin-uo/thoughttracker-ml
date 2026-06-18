#!/usr/bin/env python
"""Evaluate TF-IDF topic candidates plus a transformer relevance gate.

Evaluation is performed on a **held-out, group-aware** slice of the
dataset (see ``--test-size`` / ``--group-column``), not the whole loaded
set. Grouping by source id keeps near-duplicate chunks from the same row
out of the evaluation slice's training-adjacent neighbors, so reported
metrics aren't optimistically inflated by leakage. Model artifacts are
loaded with ``joblib`` rather than a bare ``pickle.load`` (joblib is
scikit-learn's recommended serializer and narrows the deserialization
attack surface on a potentially-writable model directory).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402  (import after sys.path bootstrap above)
from src.data.load_dataset import build_input_text  # noqa: E402
from src.inference._device import select_device  # noqa: E402
from src.utils.paths import METRICS_DIR  # noqa: E402


def parse_args(argv=None):
    """Build the CLI parser for the hybrid topic-pipeline evaluator."""
    parser = argparse.ArgumentParser(description="Evaluate hybrid topic pipeline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/processed/thoughttracker_topic_reranker_gold_standard.csv"
        ),
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        default=Path("models/topic-reranker-tfidf-sgd-supervalidation/topic_reranker_model.pkl"),
    )
    parser.add_argument(
        "--relevance-model-dir",
        type=Path,
        default=config.topic_relevance_model_dir,
    )
    parser.add_argument(
        "--topic-metadata",
        type=Path,
        default=Path("data/processed/thoughttracker_topic_relevance_gold_standard.csv"),
    )
    parser.add_argument("--candidate-top-k", type=int, default=12)
    parser.add_argument("--candidate-min-score", type=float, default=0.0)
    parser.add_argument("--max-selected", type=int, default=5)
    parser.add_argument(
        "--min-reranker-margin",
        type=float,
        default=0.0,
        help="Require selected topics after rank 1 to be within this reranker score margin of the top candidate. 0 disables.",
    )
    parser.add_argument(
        "--min-relevance-margin",
        type=float,
        default=0.0,
        help="Require selected topics after rank 1 to be within this relevance score margin of the top selected candidate. 0 disables.",
    )
    parser.add_argument(
        "--rank-mode",
        choices=["relevance", "combined"],
        default="relevance",
        help="Sort selected topics by relevance score alone or relevance*reranker.",
    )
    parser.add_argument("--relevance-threshold", type=float, default=0.7)
    parser.add_argument(
        "--threshold-sweep",
        default="0.2,0.35,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--topic-relevance-max-length", type=int, default=config.topic_relevance_max_length)
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="0 means evaluate all rows; positive values sample that many rows.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help=(
            "Fraction of the dataset to hold out for evaluation. The rest is "
            "ignored here (it's the model's training data). 0 evaluates the "
            "entire loaded set — only sensible for a model trained elsewhere."
        ),
    )
    parser.add_argument(
        "--group-column",
        default="id",
        help=(
            "Column used as the group key for the leakage-safe held-out "
            "split (no group straddles the train/eval boundary). Falls back "
            "to a plain random split if the column is absent or has <3 groups."
        ),
    )
    parser.add_argument(
        "--display-tiers",
        default="",
        help="Optional comma-separated displayTier filter, e.g. showcase,usable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=METRICS_DIR / "hybrid_topic_pipeline_metrics.json",
    )
    return parser.parse_args(argv)


def parse_labels(value: object) -> tuple[str, ...]:
    """Parse a pipe-delimited ``labels`` cell into a tuple of slugs.

    Treats NaN / ``None`` / empty as no labels; splits on ``|`` and drops
    empty fragments. Returns a tuple so it's hashable / order-stable.
    """
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    return tuple(label for label in text.split("|") if label)


def held_out_split(
    df: pd.DataFrame, test_size: float, group_column: str, seed: int
) -> pd.DataFrame:
    """Return the leakage-safe held-out evaluation slice of ``df``.

    ``test_size <= 0`` returns the whole frame (evaluate everything —
    only valid for a model trained elsewhere). Otherwise, when
    ``group_column`` is present with at least 3 distinct groups, a
    ``GroupShuffleSplit`` carves out ``test_size`` of the rows such that
    no group id straddles the held-out / remainder boundary; failing
    that, a plain random split is used. Evaluating on this slice — rather
    than the entire loaded dataset — prevents optimistically inflated
    metrics from train/eval leakage.
    """
    if test_size <= 0 or len(df) < 5:
        return df.reset_index(drop=True)
    if group_column in df.columns and df[group_column].astype(str).nunique() >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, eval_idx = next(
            splitter.split(df, groups=df[group_column].astype(str).to_numpy())
        )
        return df.iloc[eval_idx].reset_index(drop=True)
    _, eval_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    return eval_df.reset_index(drop=True)


def load_dataset(path: Path, display_tiers: set[str], max_rows: int) -> pd.DataFrame:
    """Load + clean the topic-reranker gold-standard CSV.

    Validates required columns (``id``, ``text``, ``labels``), parses the
    pipe-delimited label column, optionally filters by ``displayTier``,
    drops blank rows, and optionally subsamples to ``max_rows``.
    """
    df = pd.read_csv(path).fillna("")
    missing = {"id", "text", "labels"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df["id"] = df["id"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["label_tuple"] = df["labels"].map(parse_labels)
    if display_tiers and "displayTier" in df.columns:
        df = df[df["displayTier"].astype(str).isin(display_tiers)]
    df = df[(df["id"] != "") & (df["text"] != "")].reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, random_state=config.seed, shuffle=True)
        df = df.reset_index(drop=True)
    return df


def read_jsonl(path: Path):
    """Yield each non-empty line of a JSONL file as a dict.

    Tolerates a UTF-8 BOM and skips blank lines; raises ``ValueError``
    (with the line number) on malformed JSON.
    """
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yield row


def load_taxonomy(path: Path) -> dict[str, dict[str, str]]:
    """Load the topic taxonomy (slug → name/domain) from CSV or JSONL.

    CSV inputs need ``topicSlug`` + ``topicName`` columns; JSONL inputs
    are scanned for a ``{"type": "metadata", "taxonomy": [...]}`` record.
    Raises ``ValueError`` if neither shape yields a taxonomy.
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path).fillna("")
        missing = {"topicSlug", "topicName"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing taxonomy columns: {sorted(missing)}")
        taxonomy: dict[str, dict[str, str]] = {}
        for _, row in df[["topicSlug", "topicName"]].drop_duplicates().iterrows():
            slug = str(row["topicSlug"]).strip()
            if not slug:
                continue
            taxonomy[slug] = {
                "topicSlug": slug,
                "topicName": str(row["topicName"] or slug).strip(),
                "domain": "",
            }
        return taxonomy

    for row in read_jsonl(path):
        if row.get("type") != "metadata":
            continue
        taxonomy: dict[str, dict[str, str]] = {}
        for topic in row.get("taxonomy") or []:
            if not isinstance(topic, dict):
                continue
            slug = str(topic.get("slug") or "").strip()
            if not slug:
                continue
            taxonomy[slug] = {
                "topicSlug": slug,
                "topicName": str(topic.get("name") or slug).strip(),
                "domain": str(topic.get("domain") or "").strip(),
            }
        return taxonomy
    raise ValueError(f"No metadata taxonomy found in {path}")


def load_reranker(path: Path) -> dict[str, Any]:
    """Load + validate the TF-IDF reranker bundle via joblib.

    Requires ``vectorizer`` / ``classifier`` / ``multiLabelBinarizer``
    keys; raises ``ValueError`` if any is missing.
    """
    model = joblib.load(path)
    for key in ["vectorizer", "classifier", "multiLabelBinarizer"]:
        if key not in model:
            raise ValueError(f"{path} missing {key}")
    return model


def load_relevance_model(path: Path) -> dict[str, Any]:
    """Load the relevance gate (transformer or sklearn) and its device.

    Prefers a transformer artifact (``config.json`` present), moving it
    to the best available device (CUDA > Apple MPS > CPU); otherwise
    falls back to the sklearn ``.pkl`` (loaded via joblib). Raises
    ``FileNotFoundError`` when neither artifact exists.
    """
    config_path = path / "config.json"
    if config_path.exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        # Shared cuda > mps > cpu selection (same helper the served inference
        # path uses). On Apple Silicon set PYTORCH_ENABLE_MPS_FALLBACK=1 so any
        # op without an MPS kernel falls back to CPU instead of erroring — this
        # 512-token transformer is dramatically faster on the GPU than on CPU.
        device = select_device(torch)
        device_name = (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "mps (Apple Silicon)"
            if device.type == "mps"
            else "cpu"
        )
        model.to(device)
        model.eval()
        return {
            "kind": "transformer",
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "deviceName": device_name,
        }

    model_file = path / "topic_relevance_model.pkl"
    if model_file.exists():
        return {"kind": "sklearn", "model": joblib.load(model_file), "deviceName": "cpu"}
    raise FileNotFoundError(f"No relevance model found at {path}")


def candidate_rows(reranker: dict[str, Any], texts: list[str], top_k: int, min_score: float):
    """Generate top-``k`` reranker candidates ``(slug, score)`` per text.

    Vectorizes all ``texts`` at once, runs the multi-label
    ``predict_proba``, and for each row returns the highest-scoring
    candidates above ``min_score``, capped at ``top_k``.
    """
    matrix = reranker["vectorizer"].transform(texts)
    probabilities = np.asarray(reranker["classifier"].predict_proba(matrix), dtype=float)
    labels = list(reranker["multiLabelBinarizer"].classes_)
    out: list[list[tuple[str, float]]] = []
    for row in probabilities:
        # Stable descending sort: `argsort(row)[::-1]` used an UNSTABLE
        # quicksort and then reversed it, so tied probabilities came out in an
        # arbitrary (and reversal-dependent) order — non-reproducible eval
        # rankings. Sorting `-row` with kind="stable" keeps tied labels in
        # their original (binarizer class) order.
        ranked = np.argsort(-row, kind="stable")
        candidates: list[tuple[str, float]] = []
        for index in ranked:
            score = float(row[index])
            if score < min_score:
                continue
            candidates.append((labels[int(index)], score))
            if len(candidates) >= top_k:
                break
        out.append(candidates)
    return out


def transformer_relevance_probs(
    loaded: dict[str, Any], encoded_texts: list[str], batch_size: int, max_length: int
) -> np.ndarray:
    """Batched transformer ``P(relevant)`` for a list of encoded texts.

    Runs the relevance model in ``batch_size`` chunks on its device, with
    a numerically-stable softmax, and returns the relevant-class column.
    Returns an empty array for empty input.
    """
    import torch

    tokenizer = loaded["tokenizer"]
    model = loaded["model"]
    device = loaded["device"]
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(0, len(encoded_texts), batch_size):
            batch = encoded_texts[index : index + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits.detach().cpu().numpy()
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs.append(exp / exp.sum(axis=1, keepdims=True))
    if not probs:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(probs)[:, 1]


def sklearn_relevance_probs(loaded: dict[str, Any], encoded_texts: list[str]) -> np.ndarray:
    """sklearn-fallback ``P(relevant)`` for a list of encoded texts.

    Returns the ``relevant``-class probability column, or all-zeros if
    the model's classes don't include ``relevant``.
    """
    model = loaded["model"]
    probabilities = np.asarray(model.predict_proba(encoded_texts), dtype=float)
    classes = list(model.classes_)
    if "relevant" not in classes:
        return np.zeros(len(encoded_texts), dtype=float)
    return probabilities[:, classes.index("relevant")]


def score_candidates(
    relevance_model: dict[str, Any],
    rows: pd.DataFrame,
    candidates: list[list[tuple[str, float]]],
    taxonomy: dict[str, dict[str, str]],
    batch_size: int,
    max_length: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Attach a relevance score to every reranker candidate per row.

    Builds one ``build_input_text`` per (row, candidate) pair, scores
    them all in a single batched relevance pass, and returns, per row, a
    list of ``{topicSlug, topicName, rerankerScore, relevanceScore}``.
    Candidates whose slug is missing from the taxonomy are dropped.
    """
    encoded: list[str] = []
    refs: list[tuple[int, str, float]] = []
    for row_index, (row, row_candidates) in enumerate(zip(rows.itertuples(index=False), candidates)):
        for slug, reranker_score in row_candidates:
            topic = taxonomy.get(slug)
            if not topic:
                continue
            encoded.append(build_input_text(topic["topicName"], str(row.text)))
            refs.append((row_index, slug, reranker_score))

    if relevance_model["kind"] == "transformer":
        relevant_probs = transformer_relevance_probs(
            relevance_model,
            encoded,
            batch_size,
            max_length or config.topic_relevance_max_length,
        )
    else:
        relevant_probs = sklearn_relevance_probs(relevance_model, encoded)

    scored: list[list[dict[str, Any]]] = [[] for _ in range(len(rows))]
    for (row_index, slug, reranker_score), relevant_prob in zip(refs, relevant_probs):
        topic = taxonomy[slug]
        scored[row_index].append(
            {
                "topicSlug": slug,
                "topicName": topic["topicName"],
                "rerankerScore": float(reranker_score),
                "relevanceScore": float(relevant_prob),
            }
        )
    return scored


def predictions_at(
    scored: list[list[dict[str, Any]]],
    threshold: float,
    max_selected: int,
    *,
    min_reranker_margin: float = 0.0,
    min_relevance_margin: float = 0.0,
    rank_mode: str = "relevance",
) -> list[tuple[str, ...]]:
    """Apply the selection policy to scored candidates, per row.

    Keeps candidates at/above the relevance ``threshold``, sorts them
    (by relevance alone or relevance×reranker per ``rank_mode``),
    optionally prunes by reranker/relevance margins relative to the top
    pick, and caps at ``max_selected``. Returns the selected topic slugs
    per row as a tuple.
    """
    predictions: list[tuple[str, ...]] = []
    for row in scored:
        selected = [
            item
            for item in row
            if item["relevanceScore"] >= threshold
        ]
        if rank_mode == "combined":
            selected.sort(
                key=lambda item: (
                    item["relevanceScore"] * item["rerankerScore"],
                    item["relevanceScore"],
                    item["rerankerScore"],
                ),
                reverse=True,
            )
        else:
            selected.sort(key=lambda item: (item["relevanceScore"], item["rerankerScore"]), reverse=True)
        if selected and (min_reranker_margin > 0 or min_relevance_margin > 0):
            top_reranker = selected[0]["rerankerScore"]
            top_relevance = selected[0]["relevanceScore"]
            selected = [
                item
                for index, item in enumerate(selected)
                if index == 0
                or (
                    (min_reranker_margin <= 0 or item["rerankerScore"] >= top_reranker - min_reranker_margin)
                    and (min_relevance_margin <= 0 or item["relevanceScore"] >= top_relevance - min_relevance_margin)
                )
            ]
        if max_selected > 0:
            selected = selected[:max_selected]
        predictions.append(tuple(item["topicSlug"] for item in selected))
    return predictions


def multilabel_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute the multi-label metric bundle for one prediction set.

    Returns micro/macro/samples F1, micro precision/recall, empty-row
    accuracy, exact-match, and average predicted/true topic counts —
    enough to characterize both per-label quality and over/under-prediction.
    """
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    return {
        "micro_f1": f1_score(true_arr, pred_arr, average="micro", zero_division=0),
        "macro_f1": f1_score(true_arr, pred_arr, average="macro", zero_division=0),
        "samples_f1": f1_score(true_arr, pred_arr, average="samples", zero_division=0),
        "micro_precision": precision_score(true_arr, pred_arr, average="micro", zero_division=0),
        "micro_recall": recall_score(true_arr, pred_arr, average="micro", zero_division=0),
        "empty_row_accuracy": float(
            np.mean((true_arr.sum(axis=1) == 0) == (pred_arr.sum(axis=1) == 0))
        ),
        "exact_match": float(np.mean(np.all(true_arr == pred_arr, axis=1))),
        "avg_predicted_topics": float(pred_arr.sum(axis=1).mean()),
        "avg_true_topics": float(true_arr.sum(axis=1).mean()),
    }


def candidate_recall(rows: pd.DataFrame, candidates: list[list[tuple[str, float]]], ks=(1, 3, 5, 8, 12)):
    """Recall@k of the reranker candidate generator (upper bound on the pipeline).

    For each ``k`` in ``ks``, averages (over rows with at least one true
    label) the fraction of true topics present in the top-``k``
    candidates — i.e. how many true topics the gate could even see.
    """
    true_sets = [set(labels) for labels in rows["label_tuple"].tolist()]
    positive_indices = [index for index, labels in enumerate(true_sets) if labels]
    out = {}
    for k in ks:
        recalls = []
        for index in positive_indices:
            candidate_set = {slug for slug, _ in candidates[index][:k]}
            truth = true_sets[index]
            recalls.append(len(candidate_set & truth) / max(1, len(truth)))
        out[f"recall_at_{k}"] = float(np.mean(recalls)) if recalls else 0.0
    return out


def main(argv=None) -> int:
    """Run the hybrid topic-pipeline evaluation and write a metrics report.

    Loads the dataset, carves out a leakage-safe held-out evaluation
    slice, generates reranker candidates, scores them through the
    relevance gate, sweeps relevance thresholds, and writes both the
    full sweep and the default-threshold metrics to the ``--out`` JSON.
    """
    args = parse_args(argv)
    display_tiers = {value.strip() for value in args.display_tiers.split(",") if value.strip()}
    thresholds = [
        float(value.strip())
        for value in args.threshold_sweep.split(",")
        if value.strip()
    ]
    if args.relevance_threshold not in thresholds:
        thresholds.append(args.relevance_threshold)
    thresholds = sorted(set(thresholds))

    df = load_dataset(args.dataset, display_tiers, args.max_test_rows)
    # Evaluate on a held-out, group-aware slice rather than the entire
    # loaded set — otherwise metrics are inflated by train/eval leakage.
    df = held_out_split(df, args.test_size, args.group_column, config.seed)
    taxonomy = load_taxonomy(args.topic_metadata)
    labels = sorted(set(taxonomy) | {label for labels_ in df["label_tuple"] for label in labels_})
    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform(df["label_tuple"])

    reranker = load_reranker(args.reranker_model)
    candidates = candidate_rows(
        reranker,
        df["text"].astype(str).tolist(),
        args.candidate_top_k,
        args.candidate_min_score,
    )
    relevance_model = load_relevance_model(args.relevance_model_dir)
    scored = score_candidates(
        relevance_model,
        df,
        candidates,
        taxonomy,
        args.batch_size,
        args.topic_relevance_max_length,
    )

    sweep = []
    best = None
    for threshold in thresholds:
        pred_labels = predictions_at(
            scored,
            threshold,
            args.max_selected,
            min_reranker_margin=args.min_reranker_margin,
            min_relevance_margin=args.min_relevance_margin,
            rank_mode=args.rank_mode,
        )
        y_pred = mlb.transform(pred_labels)
        metrics = {"threshold": threshold, **multilabel_metrics(y_true, y_pred)}
        sweep.append(metrics)
        if best is None or metrics["micro_f1"] > best["micro_f1"]:
            best = metrics

    default_pred = mlb.transform(
        predictions_at(
            scored,
            args.relevance_threshold,
            args.max_selected,
            min_reranker_margin=args.min_reranker_margin,
            min_relevance_margin=args.min_relevance_margin,
            rank_mode=args.rank_mode,
        )
    )
    payload = {
        "dataset": str(args.dataset),
        "rows": len(df),
        "testSize": args.test_size,
        "groupColumn": args.group_column,
        "displayTiers": sorted(display_tiers),
        "rerankerModel": str(args.reranker_model),
        "relevanceModelDir": str(args.relevance_model_dir),
        "topicRelevanceMaxLength": args.topic_relevance_max_length,
        "device": relevance_model.get("deviceName", "unknown"),
        "candidateTopK": args.candidate_top_k,
        "candidateMinScore": args.candidate_min_score,
        "maxSelected": args.max_selected,
        "minRerankerMargin": args.min_reranker_margin,
        "minRelevanceMargin": args.min_relevance_margin,
        "rankMode": args.rank_mode,
        "relevanceThreshold": args.relevance_threshold,
        "candidateMetrics": candidate_recall(df, candidates, ks=(1, 3, 5, 8, 12)),
        "hybridMetrics": multilabel_metrics(y_true, default_pred),
        "bestThresholdByMicroF1": best,
        "thresholdSweep": sorted(sweep, key=lambda row: row["micro_f1"], reverse=True),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["hybridMetrics"], indent=2))
    print(f"Saved metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
