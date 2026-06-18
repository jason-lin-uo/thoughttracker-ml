"""
Train / validation / test splitting + tokenizer-input shaping.

The two main exports:

  - ``split_dataset(df)`` — partition a DataFrame into three reproducible,
    label-stratified splits. Falls back to a non-stratified split when
    any label is too sparse to stratify.
  - ``to_model_inputs(df)`` — turn a split DataFrame into the
    ``(texts, label_ids)`` pair the Hugging Face Trainer wants.

We deliberately keep this file thin and pure. No I/O, no torch imports
at module top — the heavy ML libraries are imported lazily by the
training script. That makes this module fast to import (tests start
in milliseconds) and easy to unit-test against synthetic DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .label_schema import label_to_id
from .load_dataset import build_input_text


#: Columns we treat as a "group" key for leakage-safe splitting, in
#: priority order. Transcript chunks that share any of these belong to
#: the same source (same video / creator / labeled row), so they must
#: never straddle the train/test boundary — otherwise near-duplicate
#: chunks leak from train into test and inflate the reported metrics.
#: The first column present in the DataFrame wins.
GROUP_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "video_id",
    "videoId",
    "creator_slug",
    "creator",
    "id",
)


@dataclass
class Splits:
    """Container for the three splits produced by ``split_dataset``.

    A dataclass instead of a tuple so call sites read as
    ``splits.train`` / ``splits.val`` / ``splits.test`` instead of
    ``splits[0]`` / ``splits[1]`` / ``splits[2]`` — easier to grep
    and harder to swap by accident.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    group_columns: Optional[Sequence[str]] = None,
) -> Splits:
    """Partition ``df`` into train / val / test splits.

    Leakage safety (the important part)
    -----------------------------------
    Transcript chunks from the same video (or creator, or labeled
    source row) are near-duplicates of one another — they share
    vocabulary, speaker idiosyncrasies, and topic framing. A naive
    row-level random split scatters those near-duplicates across train
    AND test, so the model effectively "sees the test set during
    training" and reports **optimistically inflated** metrics. To
    prevent this, when a group key is available we split by GROUP: every
    row sharing a group id lands entirely in one split.

    Algorithm:

    1. **Tiny-dataset fast path.** If the dataset has fewer than 8
       rows we don't even attempt a smart split — sklearn would error
       out, and we'd rather produce splits that "work" for smoke-testing
       the pipeline than refuse outright. The fast path gives every
       split at least one row.
    2. **Group-aware path (preferred).** If a group key resolves (see
       ``group_columns`` / :data:`GROUP_COLUMN_CANDIDATES`) AND it has at
       least 6 distinct groups that are NOT unique-per-row, we use
       ``GroupShuffleSplit`` so no group straddles a split boundary. (Six,
       not three, because the split is hierarchical — see ``_resolve_group_key``.)
    3. **Stratified fallback.** With no usable group key, we fall back
       to the original label-stratified row split (stratify only when
       every label has ≥2 rows; sklearn errors otherwise).

    In both smart paths the holdout is split a second time into val vs
    test, with the fraction computed so the FINAL test split is exactly
    ``test_size`` of the original data.

    Determinism: ``seed`` flows into every splitter so reruns produce
    byte-identical splits — important for reproducible eval metrics
    across training runs.

    Parameters
    ----------
    df
        The full labeled dataset (after ``load_stance_dataset``).
    test_size
        Fraction of the original data to reserve for the final
        held-out test set. Default 0.2.
    val_size
        Fraction of the original data to reserve for validation
        (used by the Trainer's ``eval_strategy="epoch"``). Default 0.1.
    seed
        Random seed for the splitters.
    group_columns
        Optional explicit group-key column priority list. When ``None``
        we auto-detect from :data:`GROUP_COLUMN_CANDIDATES`. Pass an
        empty sequence to force the stratified fallback (no grouping).

    Returns
    -------
    ``Splits(train, val, test)`` — three DataFrames with contiguous
    integer indexes (via ``reset_index(drop=True)``).
    """
    if not 0 < test_size + val_size < 1:
        # The two holdout fractions must leave a non-empty train split AND be a
        # valid fraction for sklearn's splitters. Catch a bad caller up front
        # with a clear message instead of a cryptic sklearn error deep in the
        # split, or a silently-empty train set.
        raise ValueError(
            "test_size + val_size must sum to a fraction in (0, 1); got "
            f"{test_size} + {val_size} = {test_size + val_size}"
        )
    if len(df) < 8:
        # Very small dataset — give every split at least one row so
        # the downstream training code doesn't choke on a 0-row split.
        train = df.iloc[: max(1, len(df) - 2)].reset_index(drop=True)
        val = df.iloc[max(1, len(df) - 2) : max(2, len(df) - 1)].reset_index(drop=True)
        test = df.iloc[max(2, len(df) - 1) :].reset_index(drop=True)
        return Splits(train=train, val=val, test=test)

    val_fraction_of_holdout = val_size / (test_size + val_size)
    group_key = _resolve_group_key(df, group_columns)

    if group_key is not None:
        train_df, holdout_df = _group_split(
            df, group_key, test_size + val_size, seed
        )
        val_df, test_df = _group_split(
            holdout_df,
            group_key.loc[holdout_df.index],
            1 - val_fraction_of_holdout,
            seed,
        )
    else:
        # Stratify only when every label has ≥2 rows; sklearn errors otherwise.
        stratify = df["label"] if _can_stratify(df["label"]) else None
        train_df, holdout_df = train_test_split(
            df,
            test_size=test_size + val_size,
            random_state=seed,
            stratify=stratify,
        )
        holdout_stratify = (
            holdout_df["label"] if _can_stratify(holdout_df["label"]) else None
        )
        val_df, test_df = train_test_split(
            holdout_df,
            # `train_test_split`'s test_size here is "fraction for the SECOND
            # split" — so we pass the complement of val_fraction_of_holdout.
            test_size=1 - val_fraction_of_holdout,
            random_state=seed,
            stratify=holdout_stratify,
        )

    return Splits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def _resolve_group_key(
    df: pd.DataFrame, group_columns: Optional[Sequence[str]]
) -> Optional[pd.Series]:
    """Pick the group-id column to split on, or ``None`` to fall back.

    Returns the first present column from ``group_columns`` (or, when
    that's ``None``, from :data:`GROUP_COLUMN_CANDIDATES`) — but ONLY if it
    yields at least 6 distinct groups that aren't unique-per-row (see the
    inline comment for why). Otherwise we decline and let the caller use the
    stratified fallback. An explicit empty ``group_columns`` disables
    grouping entirely (returns ``None``).
    """
    candidates: Sequence[str]
    if group_columns is None:
        candidates = GROUP_COLUMN_CANDIDATES
    else:
        candidates = group_columns

    for column in candidates:
        if column in df.columns:
            key = df[column].astype(str)
            n_groups = key.nunique()
            # Require ≥6 groups, AND reject a column that is UNIQUE PER ROW.
            #  - ≥6: the split is HIERARCHICAL (train/holdout, then val/test on
            #    the holdout). With only 3–5 groups the holdout can end up with
            #    a single group, and GroupShuffleSplit RAISES on a one-group
            #    split. Six guarantees ≥2 groups survive into the second split.
            #  - `< len(df)`: a column with one distinct value per row (e.g. a
            #    row-level "id") would make GroupShuffleSplit a plain row split
            #    with ZERO leakage protection while masquerading as group-aware.
            # Either case declines → the caller uses the stratified fallback.
            if 6 <= n_groups < len(df):
                return key
    return None


def _group_split(
    df: pd.DataFrame, groups: pd.Series, holdout_fraction: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One leakage-safe split via ``GroupShuffleSplit``.

    Splits ``df`` into ``(kept, holdout)`` where ``holdout`` is
    approximately ``holdout_fraction`` of the rows and — critically — no
    value in ``groups`` appears in both halves. ``groups`` is aligned to
    ``df`` by position (the caller passes a Series indexed like ``df``).
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=holdout_fraction, random_state=seed
    )
    kept_idx, holdout_idx = next(splitter.split(df, groups=groups.to_numpy()))
    return df.iloc[kept_idx], df.iloc[holdout_idx]


def assert_no_group_leakage(splits: Splits, group_columns: Optional[Sequence[str]] = None) -> List[str]:
    """Verify no group id appears in more than one split.

    Returns the resolved group column names that were actually checked
    (empty list when no group key was present, meaning leakage can't be
    asserted). Raises ``AssertionError`` if any group straddles two
    splits. Exposed as a public guard so tests — and a paranoid training
    run — can assert the leakage-safety invariant directly rather than
    trusting the splitter blindly.
    """
    candidates: Sequence[str]
    if group_columns is None:
        candidates = GROUP_COLUMN_CANDIDATES
    else:
        candidates = group_columns

    checked: List[str] = []
    frames = {"train": splits.train, "val": splits.val, "test": splits.test}
    for column in candidates:
        if not all(column in frame.columns for frame in frames.values()):
            continue
        sets = {
            name: set(frame[column].astype(str)) for name, frame in frames.items()
        }
        overlaps = (
            (sets["train"] & sets["val"])
            | (sets["train"] & sets["test"])
            | (sets["val"] & sets["test"])
        )
        assert not overlaps, (
            f"Group column {column!r} leaks across splits: {sorted(overlaps)}"
        )
        checked.append(column)
    return checked


def _can_stratify(series: pd.Series) -> bool:
    """True if a label column has at least 2 distinct values AND every
    value occurs at least twice. Both are sklearn's preconditions for
    stratified splitting — fail either and we silently fall back to
    a non-stratified split (warning the user via the test count would
    be noisier than helpful here).
    """
    counts = series.value_counts()
    return bool(len(counts) > 1 and counts.min() >= 2)


def to_model_inputs(df: pd.DataFrame) -> Tuple[list, list]:
    """Turn a split DataFrame into the pair Hugging Face's Trainer wants.

    The Trainer's ``Dataset.from_dict({"text": ..., "label": ...})``
    expects two parallel lists:

      - ``texts``: each row's ``(topic, text)`` formatted via
        ``build_input_text`` so the encoder sees identical formatting
        in training and inference.
      - ``labels``: integer label ids (Trainer wants ints, not strings).

    Parameters
    ----------
    df
        A split DataFrame (from ``split_dataset``) with at minimum
        ``topic``, ``text``, ``label`` columns.

    Returns
    -------
    ``(texts, label_ids)`` — same length, parallel ordering preserved.
    """
    texts = [build_input_text(row.topic, row.text) for row in df.itertuples()]
    labels = [label_to_id(label) for label in df["label"].tolist()]
    return texts, labels
