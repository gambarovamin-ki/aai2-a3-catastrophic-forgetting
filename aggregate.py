"""
Turn the raw results log into per-run metrics and a seed-averaged summary.

INPUT  results.csv, written by the training code (P1), one row per
       evaluation, with these columns:

           method      naive | joint | l2 | ewc | replay | random_init
           pair        permuted | fashion
           ps          side of the permuted square, 0..28
                       (ignored for pair=fashion; write -1)
           seed        integer random seed
           after_task  0 = before any training (random init)
                       1 = after training Task A
                       2 = after training Task B
           eval_task   1 = evaluated on Task A, 2 = on Task B
           accuracy    percentage, 0..100

       Every (method, pair, ps, seed) needs the four rows
       after_task/eval_task = 1/1, 1/2, 2/1, 2/2.
       `random_init` rows (after_task=0) supply the FWT baseline and are
       not treated as a method.

OUTPUT metrics_per_run.csv      one row per (method, pair, ps, seed)
       metrics_summary.csv      mean and SD across seeds

Usage:  python aggregate.py results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import all_metrics

REQUIRED_COLUMNS = {
    "method", "pair", "ps", "seed", "after_task", "eval_task", "accuracy",
}

# Methods that are protection strategies, not references.
METHOD_ORDER = ["naive", "l2", "ewc", "replay", "joint"]


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"results file is missing columns: {sorted(missing)}")
    return df


def random_baseline(df: pd.DataFrame) -> np.ndarray:
    """Accuracy of the untrained network on each task, averaged over seeds.

    Falls back to 10% per task (chance for 10 classes) if the training code
    did not log any random_init rows.
    """
    rows = df[df["after_task"] == 0]
    if rows.empty:
        return np.array([10.0, 10.0])
    b = [
        rows.loc[rows["eval_task"] == t, "accuracy"].mean()
        for t in (1, 2)
    ]
    return np.array([10.0 if np.isnan(x) else x for x in b])


def build_matrix(group: pd.DataFrame) -> np.ndarray | None:
    """Build the 2x2 accuracy matrix R for one run, or None if incomplete."""
    R = np.full((2, 2), np.nan)
    for _, row in group.iterrows():
        i, j = int(row["after_task"]), int(row["eval_task"])
        if i in (1, 2) and j in (1, 2):
            R[i - 1, j - 1] = row["accuracy"]
    return None if np.isnan(R).any() else R


def joint_lookup(df: pd.DataFrame) -> dict:
    """Task B accuracy of the Joint upper bound, per (pair, ps, seed)."""
    joint = df[(df["method"] == "joint") &
               (df["after_task"] == 2) &
               (df["eval_task"] == 2)]
    return {
        (r["pair"], r["ps"], r["seed"]): r["accuracy"]
        for _, r in joint.iterrows()
    }


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    baseline = random_baseline(df)
    joints = joint_lookup(df)
    runs = df[df["method"] != "random_init"]

    records, skipped = [], []
    for (method, pair, ps, seed), group in runs.groupby(
        ["method", "pair", "ps", "seed"], dropna=False
    ):
        R = build_matrix(group)
        if R is None:
            skipped.append((method, pair, ps, seed))
            continue
        joint_acc = joints.get((pair, ps, seed))
        m = all_metrics(R, random_baseline=baseline, joint_accuracy=joint_acc)
        records.append(
            {"method": method, "pair": pair, "ps": ps, "seed": seed, **m}
        )

    if skipped:
        print(f"warning: {len(skipped)} incomplete run(s) skipped, "
              f"first few: {skipped[:3]}", file=sys.stderr)
    if not records:
        raise ValueError("no complete runs found in the results file")
    return pd.DataFrame(records)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across seeds. No single-seed number survives this step."""
    metric_cols = [
        c for c in runs.columns
        if c not in ("method", "pair", "ps", "seed")
    ]
    grouped = runs.groupby(["method", "pair", "ps"], dropna=False)

    out = grouped[metric_cols].agg(["mean", "std", "count"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    out = out.reset_index()
    out = out.rename(columns={"forgetting_count": "n_seeds"})
    out = out.drop(columns=[c for c in out.columns if c.endswith("_count")])

    # Sort methods into the reporting order. Any method not in
    # METHOD_ORDER is kept and placed after the known ones, so an added
    # condition is never silently dropped from the summary.
    known = [m for m in METHOD_ORDER if m in set(out["method"])]
    extra = sorted(set(out["method"]) - set(METHOD_ORDER))
    if extra:
        print(f"note: unrecognised method(s) in results: {extra}",
              file=sys.stderr)
    out["method"] = pd.Categorical(
        out["method"], categories=known + extra, ordered=True
    )
    return out.sort_values(["pair", "ps", "method"]).reset_index(drop=True)


def headline(summary: pd.DataFrame) -> pd.DataFrame:
    """The table that goes in the report: forgetting and Task B accuracy."""
    cols = ["method", "pair", "ps", "n_seeds",
            "forgetting_mean", "forgetting_std",
            "acc_B_after_B_mean", "acc_B_after_B_std"]
    cols = [c for c in cols if c in summary.columns]
    t = summary[cols].copy()
    return t.rename(columns={
        "forgetting_mean": "forgetting",
        "forgetting_std": "forgetting_sd",
        "acc_B_after_B_mean": "task_B_acc",
        "acc_B_after_B_std": "task_B_acc_sd",
    })


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    df = load(argv[1])
    runs = per_run(df)
    summary = summarise(runs)

    runs.to_csv("metrics_per_run.csv", index=False)
    summary.to_csv("metrics_summary.csv", index=False)

    print("Per-run metrics  -> metrics_per_run.csv")
    print("Seed summary     -> metrics_summary.csv")
    print()
    print(headline(summary).to_string(index=False, float_format="%.2f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
