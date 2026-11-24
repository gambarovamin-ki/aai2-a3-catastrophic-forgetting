"""
Generate a SYNTHETIC results file so the evaluation pipeline can be tested
before the real experiments exist.

THESE NUMBERS ARE INVENTED. They are not results, they are not evidence,
and nothing produced from them may appear in the report or the slides as
a finding. The only purpose is to check that metrics.py and aggregate.py
read the right columns and compute the right things.

Every file this script writes is named `SYNTHETIC_*` for that reason.

Usage:  python make_synthetic_results.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PS_VALUES = [0, 8, 12, 16, 20, 24, 28]
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["naive", "l2", "ewc", "replay", "joint"]

rng = np.random.default_rng(0)


def invented_forgetting(method: str, ps: int) -> float:
    """A made-up hump, so the pipeline has something non-trivial to chew on."""
    if method == "joint":
        return 0.0
    hump = np.exp(-((ps - 16) ** 2) / (2 * 7.0 ** 2))   # peaks near PS=16
    scale = {"naive": 70.0, "l2": 45.0, "ewc": 25.0, "replay": 12.0}[method]
    return float(scale * hump * (ps > 0))


def invented_task_b(method: str, ps: int) -> float:
    if ps == 0:
        return 97.5
    penalty = {"naive": 0.0, "l2": 9.0, "ewc": 3.0, "replay": 1.0,
               "joint": 0.0}[method]
    return 97.5 - penalty


def main() -> None:
    rows = []

    # Untrained network: the FWT baseline.
    for seed in SEEDS:
        for task in (1, 2):
            rows.append(dict(method="random_init", pair="permuted", ps=-1,
                             seed=seed, after_task=0, eval_task=task,
                             accuracy=10.0 + rng.normal(0, 0.4)))

    for ps in PS_VALUES:
        for method in METHODS:
            for seed in SEEDS:
                noise = lambda s=1.0: float(rng.normal(0, s))
                acc_A_after_A = 98.0 + noise(0.3)
                acc_B_before = (97.0 if ps == 0 else max(10.0, 70.0 - 2.2 * ps)) + noise(0.8)
                acc_A_after_B = acc_A_after_A - invented_forgetting(method, ps) + noise(1.5)
                acc_B_after_B = invented_task_b(method, ps) + noise(0.5)

                for (i, j, a) in [(1, 1, acc_A_after_A),
                                  (1, 2, acc_B_before),
                                  (2, 1, acc_A_after_B),
                                  (2, 2, acc_B_after_B)]:
                    rows.append(dict(method=method, pair="permuted", ps=ps,
                                     seed=seed, after_task=i, eval_task=j,
                                     accuracy=round(float(np.clip(a, 0, 100)), 2)))

    out = "SYNTHETIC_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}  ({len(rows)} rows)")
    print("REMINDER: invented numbers, for pipeline testing only.")


if __name__ == "__main__":
    main()
