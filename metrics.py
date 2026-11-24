"""
Continual-learning evaluation metrics for AAI2 project A3.

Everything is computed from the accuracy matrix R, where

    R[i, j] = accuracy on task j after having trained through task i

with 0-based indices internally (task 0 = Task A, task 1 = Task B).
In the report we write R[1,1], R[2,1] etc. with 1-based indices, as the
literature does; `format_matrix` prints it that way.

Definitions follow:
  BWT, FWT        Lopez-Paz & Ranzato, GEM, NeurIPS 2017 (arXiv:1706.08840)
  Forgetting      Chaudhry et al., Riemannian Walk, ECCV 2018
  Intransigence   Chaudhry et al., Riemannian Walk, ECCV 2018
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_accuracy",
    "backward_transfer",
    "forgetting",
    "forward_transfer",
    "intransigence",
    "task_b_accuracy",
    "all_metrics",
    "format_matrix",
]


def _check(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square, got shape {R.shape}")
    if R.shape[0] < 2:
        raise ValueError("Need at least two tasks")
    if np.any(R < 0) or np.any(R > 100):
        raise ValueError("Accuracies must be percentages in [0, 100]")
    return R


def average_accuracy(R: np.ndarray) -> float:
    """ACC: mean accuracy over all tasks after the final task.

    ACC = (1/T) * sum_i R[T, i]
    """
    R = _check(R)
    return float(R[-1, :].mean())


def backward_transfer(R: np.ndarray) -> float:
    """BWT: how much learning later tasks changed earlier ones.

    BWT = (1/(T-1)) * sum_{i<T} (R[T, i] - R[i, i])

    Negative = forgetting. Positive = earlier tasks improved.
    """
    R = _check(R)
    T = R.shape[0]
    return float(np.mean([R[-1, i] - R[i, i] for i in range(T - 1)]))


def forgetting(R: np.ndarray) -> float:
    """Average forgetting: best-ever accuracy on a task minus its final accuracy.

    f_j = max_{l < T} R[l, j] - R[T, j],  averaged over j < T.

    For two tasks this equals -BWT.
    """
    R = _check(R)
    T = R.shape[0]
    per_task = [R[:T - 1, j].max() - R[-1, j] for j in range(T - 1)]
    return float(np.mean(per_task))


def forward_transfer(R: np.ndarray, random_baseline: np.ndarray | list) -> float:
    """FWT: how much knowing earlier tasks helps on a task before training it.

    FWT = (1/(T-1)) * sum_{i>=1} (R[i-1, i] - b[i])

    `random_baseline[i]` is the accuracy of the freshly initialised network
    on task i (about 10% for 10-class MNIST).
    """
    R = _check(R)
    b = np.asarray(random_baseline, dtype=float)
    T = R.shape[0]
    if b.shape[0] != T:
        raise ValueError(f"random_baseline needs {T} entries, got {b.shape[0]}")
    return float(np.mean([R[i - 1, i] - b[i] for i in range(1, T)]))


def task_b_accuracy(R: np.ndarray) -> float:
    """R[T, T]: accuracy on the final task after training it.

    Reported beside every forgetting number. A method can score zero
    forgetting simply by refusing to learn the new task; this column is
    what exposes that.
    """
    R = _check(R)
    return float(R[-1, -1])


def intransigence(R: np.ndarray, joint_accuracy: float) -> float:
    """Intransigence: how much plasticity the protection cost.

    I = a_joint - R[T, T]

    `joint_accuracy` is the accuracy on the final task of a model trained
    on all tasks together (our Joint upper bound). Large = the method
    protected the old task by refusing to learn the new one.
    """
    R = _check(R)
    return float(joint_accuracy - R[-1, -1])


def all_metrics(
    R: np.ndarray,
    random_baseline: np.ndarray | list | None = None,
    joint_accuracy: float | None = None,
) -> dict:
    """Every metric at once. Optional inputs give the optional metrics."""
    R = _check(R)
    out = {
        "acc_A_after_A": float(R[0, 0]),
        "acc_A_after_B": float(R[-1, 0]),
        "acc_B_after_B": task_b_accuracy(R),
        "forgetting": forgetting(R),
        "bwt": backward_transfer(R),
        "average_accuracy": average_accuracy(R),
    }
    if random_baseline is not None:
        out["fwt"] = forward_transfer(R, random_baseline)
    if joint_accuracy is not None:
        out["intransigence"] = intransigence(R, joint_accuracy)
    return out


def format_matrix(R: np.ndarray) -> str:
    """Print R with 1-based labels, the way the report writes it."""
    R = _check(R)
    T = R.shape[0]
    header = "          " + "".join(f"  task {j+1}" for j in range(T))
    rows = [header]
    for i in range(T):
        cells = "".join(f"  {R[i, j]:6.2f}" for j in range(T))
        rows.append(f"after t{i+1}  {cells}")
    return "\n".join(rows)
