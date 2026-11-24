"""
Tests for metrics.py, checked against values worked out by hand.

Run:  python -m pytest test_metrics.py -v     (or: python test_metrics.py)
"""

import numpy as np

from metrics import (
    all_metrics,
    average_accuracy,
    backward_transfer,
    forgetting,
    forward_transfer,
    intransigence,
    task_b_accuracy,
)

# A textbook run: Task A learned to 98%, collapses to 23% after Task B.
#   R[1,1] = 98.0   A right after learning A
#   R[1,2] = 12.0   B before training on it
#   R[2,1] = 23.0   A after learning B      <- the damage
#   R[2,2] = 97.0   B after learning B
R_NAIVE = np.array([[98.0, 12.0],
                    [23.0, 97.0]])

# A method that "prevents forgetting" by refusing to learn Task B.
R_FROZEN = np.array([[98.0, 12.0],
                     [98.0, 11.0]])

BASELINE = [10.0, 10.0]   # untrained network, 10 classes
JOINT_B = 98.0            # Task B accuracy of the Joint upper bound


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_forgetting_is_the_drop_in_task_a():
    # 98 - 23 = 75
    approx(forgetting(R_NAIVE), 75.0)


def test_bwt_is_negative_forgetting():
    # 23 - 98 = -75
    approx(backward_transfer(R_NAIVE), -75.0)
    approx(backward_transfer(R_NAIVE), -forgetting(R_NAIVE))


def test_average_accuracy_is_final_row_mean():
    # (23 + 97) / 2 = 60
    approx(average_accuracy(R_NAIVE), 60.0)


def test_forward_transfer_against_random_init():
    # R[1,2] - b[2] = 12 - 10 = 2
    approx(forward_transfer(R_NAIVE, BASELINE), 2.0)


def test_task_b_accuracy():
    approx(task_b_accuracy(R_NAIVE), 97.0)


def test_intransigence_is_the_plasticity_cost():
    # joint 98 - achieved 97 = 1
    approx(intransigence(R_NAIVE, JOINT_B), 1.0)


def test_frozen_network_looks_perfect_until_you_check_task_b():
    """The trap the report exists to expose."""
    approx(forgetting(R_FROZEN), 0.0)          # looks flawless
    approx(task_b_accuracy(R_FROZEN), 11.0)    # it learned nothing
    approx(intransigence(R_FROZEN, JOINT_B), 87.0)


def test_all_metrics_bundle():
    m = all_metrics(R_NAIVE, random_baseline=BASELINE, joint_accuracy=JOINT_B)
    approx(m["forgetting"], 75.0)
    approx(m["bwt"], -75.0)
    approx(m["fwt"], 2.0)
    approx(m["acc_B_after_B"], 97.0)
    approx(m["intransigence"], 1.0)


def test_identical_tasks_give_no_forgetting():
    """Our PS=0 sanity check: identical tasks must not forget."""
    R = np.array([[98.0, 98.0],
                  [98.0, 98.0]])
    approx(forgetting(R), 0.0)


def test_rejects_impossible_accuracies():
    for bad in (np.array([[98.0, 12.0], [23.0, 150.0]]),
                np.array([[98.0, 12.0], [-1.0, 97.0]])):
        try:
            forgetting(bad)
        except ValueError:
            continue
        raise AssertionError("should have rejected out-of-range accuracy")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} tests passed")
