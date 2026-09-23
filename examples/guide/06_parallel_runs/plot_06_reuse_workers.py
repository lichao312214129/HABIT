"""
Reusing workers across passes
=============================

A cold process pool pays spawn and import on the first ``map``.
``with backend.reuse_workers():`` keeps those workers for the next
``map`` in the same block. Two-step studies use this so partitioning
and labelling do not start the pool twice.
"""

# %%
# Entering the block does not spawn workers. ``map`` does, and only when
# this file is executed as a script.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, Subject
from habit.execution import backend_from_policy
from habit.spec import RunPolicy

cohort = Cohort(
    [
        Subject(subject_id="subj001", images={}, masks={}),
        Subject(subject_id="subj002", images={}, masks={}),
    ],
    name="reuse",
)

def subject_label(subject: Subject) -> str:
    """Return the subject id. Workers re-import this function."""
    return subject.subject_id

def main() -> None:
    """Keep one pool open for two maps when this file is the script."""
    policy = RunPolicy(
        workers=2,
        backend="process",
        parallel_mode="persistent",
    )
    backend = backend_from_policy(policy)
    print(type(backend).__name__, backend.policy.parallel_mode)
    with backend.reuse_workers():
        print("workers stay up for every map in this block")
        if "__file__" in globals():
            print([slot.result() for slot in backend.map(subject_label, cohort)])
            print([slot.result() for slot in backend.map(subject_label, cohort)])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(["map 1", "map 2"], [backend.workers, backend.workers], color="#4C78A8")
    ax.set_ylabel("workers")
    ax.set_title("workers stay up")
    Path("out").mkdir(exist_ok=True)
    fig.savefig("out/reuse_workers.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
