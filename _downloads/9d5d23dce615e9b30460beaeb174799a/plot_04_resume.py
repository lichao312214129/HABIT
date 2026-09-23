"""
Resuming finished subjects
==========================

A :class:`~habit.execution.CheckpointStore` records each success.
The next ``map`` with the same store and the same callable returns
those subjects from the store (``from_cache``) and does not call the
function again. ``Study.fit_predict(..., checkpoint=store)`` uses this
store.
"""

# %%
# The first pass computes three ids. The second pass prints ``True``
# for every subject and does not print ``compute``.
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, Subject
from habit.execution import CheckpointStore, SerialBackend

cohort = Cohort(
    [
        Subject(subject_id="subj001", images={}, masks={}),
        Subject(subject_id="subj002", images={}, masks={}),
        Subject(subject_id="subj003", images={}, masks={}),
    ],
    name="resume",
)

def subject_label(subject: Subject) -> str:
    """Print when the function actually runs."""
    print("compute", subject.subject_id)
    return subject.subject_id

store = CheckpointStore(Path(tempfile.mkdtemp(prefix="habit_ckpt_")))
backend = SerialBackend()
first = list(backend.map(subject_label, cohort, checkpoint=store))
second = list(backend.map(subject_label, cohort, checkpoint=store))
print([slot.from_cache for slot in second])
print([slot.result() for slot in second])

labels = [slot.subject_id for slot in second]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
positions = range(len(labels))
ax.bar(
    [index - 0.18 for index in positions],
    [0 if slot.from_cache else 1 for slot in first],
    width=0.36,
    label="first pass",
    color="#4C78A8",
)
ax.bar(
    [index + 0.18 for index in positions],
    [1 if slot.from_cache else 0 for slot in second],
    width=0.36,
    label="from cache",
    color="#54A24B",
)
ax.set_xticks(list(positions))
ax.set_xticklabels(labels)
ax.set_ylabel("done")
ax.set_title("resume skips finished subjects")
ax.legend()
Path("out").mkdir(exist_ok=True)
fig.savefig("out/resume.png", dpi=150, bbox_inches="tight")
plt.show()
