"""
Running a cohort serially
=========================

Omit ``backend`` and the study runs in this process, one subject after
another. That is the laptop default. Pass the same
:class:`~habit.contracts.Cohort` you would pass to ``fit_predict``.
"""

# %%
# Three subjects stand in for a cohort. The callable only returns the id.
# A habitat study uses the same scheduling with ``Study(spec).fit_predict(cohort)``.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, Subject
from habit.execution import SerialBackend

cohort = Cohort(
    [
        Subject(subject_id="subj001", images={}, masks={}),
        Subject(subject_id="subj002", images={}, masks={}),
        Subject(subject_id="subj003", images={}, masks={}),
    ],
    name="serial",
)

def subject_label(subject: Subject) -> str:
    """Return the subject id. A study would run the habitat steps here."""
    return subject.subject_id

ids = []
for slot in SerialBackend().map(subject_label, cohort):
    print(slot.subject_id, slot.result())
    ids.append(slot.subject_id)

fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.barh(list(reversed(ids)), list(range(len(ids), 0, -1)), color="#4C78A8")
ax.set_xlabel("order")
ax.set_title("serial order")
Path("out").mkdir(exist_ok=True)
fig.savefig("out/serial_order.png", dpi=150, bbox_inches="tight")
plt.show()
