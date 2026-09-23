"""
Skipping a subject that fails
=============================

``on_subject_failure="continue"`` keeps the batch. The failed subject is
a result slot with ``error`` set. The others still finish.
"""

# %%
# ``subj002`` raises. The loop prints the other two ids and the error type.
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
    name="skip",
)

def subject_label(subject: Subject) -> str:
    """Fail one known id so the batch can show an isolated error."""
    if subject.subject_id == "subj002":
        raise RuntimeError("unreadable series")
    return subject.subject_id

backend = SerialBackend(on_subject_failure="continue")
labels = []
ok = []
for slot in backend.map(subject_label, cohort):
    labels.append(slot.subject_id)
    if slot.error is None:
        print(slot.subject_id, slot.result())
        ok.append(True)
    else:
        print(slot.subject_id, type(slot.error).__name__)
        ok.append(False)

fig, ax = plt.subplots(figsize=(6.4, 2.4))
colors = ["#54A24B" if flag else "#E45756" for flag in ok]
ax.barh(list(reversed(labels)), [1, 1, 1], color=list(reversed(colors)))
ax.set_xlabel("finished")
ax.set_title("skip the failed subject")
Path("out").mkdir(exist_ok=True)
fig.savefig("out/skip_failed.png", dpi=150, bbox_inches="tight")
plt.show()
