"""
Stopping at the first failure
=============================

``on_subject_failure="fail_fast"`` re-raises the first subject error
and does not run the rest. Use it when a partial cohort would be
misread as a finished study.
"""

# %%
# ``subj002`` raises. ``subj003`` is not visited.
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
    name="stop",
)

def subject_label(subject: Subject) -> str:
    """Fail one known id so the run stops at that subject."""
    if subject.subject_id == "subj002":
        raise RuntimeError("unreadable series")
    return subject.subject_id

seen = []

def record(subject: Subject) -> str:
    """Record who actually ran, then apply the failing operation."""
    seen.append(subject.subject_id)
    return subject_label(subject)

try:
    list(SerialBackend(on_subject_failure="fail_fast").map(record, cohort))
except RuntimeError as exc:
    print(type(exc).__name__)
print(seen)

labels = [subject.subject_id for subject in cohort]
visited = [subject_id in seen for subject_id in labels]
fig, ax = plt.subplots(figsize=(6.4, 2.4))
colors = ["#4C78A8" if flag else "#D0D0D0" for flag in visited]
ax.barh(list(reversed(labels)), [1, 1, 1], color=list(reversed(colors)))
ax.set_xlabel("visited")
ax.set_title("stop at the first failure")
Path("out").mkdir(exist_ok=True)
fig.savefig("out/stop_on_failure.png", dpi=150, bbox_inches="tight")
plt.show()
