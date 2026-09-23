"""
Skipping a subject that fails
=============================

``on_subject_failure="continue"`` keeps the batch. The failed subject is
a result slot with ``error`` set. The others still finish.
"""

# %%
# ``subj002`` raises. The loop prints the other two ids and the error type.
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
for slot in backend.map(subject_label, cohort):
    if slot.error is None:
        print(slot.subject_id, slot.result())
    else:
        print(slot.subject_id, type(slot.error).__name__)
