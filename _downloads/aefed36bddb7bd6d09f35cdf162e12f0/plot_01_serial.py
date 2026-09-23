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

for slot in SerialBackend().map(subject_label, cohort):
    print(slot.subject_id, slot.result())
