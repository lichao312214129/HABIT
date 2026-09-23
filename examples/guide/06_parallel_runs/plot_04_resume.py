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
list(backend.map(subject_label, cohort, checkpoint=store))
second = list(backend.map(subject_label, cohort, checkpoint=store))
print([slot.from_cache for slot in second])
print([slot.result() for slot in second])
