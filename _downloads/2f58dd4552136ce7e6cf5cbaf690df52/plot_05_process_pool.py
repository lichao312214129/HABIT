"""
Running subjects in separate processes
======================================

``RunPolicy(workers=2, backend="process")`` builds a
:class:`~habit.execution.ProcessPoolBackend`. Pass that object as
``backend=`` to ``fit_predict``. ``subject_timeout_sec`` is enforced
only on this backend. On Windows the ``map`` call must sit under
``if __name__ == "__main__":``.
"""

# %%
# Building the backend does not start workers. ``map`` does.
# This docs build has no script path, so it stops after printing the backend.
from habit.contracts import Cohort, Subject
from habit.execution import backend_from_policy
from habit.spec import RunPolicy

cohort = Cohort(
    [
        Subject(subject_id="subj001", images={}, masks={}),
        Subject(subject_id="subj002", images={}, masks={}),
        Subject(subject_id="subj003", images={}, masks={}),
    ],
    name="processes",
)

def subject_label(subject: Subject) -> str:
    """Return the subject id. Workers re-import this function."""
    return subject.subject_id

def main() -> None:
    """Build the pool and, when this file is the script, map the cohort."""
    policy = RunPolicy(
        workers=2,
        backend="process",
        parallel_mode="persistent",
        on_subject_failure="continue",
        subject_timeout_sec=30.0,
    )
    backend = backend_from_policy(policy)
    print(type(backend).__name__, backend.workers, backend.policy.subject_timeout_sec)
    # Workers re-import this file. Only the original script should map.
    if "__file__" in globals():
        slots = list(backend.map(subject_label, cohort))
        print([slot.result() for slot in slots])


if __name__ == "__main__":
    main()
