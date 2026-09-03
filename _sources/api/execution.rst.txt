.. _api-execution:

Execution backends
==================

.. automodule:: habit.execution
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.execution

Integrator chapter (which backend, continue vs fail_fast, resume):
:doc:`../tutorial/execution`. Runnable demos:
:doc:`../examples/parallel_execution` and
:doc:`../examples/fault_tolerance`. YAML twins:
:doc:`spec`. Habitat CLI / recipe wiring:
:doc:`../configuration/habitat`.

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SerialBackend
   ProcessPoolBackend
   SubjectTimeoutError
   CheckpointStore
   LegacyCheckpointMigrationReport

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   backend_from_policy
   should_use_process_pool
   is_v01_checkpoint_layout
   migrate_v01_checkpoint_if_needed

Backend selection from CLI and run_from_yaml
--------------------------------------------

``habit.execution.backend_from_policy`` (shared by ``cmd_habitat`` and
``run_from_yaml``) mirrors v0.1 ``_should_use_spawn_workers``:

* ``policy.backend == "process"`` →
  :class:`~habit.execution.ProcessPoolBackend.from_policy`
* **or** ``subject_timeout_sec`` is a positive number → ProcessPoolBackend
  even when ``workers == 1`` (timeout isolation needs a child process)
* otherwise → :class:`~habit.execution.SerialBackend` with the policy's
  checkpoint / failure flags only

Therefore timeouts, spawn / graceful shutdown, OOM backoff, GPU capping,
``parallel_mode``, and ``auto_retry_rounds`` apply under ProcessPoolBackend.
True in-process serial requires ``subject_timeout_sec: null`` (and usually
``backend: serial``).

``ProcessPoolBackend.from_policy`` does **not** copy
``strict_checkpoint_hash`` or ``checkpoint_dir`` onto the backend object.
The CLI / recipe resolve ``checkpoint_dir`` (default under ``out_dir``),
bind ``spec.fingerprint()`` onto :class:`~habit.execution.CheckpointStore`,
and raise :class:`~habit.exceptions.CompatibilityError` when
``strict_checkpoint_hash=True`` meets an incompatible fingerprint or a
legacy v0.1 layout.

SerialBackend
-------------

Runs one subject at a time in the current process. Default behind
:meth:`~habit.contracts.Cohort.map`.

.. code-block:: python

   from habit.execution import SerialBackend

   backend = SerialBackend()
   results = list(backend.map(pipeline, cohort))

   # Cohort.map defaults to SerialBackend
   results = cohort.map(pipeline)
   results = cohort.map(pipeline, backend=SerialBackend())

Constructor knobs (checkpoint subset shared with ProcessPoolBackend):

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Parameter
     - Default
     - Role
   * - ``on_subject_failure``
     - ``"continue"``
     - ``"continue"`` or ``"fail_fast"``
   * - ``resume``
     - ``True``
     - Honour checkpoint successes / recorded failures when a store is attached
   * - ``retry_failed_subjects``
     - ``False``
     - Re-run subjects whose checkpoint records a failure
   * - ``force_rerun_subjects``
     - ``()``
     - Subject IDs reprocessed even when a success exists
   * - ``clear_checkpoint_on_success``
     - ``False``
     - Clear the store after a run with zero failures

SerialBackend does **not** accept timeout / spawn / OOM / ``parallel_mode`` /
``auto_retry_rounds``.

Failure policy: ``continue`` vs ``fail_fast``
---------------------------------------------

Both ``SerialBackend`` and ``ProcessPoolBackend`` accept
``on_subject_failure``:

* ``"continue"`` (default) — isolate the exception in that subject's
  ``SubjectResult.error`` and proceed
* ``"fail_fast"`` — re-raise the first subject exception immediately

.. code-block:: python

   from habit.execution import SerialBackend

   backend = SerialBackend(on_subject_failure="continue")
   slots = list(backend.map(op, subjects))
   # slots[i].error is set for failed subjects; others have .value

.. important::

   :meth:`~habit.contracts.Cohort.map` defaults to ``raise_on_failure=True``:
   it aggregates failures and raises :class:`~habit.exceptions.ProcessingError`
   when any slot has an error — **even if** the backend used ``continue``.
   Pass ``raise_on_failure=False`` to receive :class:`~habit.contracts.SubjectResult`
   slots (recipes / CLI do this so a partial cohort can finish, matching
   v0.1). Soft failure also remains available via ``backend.map`` directly.
   See :doc:`../examples/fault_tolerance`.

ProcessPoolBackend
------------------

.. include:: ../_includes/windows_multiprocessing.rst

Multiprocess backend. Constructor surface mirrors :class:`~habit.spec.RunPolicy`
field-by-field (except ``backend``, ``checkpoint_dir``, and
``strict_checkpoint_hash`` — see above).

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Parameter
     - Default
     - Role
   * - ``workers``
     - ``1``
     - Process count (``1`` still runs in a child under ProcessPoolBackend)
   * - ``subject_timeout_sec``
     - ``900.0``
     - Per-subject wall-clock cap; ``None`` disables
   * - ``subject_spawn_timeout_sec``
     - ``120.0``
     - Isolated-mode spawn startup cap; ``None`` disables
   * - ``graceful_shutdown_sec``
     - ``15.0``
     - Seconds between ``terminate()`` and ``kill()`` on timeout
   * - ``on_subject_failure``
     - ``"continue"``
     - ``"continue"`` or ``"fail_fast"``
   * - ``oom_backoff``
     - ``True``
     - Reduce workers after fatal ``MemoryError``
   * - ``oom_reduce_workers_by``
     - ``1``
     - Workers subtracted per OOM step (floor 1)
   * - ``cap_workers_to_gpu_pool``
     - ``False``
     - Clamp ``workers`` to detected GPU pool when set
   * - ``parallel_mode``
     - ``"persistent"``
     - ``"persistent"`` (long-lived workers) or ``"isolated"`` (one child per subject)
   * - ``auto_retry_rounds``
     - ``2``
     - Extra in-run dispatch rounds for failed subjects; ``0`` disables
   * - ``resume``
     - ``True``
     - Honour checkpoint successes / recorded failures
   * - ``retry_failed_subjects``
     - ``False``
     - Re-run checkpointed failures
   * - ``force_rerun_subjects``
     - ``()``
     - Force-rerun subject IDs
   * - ``clear_checkpoint_on_success``
     - ``False``
     - Clear store after a clean run
   * - ``persistent_worker_max_consecutive_failures``
     - ``1``
     - Restart a persistent slot after this many consecutive fatal failures
   * - ``persistent_worker_recycle_after_tasks``
     - ``0``
     - Restart a persistent worker after this many successes (``0`` disables)

.. code-block:: python

   from habit.spec import RunPolicy
   from habit.execution import ProcessPoolBackend

   backend = ProcessPoolBackend(
       workers=4,
       subject_timeout_sec=900.0,
       on_subject_failure="continue",  # or "fail_fast"
       parallel_mode="persistent",     # library default; use "isolated" for per-subject isolation
       auto_retry_rounds=2,            # in-run retries for flaky subjects
       oom_backoff=True,               # reduce workers after MemoryError
   )
   results = list(backend.map(pipeline, cohort))

   # From RunPolicy (does not apply checkpoint_dir / strict_checkpoint_hash)
   backend = ProcessPoolBackend.from_policy(
       RunPolicy(workers=4, backend="process")
   )

Only path-backed lazy subjects cross the process boundary. Exceeding
``subject_timeout_sec`` raises ``SubjectTimeoutError`` (isolated under
``continue``, aborting under ``fail_fast``).

YAML equivalents (habitat v0.1 top-level): ``on_subject_failure``,
``individual_subject_timeout_sec``, ``individual_subject_auto_retry_rounds``,
``retry_failed_subjects``, … — see :doc:`../configuration/habitat` and the
mapping table in :doc:`spec`.

CheckpointStore
---------------

.. code-block:: python

   from habit.execution import CheckpointStore, SerialBackend

   store = CheckpointStore("out/run/.habitat_checkpoint")
   results = cohort.map(
       pipeline,
       backend=SerialBackend(),
       checkpoint=store,
   )
   # Re-running skips subjects already recorded as success

Recorded terminal failures are skipped on resume (v0.1 rule) unless
``retry_failed_subjects=True``. Successful subjects restore ``from_cache=True``.
Pass the same store to :meth:`~habit.recipes.Study.fit_predict` together
with a :class:`~habit.report.Report` so one-step product files (maps,
models, figures) are rewritten from cached payloads on resume
(:doc:`../examples/one_step_habitat`).

On the v1 habitat CLI path, cache keys embed the spec fingerprint, and the
store is also bound to ``run_fingerprint.json``. With
``strict_checkpoint_hash=True``, an incompatible fingerprint raises
:class:`~habit.exceptions.CompatibilityError` (v0.1
``CheckpointConfigHashError`` parity). With ``strict=False``, mismatches
are warned and left unreachable.

A legacy v0.1 ``manifest.json`` / ``subjects/`` tree is **auto-migrated**
on store open (:func:`~habit.execution.migrate_v01_checkpoint_if_needed`):
failed subject IDs become v1 ``.failed`` records under fingerprint-scoped
recipe keys; completed pickles are converted to
:class:`~habit.contracts.Supervoxelization` when geometry and labels are
present, otherwise those subjects are logged and recomputed. The legacy
tree is moved under ``.v01_legacy_archive/``. Corrupt/unreadable
``manifest.json`` still raises :class:`~habit.exceptions.CompatibilityError`.

Exports: ``SerialBackend``, ``ProcessPoolBackend``, ``CheckpointStore``,
``LegacyCheckpointMigrationReport``, ``is_v01_checkpoint_layout``,
``migrate_v01_checkpoint_if_needed``, ``SubjectTimeoutError``,
``backend_from_policy``, ``should_use_process_pool``.
