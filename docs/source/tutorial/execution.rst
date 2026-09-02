Parallel execution and fault tolerance
======================================

:class:`~habit.HabitatSpec` declares **what** to compute.
:class:`~habit.RunPolicy` (and an execution backend) declare **how** to
schedule subjects. Scientific numbers do not change when you switch
serial ↔ process pool if ``random_seed`` is fixed.

This page is the integrator chapter. Knob-by-knob reference:
:doc:`../api/execution`. Runnable demos:
:doc:`../examples/parallel_execution` and
:doc:`../examples/fault_tolerance`.

Beginners can ignore this page until a cohort is slow or one subject
crashes the run. The quickstarts use a small process pool already
(``workers=2``).

Pick a backend
--------------

:func:`~habit.execution.backend_from_policy` chooses:

* :class:`~habit.execution.ProcessPoolBackend` when
  ``policy.backend == "process"``, **or** when
  ``subject_timeout_sec`` is a positive number (timeout isolation needs
  a child process)
* :class:`~habit.execution.SerialBackend` otherwise

True in-process serial needs ``backend="serial"`` **and**
``subject_timeout_sec=None``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Use
     - Policy
   * - Debug one subject / notebook
     - ``RunPolicy(workers=1, backend="serial", subject_timeout_sec=None)``
   * - Cohort, default isolation
     - ``RunPolicy(workers=2, backend="process", subject_timeout_sec=900.0)``
   * - Fresh process per subject
     - same, plus ``parallel_mode="isolated"`` (higher spawn cost)

.. include:: ../_includes/windows_multiprocessing.rst

Pass the backend into a recipe or into :meth:`~habit.contracts.Cohort.map`::

   from habit import RunPolicy
   from habit.execution import backend_from_policy
   import habit.recipes as recipes

   policy = RunPolicy(
       workers=2,
       backend="process",
       subject_timeout_sec=900.0,
       on_subject_failure="continue",
       resume=True,
   )
   backend = backend_from_policy(policy)
   result = recipes.Study(spec=spec).fit_predict(cohort, backend=backend)

Atomic path (no ``Study``)::

   from habit.execution import SerialBackend

   maps = cohort.map(pipe, backend=SerialBackend())

One subject does not need a backend: ``habitat_map = pipe(subject)``.

Failure policy
--------------

Both backends accept ``on_subject_failure``:

* ``"continue"`` (default) — record the exception on that subject's slot
  and keep going
* ``"fail_fast"`` — raise on the first subject exception

:meth:`~habit.contracts.Cohort.map` still raises
:class:`~habit.exceptions.ProcessingError` when any slot failed, **even
if** the backend used ``continue``. Recipes / CLI pass
``raise_on_failure=False`` so a partial cohort can finish. Embedders who
want the same::

   slots = cohort.map(pipe, backend=backend, raise_on_failure=False)

Or call ``backend.map(pipe, cohort)`` and read ``slot.error``.

Per-subject wall-clock cap: ``subject_timeout_sec`` (ProcessPool only).
Expiry raises ``SubjectTimeoutError`` (isolated under ``continue``).

In-run retries of flaky subjects: ``auto_retry_rounds`` (ProcessPool).
After a fatal ``MemoryError``, ``oom_backoff=True`` reduces ``workers``.

Resume and checkpoints
----------------------

Attach a :class:`~habit.execution.CheckpointStore` so a second run skips
subjects already recorded as success::

   from habit.execution import CheckpointStore, SerialBackend

   store = CheckpointStore("out/run/.habitat_checkpoint")
   maps = cohort.map(
       pipe,
       backend=SerialBackend(),
       checkpoint=store,
   )

Recorded **failures** stay skipped unless ``retry_failed_subjects=True``.
Force a few IDs with ``force_rerun_subjects``.

On the habitat recipe / CLI path, cache keys embed the spec fingerprint.
Changing extractors or preprocess stages without a new store looks like
a cache hit on the **wrong** definition — treat that as a new run.

Other soft-failure knobs
------------------------

Not execution backends, but the same "raise vs continue" idea:

* **GeometryPolicy** — image vs mask grid (``STRICT`` / ``RESAMPLE_MASK`` / …)
* **extract_batch(fail_fast=)** — feature extraction over pairs
* **load_plugins(strict=)** — missing third-party plugins
* **HabitatModel.load** — refuses a file that is not
  ``habit.habitatmodel`` (:class:`~habit.exceptions.CompatibilityError`)

Walkthrough: :doc:`../examples/fault_tolerance`.

YAML
----

Native v1 documents use a top-level ``policy:`` block (``workers``,
``backend``, ``subject_timeout_sec``, …). Older YAML keeps the same
knobs at the document top level (``processes``,
``individual_subject_timeout_sec``, …). Field rename table:
:doc:`../api/spec`. Full habitat reference:
:doc:`../configuration/habitat`.

Next
----

* Habitat core: :doc:`habitat_analysis`
* Embed operators: :doc:`../examples/habitat_atomic_ops`
* API reference: :doc:`../api/execution`
* Demos: :doc:`../examples/parallel_execution` ·
  :doc:`../examples/fault_tolerance`
