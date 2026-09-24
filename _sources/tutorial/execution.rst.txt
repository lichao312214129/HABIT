:orphan:

Parallel execution and fault tolerance
======================================

:class:`~habit.spec.HabitatSpec` declares **what** to compute.
:class:`~habit.spec.RunPolicy` (and an execution backend) declare **how** to
schedule subjects. Scientific numbers do not change when you switch
serial ↔ process pool if ``random_seed`` is fixed.

This page is the integrator chapter. Knob-by-knob reference:
:doc:`../api/execution`. Runnable demos:
:doc:`/auto_examples/07_parallel/index`.

Beginners can ignore this page until a cohort is slow or one subject
crashes the run. The quickstarts use a small process pool already
(``workers=2``).

Pick a backend
--------------

:func:`~habit.execution.backend_from_policy` chooses:

* :class:`~habit.execution.ProcessPoolBackend` when
  ``policy.backend == "process"``, **or** ``workers > 1``, **or**
  ``parallel_mode == "isolated"``
* :class:`~habit.execution.SerialBackend` otherwise

A positive ``subject_timeout_sec`` alone does **not** force spawn (the
default is ``900.0``). Timeout isolation still needs ``backend="process"``
or ``parallel_mode="isolated"``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Use
     - Policy
   * - Debug one subject / notebook
     - ``RunPolicy(workers=1, backend="serial")``
   * - Cohort, default isolation
     - ``RunPolicy(workers=2, backend="process", subject_timeout_sec=900.0)``
   * - Fresh process per subject
     - same, plus ``parallel_mode="isolated"`` (higher spawn cost)

.. include:: ../_includes/windows_multiprocessing.rst

Pass the backend into a recipe or into :meth:`~habit.contracts.Cohort.map`::

   from habit.contracts import cohort_from_directory
   from habit.datasets import fetch_demo
   from habit.execution import backend_from_policy
   from habit.recipes import one_step_habitat
   from habit.spec import RunPolicy

   # Change DATA / MODALITIES / ROI to your preprocessed layout
   DATA = fetch_demo()  # or "demo_data/preprocessed"
   MODALITIES = ("LAP",)
   ROI = "LAP"
   cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
   study = one_step_habitat(
       modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
   )
   policy = RunPolicy(
       workers=1,
       backend="serial",
       subject_timeout_sec=None,
       on_subject_failure="continue",
       resume=True,
   )
   backend = backend_from_policy(policy)
   result = study.fit_predict(cohort, backend=backend)

Atomic path (no ``Study``): extract texture, preprocess, fit in this
process, then ``backend.map`` the assigner on units already in memory.
Subject versus cohort preprocessing of that texture, and the equivalent
``HabitatSpec``, is
:doc:`/auto_examples/02_stages/plot_06_texture_preprocessing`.
Serial versus pooled runs of one study, with timings, are on the first
gallery page.

* :doc:`/auto_examples/02_stages/plot_06_texture_preprocessing`
* :doc:`/auto_examples/07_parallel/plot_01_backends`

Failure policy
--------------

Both backends accept ``on_subject_failure``:

* ``"continue"`` (default) — record the exception on that subject's slot
  and keep going
* ``"fail_fast"`` — raise on the first subject exception

:meth:`~habit.contracts.Cohort.map` still raises
:class:`~habit.exceptions.ProcessingError` when any slot failed, **even
if** the backend used ``continue``. Recipes / CLI pass
``raise_on_failure=False`` so a partial cohort can finish. Calling
``backend.map`` directly leaves the exception on ``slot.error``.

Both, on a study with one incomplete subject:
:doc:`/auto_examples/07_parallel/plot_01_backends`.

Per-subject wall-clock cap: ``subject_timeout_sec`` (ProcessPool only).
Expiry raises ``SubjectTimeoutError``.

In-run retries of flaky subjects: ``auto_retry_rounds`` (ProcessPool).
After a fatal ``MemoryError``, ``oom_backoff=True`` reduces ``workers``.

Resume and checkpoints
----------------------

Attach a :class:`~habit.execution.CheckpointStore` so a second run skips
subjects already recorded as success:
:doc:`/auto_examples/07_parallel/plot_01_backends`.

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
* Demos: :doc:`/auto_examples/07_parallel/index`
