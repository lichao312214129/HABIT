Errors and optional dependencies
================================

Public exception types (stable):

.. code-block:: python

   from habit import (
       HABITAPIError,
       HabitError,
       ConfigurationError,
       DataFormatError,
       GeometryError,
       OptionalDependencyError,
       ComponentNotFoundError,
       CompatibilityError,
       ProcessingError,
       NotFittedError,
   )

* ``HABITAPIError`` — invalid API use / contract breach
* ``ConfigurationError`` — bad config / Spec
* ``DataFormatError`` — unreadable or ill-formed data
* ``GeometryError`` — incompatible image/mask geometry
* ``OptionalDependencyError`` — missing optional backend
* ``ComponentNotFoundError`` — unknown registry name
* ``CompatibilityError`` — version / format mismatch
* ``ProcessingError`` — runtime processing failure
* ``NotFittedError`` — ``transform`` before ``fit``; single canonical class
  defined in ``habit.exceptions`` that subclasses
  ``sklearn.exceptions.NotFittedError``, so one ``except`` clause catches
  HABIT estimators and sklearn pipelines alike

The canonical import home is ``habit.exceptions``; ``habit.api.exceptions``
remains as a backward-compatible facade.

When each exception is raised
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Exception
     - Typical trigger
   * - ``HABITAPIError``
     - Wrong argument types, unsupported backend name, invalid volume ndim
   * - ``ConfigurationError`` / ``ConfigValidationError``
     - YAML load failure, unknown fields (``extra="forbid"``), schema mismatch;
       CLI surfaces these via ``habit check-config`` without a traceback
   * - ``DataFormatError``
     - Directory cohort has zero complete subjects; ill-formed input tables
   * - ``GeometryError``
     - Image/mask mismatch under ``GeometryPolicy.STRICT`` (API radiomics default)
   * - ``OptionalDependencyError``
     - Missing ``radiomics`` / ``torch`` / GUI stacks; subclass of ``ImportError``
   * - ``ComponentNotFoundError``
     - Unknown registry / plugin name for a domain
   * - ``CompatibilityError``
     - ``HabitatModel.load``: bad ZIP, wrong ``format``, newer ``format_version``;
       never returns a "plausible" wrong habitat map. Also raised by
       :class:`~habit.execution.CheckpointStore` when
       ``strict_checkpoint_hash=True`` meets an incompatible fingerprint or
       legacy v0.1 layout
   * - ``ProcessingError``
     - Pipeline / subject failure; **also** default
       :meth:`~habit.contracts.Cohort.map` when any subject failed (even if
       the backend used ``continue``). Pass ``raise_on_failure=False`` for
       soft failure
   * - ``NotFittedError``
     - Estimator ``transform`` / ``predict`` before ``fit``

Soft-failure switches (not exceptions)
--------------------------------------

These APIs collect errors instead of raising immediately:

* ``extract_batch(..., fail_fast=False)`` → ``FeatureTableResult.failures``
  (:doc:`image_io`)
* ``load_plugins(strict=False)`` → ``PluginLoadReport.failures`` (:doc:`plugins`)
* ``SerialBackend`` / ``ProcessPoolBackend`` with
  ``on_subject_failure="continue"`` → error slots in ``SubjectResult``
  (:doc:`execution`)

.. important::

   ``backend.map(..., on_subject_failure="continue")`` isolates failures;
   default ``cohort.map(op, backend=...)`` **still raises** ``ProcessingError``
   if any slot failed. Pass ``raise_on_failure=False`` (recipes / CLI) or call
   the backend directly for soft failure (see
   :doc:`../examples/fault_tolerance`).

Probe optional stacks without importing heavy backends:

.. code-block:: python

   import habit

   if habit.is_available("torch"):
       ...
   if habit.is_available("radiomics"):
       ...

Logger helper for scripts:

.. code-block:: python

   from habit import setup_logger

   logger = setup_logger(
       name="study",
       output_dir="out",
       log_filename="run.log",
   )
