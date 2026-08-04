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
