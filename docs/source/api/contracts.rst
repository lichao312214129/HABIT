API contracts, provenance, and errors
=====================================

Shared types used by public ``run_*`` helpers and object APIs.

Workflow results
----------------

::

   from habit import WorkflowResult

   # Most run_* helpers return WorkflowResult with output_dir / data / metadata.

Provenance
----------

::

   from habit import RunManifest, create_run_manifest, write_run_manifest

Installers and workflow facades write a run manifest next to outputs so
experiments remain reproducible.

Exceptions
----------

Public error types include ``HABITAPIError``, ``ConfigurationError``,
``DataFormatError``, ``GeometryError``, ``OptionalDependencyError``,
``ComponentNotFoundError``, ``CompatibilityError``, ``ProcessingError``, and
``NotFittedError``.

Module reference
----------------

.. automodule:: habit.api.contracts
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.api.provenance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.api.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
