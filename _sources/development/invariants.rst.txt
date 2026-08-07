Invariants and Architecture Contracts
=====================================

This page lists rules that must not be broken. They protect scientific
correctness, implementation consistency, and predictable public behavior.
Most of these rules are checked automatically by
``tests/test_architecture_contracts.py``.

.. important::

   Run the architecture contract tests before submitting changes:

   .. code-block:: bash

      pytest tests/test_architecture_contracts.py -m unit

Scientific correctness
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Invariant
     - Rationale
   * - **Prevent data leakage**
     - Feature selection, scaling, and resampling must be inside one sklearn
       Pipeline and fitted only on training folds.
   * - **Train/predict symmetry**
     - Predict must reuse the clustering centers, scaler parameters, selected
       features, and other state learned by fit.
   * - **Controlled randomness**
     - Every stochastic step must accept and propagate a random seed so the
       same configuration can be reproduced.

Configuration rules
-------------------

* Root configuration models use ``extra='forbid'`` where strict validation is
  required. New fields must be declared in the schema.
* ``habit/schemas/`` is the source of truth for schema definitions.
  Compatibility modules may re-export schemas but must not define duplicates.
* A component with configurable ``params`` must define and register a Pydantic
  parameter model.
* v1 spec objects (``habit/spec/``) are immutable and fingerprinted; a fitted
  model must always be traceable to the exact spec that produced it.

Registry contracts
------------------

All registries must:

* inherit from the appropriate ``_BaseRegistry`` subclass;
* expose ``register``, ``get``, ``available``,
  ``register_params_model``, and ``get_params_model``;
* keep an independent ``_registry`` dictionary;
* return a list from ``available()``.

Class factories additionally provide ``create()``. Callable registries provide
their callable-entry accessors.

Orchestrator contracts
----------------------

Every top-level orchestrator must expose the terminal methods declared in
``ORCHESTRATOR_CONTRACT``. Batch processors and workflows normally expose
``run()``; habitat analysis exposes ``fit()`` and ``predict()``.

Engineering conventions
-----------------------

* Use ``habit/utils/progress_utils.py`` for all progress bars.
* Put reusable cross-subsystem utilities in ``habit/utils/``.
* Text generated inside plots must be English.
* Import heavy optional dependencies lazily inside command or factory methods.
* Keep business logic out of the command layer: commands are L5 wiring that
  delegate to ``habit/recipes/``; v0.1 engine logic lives in
  ``compat/engines/*/run.py``.
* Respect the layer direction: L0 kernels stay pure (no I/O, state, or
  logging) and no layer imports from a layer above it.
* Annotate function inputs and outputs explicitly, and write code comments in
  English.

See also
--------

* :doc:`philosophy` for the design rationale.
* :doc:`extension_points` and :doc:`dev_workflow` for implementation steps.
