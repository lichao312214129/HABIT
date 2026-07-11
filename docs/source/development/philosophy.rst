Design Philosophy
=================

This page explains why HABIT has its current architecture. The architecture
page describes what the system is, while the extension guide describes how to
change it. Understanding the intent and trade-offs behind the design helps
contributors make decisions consistent with the rest of the project.

Research constraints
--------------------

Medical-image habitat analysis has five recurring engineering constraints:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constraint
     - Meaning
   * - **Reproducibility**
     - The same data and parameters must reproduce the same result.
   * - **Batch processing**
     - Large cohorts require parallel execution, checkpoints, and isolated
       subject failures.
   * - **Pluggable algorithms**
     - Clustering, feature, model, and selector implementations must be
       replaceable without rewriting the workflow.
   * - **Train/predict consistency**
     - Inference must reuse exactly the processing learned during training.
   * - **CLI/GUI parity**
     - The command line and graphical interfaces must use the same behavior.

HABIT addresses these constraints through five design pillars.

Five design pillars
-------------------

.. mermaid::

   flowchart LR
     P1["Reproducible"] --> S1["Config as interface"]
     P1 --> S2["Typed schema"]
     P2["Batch subjects"] --> S5["Thin command, thick core"]
     P3["Pluggable algorithms"] --> S3["Registry decoupling"]
     P4["Train = predict"] --> S4["Unified contracts"]
     P5["CLI = GUI"] --> S5

1. **Configuration is the interface.** Workflow parameters are represented by
   a versionable configuration object. YAML files and Python dictionaries use
   the same validation path.
2. **Schemas fail fast.** Pydantic validates configuration before computation.
   Unknown fields are rejected where the schema uses ``extra='forbid'``.
3. **Registries decouple algorithms.** Components are registered by name, so
   a workflow can select an implementation from configuration.
4. **Contracts unify execution.** Factories and orchestrators expose stable
   interfaces such as ``run()`` or ``fit()``/``predict()``.
5. **Commands remain thin.** CLI, GUI, and Python API entry points delegate to
   the same core workflow implementation.

Trade-offs
----------

These rules deliberately constrain implementation freedom in exchange for
reproducibility, testability, and consistent behavior across subsystems.
Configuration requires explicit schema maintenance, registries require import
discipline, and standardized orchestrators must expose the declared terminal
methods. The non-negotiable rules are documented in :doc:`invariants`.

See also
--------

* :doc:`configuration_system` for configuration assembly.
* :doc:`request_lifecycle` for a complete command lifecycle.
* :doc:`mental_model` for the core vocabulary.
