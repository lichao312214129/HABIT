Developer Guide
===============

This chapter is for HABIT contributors and developers building extensions. It
is the entry point for understanding HABIT at the architectural and
implementation levels. Researchers who only want to use the software should
start with :doc:`../tutorial/installation`.

The chapter follows three stages: understand the system, locate the relevant
code, and extend or contribute to the project.

.. toctree::
   :maxdepth: 2
   :caption: 1. Understand HABIT (design and mental model)

   philosophy
   mental_model
   architecture
   request_lifecycle

.. toctree::
   :maxdepth: 2
   :caption: 2. Code organization (where to look)

   repo_layout
   configuration_system
   subsystems

.. toctree::
   :maxdepth: 2
   :caption: 3. Extensions and contributions (how to work)

   extension_points
   invariants
   dev_workflow
   contributing

Recommended reading order
--------------------------

**New to HABIT**

1. :doc:`philosophy` — Limitations of the research setting, and why the
   API is the product (CLI / YAML are shells).
2. :doc:`mental_model` — Core terminology and the system mental model.
3. :doc:`architecture` — Layers, configuration assembly, and CLI-to-core
   mapping.
4. :doc:`request_lifecycle` — Follow one command through the complete request
   lifecycle.

**Changing the code**

5. :doc:`repo_layout` — Where each type of code lives and where to start.
6. :doc:`configuration_system` — The YAML → Schema → Configurator → runtime
   object pipeline.
7. :doc:`subsystems` — Internal execution flows for habitat analysis and
   machine learning.

**Extending or contributing**

8. :doc:`extension_points` — The eight extension points and their registration
   mechanisms.
9. :doc:`invariants` — Rules that must not be broken before changing code.
10. :doc:`dev_workflow` — Environment setup, tests, and implementation steps.
11. :doc:`contributing` — Pull-request and coding conventions.

.. note::

   This chapter focuses on architecture and mechanisms. Complete,
   copy-ready component templates are available in
   :doc:`../customization/index`.

Environment summary: Python 3.10–3.14 (day-to-day: ``py310``),
``pip install -e .``, and ``pytest tests/``.

Related top-level docs under **Development**:
:doc:`../customization/index` (copy-ready extension templates) and
:doc:`../reference/upstream_libraries` (third-party library notes).
Public Python symbols live under **Python API** (:doc:`../api/index`).
