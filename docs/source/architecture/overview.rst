Architecture Overview
=====================

HABIT V1 is organised as four core subpackages on top of a shared utilities
layer, wired together by a single dependency-injection point.

High-level layers
-----------------

- **Entry layer** — two parallel entry points: the lazy Python API
  (``habit/__init__.py``) and the Click CLI (``habit/cli.py`` +
  ``habit/cli_commands/commands/``). The CLI is the recommended path
  for end users; the legacy ``scripts/app_*.py`` dual-track runners
  have been removed in V1.
- **Service layer** — domain-specific configurators in
  ``habit.core.common.configurators``: ``HabitatConfigurator`` /
  ``MLConfigurator`` / ``PreprocessingConfigurator``, each extending
  ``BaseConfigurator``. Each cmd_* entry imports only the domain
  configurator it actually needs; ``common`` no longer pulls every
  business subpackage into the import surface.
- **Domain subpackages** — ``habitat_analysis``, ``machine_learning``, and
  ``preprocessing``. The three never import one another; they exchange data
  through file artefacts (CSV, NRRD, ``.pkl``).
- **Pipeline / manager layer (per domain)** — each domain has its own
  sklearn-style pipeline. In ``habitat_analysis`` the pipeline is
  ``HabitatPipeline`` driven by ``HabitatAnalysis`` (a deep module). In
  ``machine_learning`` it is ``BaseWorkflow`` + ``PipelineBuilder``.
- **Utilities** — ``habit/utils/`` holds I/O, logging, the unified
  ``CustomTqdm`` progress bar, parallel helpers, and visualization style.

Where to read next
------------------

For the full developer-facing architecture (dependency rules, configuration
system, data flow per domain, persistence formats, and the list of currently
known architecture concerns), see:

- :doc:`../development/architecture` — project-wide architecture.
- ``habit/core/habitat_analysis/ARCHITECTURE.md`` — internals of the
  habitat-analysis subpackage (recipe dispatch, manager injection,
  pipeline steps).
- ``habit/core/habitat_analysis/PIPELINE_DESIGN.md`` — per-step input/output
  contracts.

.. note::
   The pre-V1 "Strategy Layer" (``TwoStepStrategy`` / ``OneStepStrategy``
   / ``DirectPoolingStrategy``) and the standalone ``pipeline_builder`` module
   have been removed. Mode dispatch is now handled by a recipe dictionary
   inside ``HabitatAnalysis``. Older docs that mention those classes are
   out of date.
