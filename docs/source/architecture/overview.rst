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

- **Configurator layer** — shared base in
  :py:mod:`habit.core.common.configurators.base` (``BaseConfigurator``);
  domain factories live next to their packages:
  ``habit.core.habitat_analysis.configurator.HabitatConfigurator``,
  ``habit.core.machine_learning.configurator.MLConfigurator``,
  ``habit.core.preprocessing.configurator.PreprocessingConfigurator``.
  Each cmd_* entry imports only the domain configurator it needs; ``common``
  does not re-export the three domain factories at package root.

- **Domain subpackages** — ``habitat_analysis``, ``machine_learning``, and
  ``preprocessing``. The three never import one another; they exchange data
  through file artefacts (CSV, NRRD, ``.pkl``).

- **Pipeline / manager layer (per domain)** — each domain has its own
  sklearn-style pipeline. In ``habitat_analysis`` the pipeline is
  ``HabitatPipeline`` driven by ``HabitatAnalysis`` (a deep module). In
  ``machine_learning`` it is ``workflows.base.BaseWorkflow`` + ``PipelineBuilder``.

- **Utilities** — ``habit/utils/`` holds I/O, logging, the unified
  ``CustomTqdm`` progress bar, parallel helpers, and visualization style.

Where to read next
------------------

For the full developer-facing architecture (dependency rules, configuration
system, data flow per domain, persistence formats, and the list of currently
known architecture concerns), see:

- :doc:`../development/architecture` — project-wide architecture, including
  the ``habitat_analysis`` ②-a section that describes recipe dispatch,
  service injection (``_PIPELINE_SERVICE_ATTRS``), persistence format
  and design decisions.

- :doc:`../development/module_architecture` — per-subpackage code
  organisation, including the per-step I/O contract and state-management
  rules for every pipeline step in ``habitat_analysis``.

.. note::
   In **habitat_analysis**, the pre-V1 "Strategy Layer"
   (``TwoStepStrategy`` / ``OneStepStrategy`` / ``DirectPoolingStrategy``)
   and the habitat ``pipeline_builder`` helper are gone: mode dispatch is a
   recipe map inside :class:`~habit.core.habitat_analysis.HabitatAnalysis`.
   The **machine_learning** subpackage still has its own
   ``PipelineBuilder`` (unrelated to the removed habitat helper). Docs that
   conflate the two are outdated.
