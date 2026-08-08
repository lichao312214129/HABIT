Architecture
============

Since v1.0.0 HABIT is **API-first**: the Python API is the product, and the
CLI and YAML configs are thin shells over it. The codebase is organized
as six layers, L0 at the bottom to L5 at the top. A layer may only import
from the layers below it, which keeps the numeric core free of I/O and keeps
configuration parsing out of the algorithms.

Design principles
-----------------

1. **API first**: every study is a Python function call
   (``result = recipes.two_step(cohort, spec)``); YAML and the CLI are
   projections of the same call.
2. **Typed specifications**: analyses are declared as immutable, fingerprinted
   spec objects (:class:`~habit.spec.HabitatSpec`, :class:`~habit.spec.MLSpec`)
   rather than loosely structured dictionaries.
3. **Registry pattern**: algorithms are registered by name instead of being
   hard-coded into workflow logic.
4. **Unified contracts**: data travels between layers as typed contracts
   (:class:`~habit.contracts.Subject`, :class:`~habit.contracts.FeatureTable`,
   :class:`~habit.contracts.HabitatModel`, ...).
5. **Lazy imports**: optional and heavy dependencies load only when needed.

The six layers
--------------

.. mermaid::

   flowchart TD
     subgraph L5["L5 — interfaces (habit/cli.py, habit/commands/)"]
       CLI["CLI commands"]
       YAML["YAML documents"]
     end
     L4["L4 — recipes (habit/recipes/)<br/>two_step · one_step · direct_pooling<br/>train_model · cross_validate · predict_model<br/>extract · radiomics · compare_models<br/>run_from_yaml · apply_habitat_model"]
     L3["L3 — domain (habit/domain/)<br/>protocols + component registries<br/>SubjectPipeline · TablePipeline"]
     L2["L2 — contracts (habit/contracts/)<br/>Subject · Cohort · FeatureTable<br/>HabitatModel · RunManifest"]
     L1["L1 — adapters (habit/adapters/)<br/>DirectoryDataSource · sinks<br/>(the only layer that reads files)"]
     L0["L0 — kernels (habit/kernels/)<br/>pure numpy / SimpleITK math<br/>no I/O, no state, no logging"]

     L5 --> L4
     L4 --> L3
     L3 --> L2
     L2 --> L1
     L1 --> L0

Supporting packages beside the stack: ``habit/spec/`` (spec objects and the
v0.1 → v1 translator), ``habit/execution/`` (parallel backends and
checkpoints), ``habit/datasets/`` (synthetic data), ``habit/plugins/``
(component discovery), ``habit/schemas/`` (v0.1 YAML schemas), and
``habit/compat/`` (the v0.1 engines, kept for YAML parity and legacy files).

Two execution paths
-------------------

**The v1 path (specs and recipes).** Python callers build a spec, assemble a
cohort or feature table through the contracts/adapters, and call a recipe.
The recipe wires domain components from the L3 registries into a pipeline,
runs it through an execution backend, and returns a typed result with a
:class:`~habit.contracts.RunManifest` for provenance:

.. mermaid::

   flowchart TD
     S["HabitatSpec / MLSpec<br/>(immutable, fingerprinted)"] --> R["habit.recipes.*"]
     C["Cohort / FeatureTable"] --> R
     R --> P["SubjectPipeline / TablePipeline<br/>(L3 registries resolve Spec('name', params))"]
     P --> X["Execution backend<br/>serial / multiprocessing"]
     X --> RES["StudyResult / ModelResult<br/>+ RunManifest"]

**The v0.1 path (YAML schemas and configurators).** The CLI still accepts
v0.1 YAML. Commands validate it against the Pydantic schemas in
``habit/schemas/``, translate it with
:class:`~habit.spec.legacy.LegacyConfigAdapter`, and then call the **same**
v1 recipes. The classic schema → configurator → orchestrator chain survives
only inside ``habit/compat/engines/``, which a few workflows (feature
extraction, traditional radiomics, legacy pickle prediction) still route
through:

.. mermaid::

   flowchart TD
     Y["v0.1 YAML"] --> SC["habit/schemas/<br/>Pydantic validation"]
     SC --> TR["LegacyConfigAdapter<br/>translate to v1"]
     TR --> REC["habit.recipes.*<br/>(same as the Python path)"]
     SC -.->|"extract / radiomics /<br/>legacy pickles"| CF["compat configurators<br/>MLConfigurator etc."]
     CF --> ORC["compat orchestrators<br/>BatchProcessor · HabitatAnalysis<br/>KFoldWorkflow"]

Key components
--------------

* **Specs** (``habit/spec/``) declare an analysis as data. A spec is
  immutable, serializable, and fingerprinted, so a saved model always knows
  exactly which analysis produced it.
* **Recipes** (``habit/recipes/``) are the standard study designs. They are
  the only orchestration most users and all entry points need.
* **Contracts** (``habit/contracts/``) are the typed data model; every layer
  speaks in these objects.
* **Domain registries** (``habit/domain/``) map ``Spec("kmeans", {...})``
  component references to implementations.
* **Execution backends** (``habit/execution/``) run subject-parallel work and
  provide checkpointing for interrupted cohorts.
* **Compat engines** (``habit/compat/engines/``) are the pre-v1
  implementations (``BatchProcessor``, ``HabitatAnalysis``,
  ``HoldoutWorkflow``/``KFoldWorkflow``). They remain fully supported for
  YAML-parity workflows but are no longer the architecture's center of
  gravity.

Subsystems
----------

**Preprocessing**: the CLI command ``habit preprocess`` calls the v1 recipe
:func:`habit.recipes.preprocess_images`. The v0.1 engine
(``BatchProcessor`` + ``PreprocessorFactory`` under
``habit/compat/engines/preprocessing/``) remains available through
``habit.api.preprocessing``.

**Habitat analysis** supports three strategies — ``two_step``, ``one_step``,
and ``direct_pooling`` — implemented as v1 recipes over the same domain
components. Fitted state is persisted as a self-describing
``.habitatmodel`` archive for reproducible prediction.

**Machine learning**: the v1 recipes (:func:`~habit.recipes.train_model`,
:func:`~habit.recipes.cross_validate`, :func:`~habit.recipes.predict_model`)
run a :class:`~habit.domain.TablePipeline`, which keeps preprocessing,
feature selection, and the classifier inside one fitted object to prevent
data leakage. Figures go through :mod:`habit.recipes.ml_reporting` and
``habit.viz``. Multi-model comparison is
:func:`~habit.recipes.compare_models`. The v0.1 engine (``workflows/`` +
``runners/`` under ``habit/compat/engines/machine_learning/``) remains for
legacy configuration-object callers and opaque pickle pipeline loads.

CLI-to-core mapping
-------------------

Every CLI command is L5 wiring: parse and validate YAML, translate if
needed, call a recipe, write outputs. No algorithms live in the command
layer.

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - CLI command
     - Configuration
     - v1 entry point
   * - ``preprocess``
     - ``PreprocessingConfig``
     - :func:`habit.recipes.preprocess_images`
   * - ``get-habitat``
     - ``HabitatAnalysisConfig`` → ``LegacyConfigAdapter``
     - :func:`habit.recipes.two_step` / ``one_step`` / ``direct_pooling``
   * - ``extract``
     - ``FeatureExtractionConfig``
     - :func:`habit.recipes.extract_habitat_features` (domain extractors;
       compat fallback only for unregistered plugins)
   * - ``radiomics``
     - ``RadiomicsConfig``
     - :func:`habit.recipes.traditional_radiomics`
   * - ``model``
     - ``MLConfig`` → ``LegacyConfigAdapter``
     - :func:`habit.recipes.train_model` (+ ``ml_reporting`` / ``habit.viz``)
   * - ``cv``
     - ``MLConfig`` → ``LegacyConfigAdapter``
     - :func:`habit.recipes.cross_validate` (+ ``ml_reporting`` / ``habit.viz``)
   * - ``compare``
     - ``ModelComparisonConfig``
     - :func:`habit.recipes.compare_models` (domain evaluation +
       ``comparison_reporting`` / ``habit.viz``)

See :doc:`invariants` for the contracts enforced by architecture tests.
