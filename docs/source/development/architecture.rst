Architecture
============

Since v1.0.0 HABIT is **API-first**: the Python API is the product, and the
CLI and YAML configs are thin shells over it. The codebase is organized
as six layers, L0 at the bottom to L5 at the top. A layer may only import
from the layers below it, which keeps the numeric core free of I/O and keeps
configuration parsing out of the algorithms.

Read the public Developer docs in this order:

1. **This page** — layers, public API boundary, invariants.
2. :doc:`contributing` — environment, tests, and pull requests.
3. :doc:`../customization/index` — registry and entry-point plugins.

Design principles
-----------------

1. **API first**: every study is a :class:`~habit.recipes.Study`
   (``result = recipes.Study(spec=spec).fit_predict(cohort)`` with
   ``HabitatSpec.stages``); YAML and the CLI are projections of the same
   call. Habitat factories (``two_step_habitat`` / ``one_step_habitat`` /
   ``direct_pooling_habitat``) return a Study with a declared design.
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
     L4["L4 — recipes (habit/recipes/)<br/>Study.fit / fit_predict / predict<br/>two_step_habitat · one_step_habitat · direct_pooling_habitat<br/>train_model · cross_validate · predict_model<br/>extract · radiomics · compare_models · run_from_yaml"]
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

Public Python symbols are listed under :doc:`../api/index`. A third party
should be able to call ``op(subject)`` or ``Study(spec).fit_predict(cohort)``
without HABIT's directory layout. If ``habit/cli.py`` and ``habit/commands/``
were deleted, the scientific capability would still be callable.

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

Maintainer notes for the YAML loader, PathResolver, and compat engine
tours live in the repository at ``developer/sphinx_archive/``
(``configuration_system.rst``, ``request_lifecycle.rst``,
``subsystems.rst``).

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

**Habitat analysis** supports three strategies — ``two_step``, ``one_step``,
and ``direct_pooling`` — implemented as v1 recipes over the same domain
components. Fitted state is persisted as a self-describing
``.habitatmodel`` archive for reproducible prediction.

**Preprocessing** and **tabular machine learning** are supporting shells:
``habit preprocess`` calls :func:`habit.recipes.preprocess_images`;
:func:`~habit.recipes.train_model` / :func:`~habit.recipes.cross_validate` /
:func:`~habit.recipes.predict_model` run a
:class:`~habit.pipeline.TablePipeline`. Compat engines under
``habit/compat/engines/`` remain for YAML parity. Those trees are frozen
for product work; habitat analysis is the extension surface.

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
     - :class:`habit.recipes.Study` (factories: ``two_step_habitat`` /
       ``one_step_habitat`` / ``direct_pooling_habitat``)
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

Design philosophy
-----------------

Imaging papers need **repeatable** and **reproducible** habitat maps.
Voxel intensities depend on scanner and preprocessing; clustering depends
on :math:`k` and the feature vector; a one-voxel mask shift can rewrite a
radiomic map. HABIT does not remove that physics. It makes the choices that
change the map **explicit, typed, and carried with the result**
(:class:`~habit.contracts.habitat.HabitatModel`,
:class:`~habit.spec.specs.Spec`,
:meth:`~habit.contracts.RunManifest.describe_methods`). A green unit test
is not a multi-centre replication.

The product answer is a **library API** that can be copied into a notebook
or another pipeline. CLI and YAML are shells over that API, not a second
science stack. Defaults (for k-means habitat count: inertia **elbow**,
:math:`k\in[2,10]`) are a starting protocol, not a claim that the default
is optimal for every tumour.

Five engineering pillars follow from that:

1. **Configuration is an interface**, not a layer — YAML and Python share
   one meaning via :class:`~habit.spec.specs.Spec`.
2. **Schemas fail fast** — Pydantic validates before computation;
   ``extra='forbid'`` where strict.
3. **Registries decouple algorithms** — swap an implementation by name.
4. **Train and predict share a contract** — inference reuses fitted state,
   including cohort-level preprocessing on
   :class:`~habit.contracts.habitat.HabitatModel`.
5. **Commands stay thin** — L5 delegates to ``habit.recipes`` /
   ``habit.api``.

Habitat and API glossary
------------------------

Task-level habitat walkthroughs live in the Habitat Guide
(:doc:`../auto_examples/index`). The table below is the **developer**
vocabulary used in this chapter.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Term
     - Meaning
   * - **Voxel feature**
     - A per-voxel vector (intensity, kinetics, local radiomics, …)
       that clustering reads.
   * - **Supervoxel**
     - A local group of similar voxels in one subject; intermediate
       product of the ``two_step`` strategy.
   * - **Habitat**
     - An image-phenotype region inside a tumor, stored as an integer
       label image.
   * - **Habitat feature**
     - A downstream quantity after habitat maps exist (volume, MSI, ITH,
       radiomics, graph metrics, …).
   * - **Clustering mode**
     - ``two_step``, ``one_step``, or ``direct_pooling``.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Role
     - Responsibility
     - Representative symbol
   * - **Spec**
     - Declares an analysis as immutable, fingerprinted data.
     - ``HabitatSpec``, ``MLSpec``, ``Spec``
   * - **Recipe**
     - Assembles domain components, executes, returns a typed result.
     - ``recipes.Study(...).fit_predict``, ``recipes.train_model``
   * - **Contract**
     - Typed objects that travel between layers.
     - ``Subject``, ``Cohort``, ``FeatureTable``, ``HabitatModel``
   * - **Pipeline**
     - Executable object a recipe builds; fitted state lives here.
     - ``SubjectPipeline``, ``TablePipeline``
   * - **DataSource / Sink**
     - L1 adapters; the only place files are read.
     - ``DirectoryDataSource``
   * - **Component registry**
     - Maps ``Spec("name", params)`` to implementations.
     - ``VoxelFeatureExtractorRegistry``, ``ComponentRegistry``

On the v0.1 YAML path, a **Configurator** assembles and an **Orchestrator**
executes; a **Factory** resolves names. Those roles still describe
``habit/compat/engines/``. On the v1 path the recipe plays all three,
driven by a spec.

Invariants
----------

These rules must not be broken. Most are checked by
``tests/test_architecture_contracts.py``.

.. important::

   Run the architecture contract tests before submitting changes::

      pytest tests/test_architecture_contracts.py -m unit

**Scientific correctness**

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

**Configuration**

* Root configuration models use ``extra='forbid'`` where strict validation is
  required. New fields must be declared in the schema.
* ``habit/schemas/`` is the source of truth for schema definitions.
  Compatibility modules may re-export schemas but must not define duplicates.
* A component with configurable ``params`` must define and register a Pydantic
  parameter model (v0.1 YAML) or expose them on the component constructor
  (v1 ``Registry.create``).
* v1 spec objects (``habit/spec/``) are immutable and fingerprinted; a fitted
  model must always be traceable to the exact spec that produced it.

**Registry contracts**

All registries must:

* inherit from the appropriate ``_BaseRegistry`` subclass;
* expose ``register``, ``get``, ``available``,
  ``register_params_model``, and ``get_params_model``;
* keep an independent ``_registry`` dictionary;
* return a list from ``available()``.

Class factories additionally provide ``create()``. Callable registries provide
their callable-entry accessors.

**Orchestrator contracts**

Every top-level orchestrator must expose the terminal methods declared in
``ORCHESTRATOR_CONTRACT``. Batch processors and workflows normally expose
``run()``; habitat analysis exposes ``fit()`` and ``predict()``.

**Engineering conventions**

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

Where to look in the repo
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Path
     - Responsibility
   * - ``habit/kernels/``
     - **L0** numeric kernels. No I/O, no state, no logging.
   * - ``habit/adapters/``
     - **L1** data sources and sinks.
   * - ``habit/contracts/``
     - **L2** ``Subject``, ``Cohort``, ``FeatureTable``, ``HabitatModel``.
   * - ``habit/domain/``
     - **L3** protocols, registries, ``SubjectPipeline``.
   * - ``habit/recipes/``
     - **L4** ``Study`` and habitat / extract / ML recipes.
   * - ``habit/cli.py``, ``habit/commands/``
     - **L5** Click wiring only.
   * - ``habit/spec/``
     - ``HabitatSpec`` / ``MLSpec`` / ``LegacyConfigAdapter``.
   * - ``habit/execution/``
     - Parallel backends and checkpoints.
   * - ``habit/plugins/``
     - Entry-point discovery (``list_plugins``).
   * - ``habit/compat/``
     - Frozen v0.1 engines (YAML parity).
   * - ``tests/``
     - Pytest; architecture contracts in
       ``tests/test_architecture_contracts.py``.

Start here when changing habitat analysis:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Goal
     - Starting point
   * - Add a voxel / supervoxel / habitat plugin
     - :doc:`../customization/index` and the matching L3 registry
   * - Change the three habitat strategies
     - ``habit/recipes/habitat.py``
   * - Change CLI wiring
     - ``habit/cli.py`` + ``habit/commands/cmd_*.py`` (keep commands thin)
   * - Change a numeric kernel
     - ``habit/kernels/`` (definition unchanged unless the Spec changes)

The longer package map (utils inventory, compat engine directories, ML
starting points) is archived at ``developer/sphinx_archive/repo_layout.rst``.

See also
--------

* :doc:`contributing` — environment, tests, pull requests.
* :doc:`../customization/index` — habitat registry plugins.
* :doc:`../how_to/habitat_components` — built-in ``Spec`` names.
* :doc:`../reference/upstream_libraries` — third-party library notes.
