Code Map
========

This page helps you locate code quickly: it describes the repository layout,
the responsibilities of ``habit/`` subpackages, and where to start when
changing a particular feature.

Repository root
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Path
     - Contents
   * - ``habit/``
     - Main Python package containing the application code.
   * - ``config/``
     - Example and production YAML configurations organized by subsystem.
   * - ``demo_data/``
     - Demo data and generated artifacts, including images and ML tables.
   * - ``launchers/``
     - Windows lightweight-release entry points for installation, optional
       profiles, and the configured HABIT command prompt.
   * - ``tools/bin/``
     - Versioned third-party runtime executables used by the lightweight release.
   * - ``tests/``
     - Pytest tests and executable demo scripts (see :doc:`dev_workflow`).
   * - ``docs/``
     - Documentation; Sphinx sources are under ``docs/source/``.
   * - ``pyproject.toml``
     - Build metadata and entry-point definitions
       (``habit = "habit.cli:cli"``).

``habit/`` package structure
----------------------------

The v1.0 package follows the six-layer architecture (see
:doc:`architecture`): L0 kernels → L1 adapters → L2 contracts → L3 domain →
L4 recipes → L5 cli.

.. mermaid::

   flowchart TD
     ROOT["habit/"]
     ROOT --> L5["L5 — cli.py + commands/"]
     ROOT --> L4["L4 — recipes/"]
     ROOT --> L3["L3 — domain/"]
     ROOT --> L2["L2 — contracts/"]
     ROOT --> L1["L1 — adapters/"]
     ROOT --> L0["L0 — kernels/"]
     ROOT --> SPEC["spec/ — HabitatSpec / MLSpec / LegacyConfigAdapter"]
     ROOT --> EXEC["execution/ — backends & checkpoints"]
     ROOT --> ENG["compat/ — v0.1 engines & bridges"]
     ROOT --> SCH["schemas/ — v0.1 YAML schemas"]
     ROOT --> API["api/ — v0.1 configuration-object facade"]
     ROOT --> UTILS["utils/ — shared utilities"]

     ENG --> PRE["engines/preprocessing/"]
     ENG --> HAB["engines/habitat_analysis/"]
     ENG --> MLC["engines/machine_learning/"]

     L5 --> L4
     L4 --> L3 --> L2 --> L1 --> L0

Top-level package responsibilities
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Package
     - Responsibility
   * - ``habit/kernels/``
     - **L0**: pure numeric kernels (clustering, validation metrics, feature
       math). No I/O, no state, no logging.
   * - ``habit/adapters/``
     - **L1**: data sources and sinks (``DirectoryDataSource``, file
       discovery); the only layer that touches the filesystem for reads.
   * - ``habit/contracts/``
     - **L2**: the typed data model — ``Subject``, ``Cohort``,
       ``FeatureTable``, ``HabitatModel``, ``RunManifest``, outcome types.
   * - ``habit/domain/``
     - **L3**: domain protocols and component registries;
       ``SubjectPipeline`` and ``TablePipeline`` live here.
   * - ``habit/recipes/``
     - **L4**: the standard study designs — ``two_step`` / ``one_step`` /
       ``direct_pooling``, ``train_model`` / ``cross_validate`` /
       ``predict_model``, ``extract_habitat_features`` /
       ``traditional_radiomics``, ``compare_models``, ``run_from_yaml``,
       ``apply_habitat_model``. ML/compare disk figures go through
       ``ml_reporting.py`` / ``comparison_reporting.py``.
   * - ``habit/viz/``
     - Publication figures (``Figure`` in, no filesystem). Classification
       curves and SHAP helpers used by ML and compare reporting.
   * - ``habit/cli.py`` + ``habit/commands/``
     - **L5**: Click command group and command implementations. Commands
       validate YAML, translate via ``LegacyConfigAdapter`` when needed, and
       delegate to L4 recipes. Shared helpers are in ``commands/common.py``.
   * - ``habit/spec/``
     - Spec objects (``HabitatSpec``, ``MLSpec``, ``RunPolicy``) and
       ``LegacyConfigAdapter`` for v0.1 → v1 translation.
   * - ``habit/execution/``
     - Execution backends (serial / multiprocessing) and checkpoint stores.
   * - ``habit/datasets/``
     - Synthetic cohorts and feature tables for examples and tests.
   * - ``habit/plugins/``
     - Plugin discovery and ``list_plugins`` introspection.
   * - ``habit/api/``
     - v0.1 configuration-object facade (``run_ml``, ``run_kfold``,
       ``run_preprocess``, ...), kept for YAML parity.
   * - ``habit/schemas/``
     - Pydantic configuration models for workflows, step parameters, parameter
       registration, and validation (v1.0 canonical location).
   * - ``habit/registry/``
     - Shared registry base (:class:`~habit.registry.base.ClassRegistry`) used
       by plugin factories across engines.
   * - ``habit/compat/engines/preprocessing/``
     - Batch image preprocessing: ``BatchProcessor``,
       ``BasePreprocessor``, ``PreprocessorFactory``, and step implementations
       (v0.1 engine, retained for YAML/CLI parity).
   * - ``habit/compat/engines/habitat_analysis/``
     - Habitat segmentation, clustering features, post-segmentation feature
       extraction, and traditional radiomics (see :doc:`subsystems`).
   * - ``habit/compat/engines/machine_learning/``
     - Legacy tabular ML engine (holdout/K-fold workflows, pickle pipelines).
       CLI train/cv/compare prefer v1 recipes + ``habit.viz``; this tree is
       retained for YAML parity and opaque v0.1 pipeline loads.
   * - ``habit/compat/dicom_sort_runner.py``
     - Standalone DICOM sorting based on dcm2niix; invoked from
       ``habit.recipes.sort_dicom``.
   * - ``habit/utils/``
     - Shared utilities used across subsystems (see below).

Shared utilities: ``habit/utils/``
----------------------------------

By convention, reusable cross-subsystem utilities live here. **All progress
bars must use** ``progress_utils.py``. Common modules include:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Purpose
   * - ``progress_utils.py``
     - **Shared progress bars** (the package standard; do not create local
       tqdm wrappers).
   * - ``yaml_utils.py``
     - YAML read/write helpers.
   * - ``log_utils.py``
     - Logging and ``LoggerManager`` / ``setup_logger``.
   * - ``io_utils.py`` / ``file_system_utils.py``
     - Image/mask path discovery and Windows/WSL path conversion.
   * - ``parallel_utils.py`` / ``parallel_gpu_utils.py``
     - General parallel execution and GPU slot allocation for torch radiomics.
   * - ``visualization_utils.py`` / ``font_config.py``
     - Plotting and font configuration (**plot text must be English**).
   * - ``habitats_results_io.py`` / ``habitat_postprocess_utils.py``
     - Habitat result I/O and post-processing.
   * - ``radiomics_params_utils.py`` / ``torch_radiomics_utils.py``
     - Radiomics parameters and helpers.
   * - ``job_cancel.py``
     - Cancellation detection for long-running tasks.

Cross-subsystem contract files
------------------------------

The following files define package-wide interface contracts. Update them and
run the contract tests when adding a factory or orchestrator:

.. mermaid::

   flowchart LR
     REG["habit/registry/base.py<br/>ClassRegistry"] --> PF["PreprocessorFactory"]
     REG --> MF["ModelFactory"]
     REG --> CF["ClusteringAlgorithmFactory"]
     REG --> EF["FeatureExtractorRegistry"]
     REG --> PP["PreprocessingMethodFactory"]
     REG --> HF["HabitatFeatureFactory"]

     ORC["recipes/habitat.py<br/>two_step / one_step / direct_pooling"] --> DOM["habit/domain/<br/>component registries"]
     ORC2["recipes/modeling.py<br/>train_model / cross_validate / predict_model"] --> TP["TablePipeline"]
     ORC3["recipes/comparison.py<br/>compare_models"] --> EV["domain/evaluation/comparison"]
     ORC3 --> VIZ["habit.viz"]

     TST["tests/test_architecture_contracts.py"] -.-> REG
     TST -.-> ORC

Where to start when changing X
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Goal
     - Starting point
   * - Add or modify a CLI command
     - ``habit/cli.py`` + ``habit/commands/cmd_*.py``
   * - Add a preprocessing step
     - ``habit/compat/engines/preprocessing/`` + ``PreprocessorFactory``
   * - Add a clustering algorithm
     - ``habit/compat/engines/habitat_analysis/clustering/base_clustering.py``
   * - Add a clustering feature extractor
     - ``habit/compat/engines/habitat_analysis/clustering_features/base_extractor.py``
   * - Add a machine-learning model
     - ``habit/compat/engines/machine_learning/models/factory.py``
   * - Add a feature-selection method
     - ``habit/compat/engines/machine_learning/feature_selectors/selector_registry.py``
   * - Change configuration fields or validation rules
     - ``habit/schemas/workflows/`` and ``schemas/steps/``
   * - Change the three habitat pipeline strategies
     - ``habit/recipes/habitat.py`` (v1 recipes) + ``compat/engines/habitat_analysis/pipelines/steps/``
   * - Change the ML training or prediction flow
     - v1: ``habit/recipes/modeling.py`` + ``habit/domain/`` components;
       figures: ``recipes/ml_reporting.py`` + ``habit/viz/``;
       v0.1 engine: ``compat/engines/machine_learning/workflows/`` and
       ``runners/``
   * - Change multi-model comparison (``habit compare``)
     - ``habit/recipes/comparison.py``,
       ``habit/domain/evaluation/comparison.py``,
       ``habit/recipes/comparison_reporting.py``, ``habit/viz/``
   * - Add a class-based factory
     - Subclass ``ClassRegistry`` from ``habit/registry/base.py``;
       follow an existing factory in the same domain.
   * - Add a top-level orchestrator for a new CLI pipeline
     - Implement the recipe in ``habit/recipes/`` and wire the CLI command.
       ``tests/test_architecture_contracts.py`` validates layer dependencies.

.. seealso::

   See :doc:`extension_points` for the complete extension and registration
   reference, and :doc:`../customization/index` for implementation templates.
