Code Map
========

This page helps you locate code quickly: it describes the repository layout,
the responsibilities of ``habit/`` subpackages, and where to start when
changing a particular feature.

Repository root
--------

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
     - Demo data and generated artifacts, including images, ML tables, and
       registration executables.
   * - ``tests/``
     - Pytest tests and executable demo scripts (see :doc:`dev_workflow`).
   * - ``docs/``
     - Documentation; Sphinx sources are under ``docs/source/``.
   * - ``habit-gui/``
     - Optional standalone Web GUI (FastAPI + React + bridge), not checked out
       by default. See ``.gitignore``; ``habit gui`` searches for a sibling
       ``habit-gui/`` or bundled ``habit/_gui_bundle``.
   * - ``pyproject.toml``
     - Build metadata and entry-point definitions
       (``habit = "habit.cli:cli"``).

``habit/`` package structure
-----------------

.. mermaid::

   flowchart TD
     ROOT["habit/"]
     ROOT --> CLI["cli.py — Click command group"]
     ROOT --> CMD["commands/ — cmd_*.py (active CLI impl)"]
     ROOT --> CORE["core/ — business logic"]
     ROOT --> UTILS["utils/ — shared utilities"]

     CORE --> COM["common/ — configs, configurators, contracts"]
     CORE --> SCH["schemas/ — workflow & step schemas, registry, reflect"]
     CORE --> PRE["preprocessing/"]
     CORE --> HAB["habitat_analysis/"]
     CORE --> MLC["machine_learning/"]
     CORE --> DCM["dicom_sort/"]

     COM --> CFG["configs/"]
     COM --> CON["configurators/"]
     COM --> REG["registry.py"]
     COM --> ORC["orchestrator.py"]

Top-level package responsibilities
------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Package
     - Responsibility
   * - ``habit/cli.py``
     - Click root command group; subcommands are declared here and command
       bodies only perform lazy imports and forwarding.
   * - ``habit/commands/``
     - **Active command implementation layer**: each ``cmd_*.py`` loads
       configuration, calls the core, and reports results. Shared helpers are
       in ``common.py``.
   * - ``habit/core/common/``
     - Cross-domain infrastructure: YAML loading and path resolution
       (``configs/``), the Configurator base classes (``configurators/``), the
       shared registry base (``registry.py`` →
       :class:`~habit.core.common.registry.ClassRegistry`), and the
       orchestrator contract table (``orchestrator.py`` →
       :data:`~habit.core.common.orchestrator.ORCHESTRATOR_CONTRACT`).
   * - ``habit/core/schemas/``
     - Pydantic configuration models for workflows, step parameters, parameter
       registration, validation, and GUI reflection.
   * - ``habit/core/preprocessing/``
     - Batch image preprocessing: ``BatchProcessor``,
       ``BasePreprocessor``, ``PreprocessorFactory``, and step implementations.
   * - ``habit/core/habitat_analysis/``
     - Habitat segmentation, clustering features, post-segmentation feature
       extraction, and traditional radiomics (see :doc:`subsystems`).
   * - ``habit/core/machine_learning/``
     - Tabular machine learning: data assembly, feature selection, modeling,
       evaluation, reporting, visualization, and statistical tests.
   * - ``habit/core/dicom_sort/``
     - Standalone DICOM sorting based on dcm2niix; it does not use
       ``BatchProcessor``.
   * - ``habit/utils/``
     - Shared utilities used across subsystems (see below).

Shared utilities: ``habit/utils/``
-------------------------

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
     - Cancellation detection for long-running tasks and the GUI.

Cross-subsystem contract files
----------------

The following files define package-wide interface contracts. Update them and
run the contract tests when adding a factory or orchestrator:

.. mermaid::

   flowchart LR
     REG["common/registry.py<br/>ClassRegistry"] --> PF["PreprocessorFactory"]
     REG --> MF["ModelFactory"]
     REG --> CF["ClusteringAlgorithmFactory"]
     REG --> EF["FeatureExtractorRegistry"]
     REG --> PP["PreprocessingMethodFactory"]
     REG --> HF["HabitatFeatureFactory"]

     ORC["common/orchestrator.py<br/>ORCHESTRATOR_CONTRACT"] --> BP["BatchProcessor"]
     ORC --> HA["HabitatAnalysis"]
     ORC --> HW["HoldoutWorkflow / ..."]

     TST["tests/test_architecture_contracts.py"] -.-> REG
     TST -.-> ORC

Where to start when changing X
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Goal
     - Starting point
   * - Add or modify a CLI command
     - ``habit/cli.py`` + ``habit/commands/cmd_*.py``
   * - Add a preprocessing step
     - ``habit/core/preprocessing/`` + ``PreprocessorFactory``
   * - Add a clustering algorithm
     - ``habit/core/habitat_analysis/clustering/base_clustering.py``
   * - Add a clustering feature extractor
     - ``habit/core/habitat_analysis/clustering_features/base_extractor.py``
   * - Add a machine-learning model
     - ``habit/core/machine_learning/models/factory.py``
   * - Add a feature-selection method
     - ``habit/core/machine_learning/feature_selectors/selector_registry.py``
   * - Change configuration fields or validation rules
     - ``habit/core/schemas/workflows/`` and ``schemas/steps/``
   * - Change the three habitat pipeline strategies
     - ``habit/core/habitat_analysis/habitat_analysis.py``\ （``_PIPELINE_RECIPES``）+ ``pipelines/steps/``
   * - Change the ML training or prediction flow
     - ``habit/core/machine_learning/workflows/`` and ``runners/``
   * - Add a class-based factory
     - Subclass ``ClassRegistry`` from ``habit/core/common/registry.py``;
       follow an existing factory in the same domain.
   * - Add a top-level orchestrator for a new CLI pipeline
     - Implement the class and update ``ORCHESTRATOR_CONTRACT`` in
       ``common/orchestrator.py``. ``tests/test_architecture_contracts.py``
       reads this table automatically to validate terminal methods.

.. seealso::

   See :doc:`extension_points` for the complete extension and registration
   reference, and :doc:`../customization/index` for implementation templates.
