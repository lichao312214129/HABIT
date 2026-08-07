Subsystems
==========

This page describes the internals of the two most complex subsystems:
habitat analysis and machine learning. Preprocessing and DICOM sorting use
simpler structures; see :doc:`repo_layout`.

.. note::

   **v1.0 framing.** User-facing orchestration lives in the v1 recipes
   (:doc:`../api/index`):

   * Habitat studies: ``habit.recipes.two_step`` / ``one_step`` /
     ``direct_pooling`` over ``habit/domain/`` registries.
   * Tabular ML: ``habit.recipes.train_model`` / ``cross_validate`` /
     ``predict_model`` over :class:`~habit.domain.TablePipeline`, with
     figures via :mod:`habit.recipes.ml_reporting` + ``habit.viz``.
   * Feature extract / radiomics: ``extract_habitat_features`` /
     ``traditional_radiomics`` in :mod:`habit.recipes.features` (parallel
     L4 recipes; built-in extract is domain-native).
   * Model comparison: :func:`~habit.recipes.compare_models` in
     :mod:`habit.recipes.comparison` (domain evaluation +
     :mod:`habit.recipes.comparison_reporting` + ``habit.viz``).

   What follows are the internals of the **v0.1 engines** under
   ``habit/compat/engines/``, retained for YAML/CLI parity and residual
   fallbacks. Read this page when changing engine internals; read
   :doc:`architecture` for the v1 layering.

Habitat analysis
----------------

Habitat analysis aggregates voxel-level image features into spatially
coherent tumor subregions. In the v0.1 engine
(``habit/compat/engines/habitat_analysis/``), ``HabitatAnalysis`` organizes
the work as a pipeline of steps and services. The directories below are
relative to that package.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Directory
     - Responsibility
   * - ``clustering/``
     - K-means, GMM, SLIC, and cluster-count validation.
   * - ``clustering_features/``
     - Voxel and supervoxel feature extractors.
   * - ``feature_preprocessing/``
     - DataFrame normalization and feature filtering.
   * - ``habitat_features/``
     - Post-segmentation radiomics, MSI, ITH, and related features.
   * - ``pipelines/``
     - Pipeline definitions and step implementations.
   * - ``services/``
     - Feature, clustering, habitat-map, and result services.
   * - ``checkpoint/``
     - Subject-level training checkpoints.

Clustering strategies
~~~~~~~~~~~~~~~~~~~~~

The ``habitat_segmentation.clustering_mode`` setting selects one of three
pipeline recipes:

* **``two_step``** first clusters voxels into subject-level supervoxels and
  then clusters pooled supervoxels into habitats.
* **``one_step``** clusters each subject's voxels directly into habitats.
* **``direct_pooling``** pools voxels across subjects and performs one
  clustering operation.

Training and prediction
~~~~~~~~~~~~~~~~~~~~~~~

* **Train** builds and fits a ``HabitatPipeline`` and serializes it to
  ``habitat_pipeline.pkl``.
* **Predict** loads the pipeline, injects approved services, and transforms
  new subjects with the same fitted state.
* **Resume** uses subject-level checkpoints for interrupted large cohorts.

After habitat maps are generated, ``habit extract`` runs
:func:`~habit.recipes.extract_habitat_features` (domain
``HabitatFeatureExtractor`` components for built-in types; compat
``HabitatMapAnalyzer`` only when YAML requests an unregistered plugin).
``habit radiomics`` runs :func:`~habit.recipes.traditional_radiomics`.

Machine learning
----------------

**v1 path (CLI / recipes).** ``habit model`` and ``habit cv`` call
:func:`~habit.recipes.train_model` / :func:`~habit.recipes.cross_validate`.
Reporting writes ``metrics.json``, prediction tables, and — when
visualization is enabled — figures under ``<output>/visualizations/`` via
:mod:`habit.recipes.ml_reporting` and ``habit.viz.classification``
(prefixes ``train_`` / ``test_`` / ``cv_``). ``habit compare`` is a separate
recipe (:func:`~habit.recipes.compare_models`); it merges prediction CSVs,
computes metrics / DeLong in
:mod:`habit.domain.evaluation.comparison`, and writes multi-model curves
through :mod:`habit.recipes.comparison_reporting`.

**v0.1 engine (compat).** The modules below live under
``habit/compat/engines/machine_learning/``. They remain available for
legacy configuration-object callers and opaque ``*_final_pipeline.pkl``
predict loads. New work should target the v1 recipes and ``habit.viz``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - ``data_manager.py``
     - Load tables, merge data, and split train/test sets.
   * - ``pipeline_builder.py``
     - Build selector, scaler, resampler, and model pipelines.
   * - ``models/``
     - Base models, factory registration, and model implementations.
   * - ``feature_selectors/``
     - Registered selectors and ICC/retest analysis.
   * - ``workflows/``
     - High-level holdout and K-fold workflows (legacy). Multi-model
       comparison for ``habit compare`` is v1
       (:mod:`habit.recipes.comparison`), not this tree.
   * - ``runners/``
     - Concrete training and inference execution.
   * - ``contracts/``
     - Immutable workflow plans and structured results.
   * - ``evaluation/``
     - Metrics, thresholds, prediction containers, and evaluation.
   * - ``reporting/`` and ``visualization/``
     - Reports and English-labeled ROC, calibration, DCA, and KM plots.
   * - ``statistics/``
     - DeLong, Hosmer-Lemeshow, and Spiegelhalter-Z tests.
   * - ``resampling.py``
     - Over-sampling, under-sampling, and SMOTE.

Execution flow
~~~~~~~~~~~~~~

.. mermaid::

   flowchart TD
     C["Validated MLConfig"] --> W["Workflow.run()"]
     W --> P["WorkflowPlan"]
     P --> R["Runner"]
     R --> D["DataManager"]
     D --> B["PipelineBuilder"]
     B --> F["Fit and evaluate"]
     F --> O["Structured result and report"]

The single sklearn Pipeline is fitted only on training data in each fold,
which prevents leakage and keeps training and evaluation behavior consistent.
