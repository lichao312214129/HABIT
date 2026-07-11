Subsystems
==========

This page describes the two most complex subsystems: habitat analysis and
machine learning. Preprocessing and DICOM sorting use simpler structures; see
:doc:`repo_layout`.

Habitat analysis
----------------

Habitat analysis aggregates voxel-level image features into spatially
coherent tumor subregions. ``HabitatAnalysis`` organizes the work as a
pipeline of steps and services.

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

After habitat maps are generated, ``habit extract`` uses
``HabitatMapAnalyzer`` for downstream features and ``habit radiomics`` uses
``TraditionalRadiomicsExtractor``.

Machine learning
----------------

The machine-learning subsystem assembles feature tables, splits data, builds an
sklearn pipeline, trains and evaluates models, and writes reports.

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
     - High-level holdout, K-fold, and model-comparison workflows.
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
