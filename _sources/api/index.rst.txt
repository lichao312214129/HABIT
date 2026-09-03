API Reference
=============

HABIT v2.0 is **API-first**: the Python API is the product, and the CLI and
YAML configuration files are thin shells over the objects documented here.
The layered core is:

``kernels`` → ``contracts`` → capability packages → ``spec`` / ``execution`` /
``adapters`` → ``recipes`` / ``report``

Every component listed on this page is imported from its canonical capability
package. The package root exports version metadata only.
The declarative registry of the stable surface lives in
``habit/_public_api.py``.

**Stability.** Symbols listed here are the canonical v2.0 public surface.
Anything not listed here is internal and may change without notice.

.. rubric:: Contents

* :ref:`api-recipes`
* :ref:`api-report`
* :ref:`api-contracts`
* :ref:`api-spec`
* :ref:`api-domain`
* :ref:`api-execution`
* :ref:`api-adapters`
* :ref:`api-image-io`
* :ref:`api-datasets`
* :ref:`api-registry`
* :ref:`api-kernels`
* :ref:`api-compat`
* :ref:`api-viz`
* :ref:`api-exceptions`
* :ref:`api-guides`

.. _api-recipes:

Study recipes (``habit.recipes``)
---------------------------------

Recipes are the named study designs — the highest level of the library. Each
takes a :class:`~habit.contracts.Cohort` (or a
:class:`~habit.contracts.FeatureTable` for machine learning) plus a spec, and
returns a typed result object. See :doc:`python_api` for a narrative
walkthrough.

Habitat analysis
~~~~~~~~~~~~~~~~

Primary entry: :class:`~habit.recipes.Study` (sklearn-style
:meth:`~habit.recipes.Study.fit` / :meth:`~habit.recipes.Study.fit_predict` /
:meth:`~habit.recipes.Study.predict`). Factories
``two_step_habitat`` / ``one_step_habitat`` / ``direct_pooling_habitat``
build a :class:`~habit.recipes.Study` with a declared design.

.. autosummary::
   :toctree: generated

   habit.recipes.Study
   habit.recipes.two_step_habitat
   habit.recipes.one_step_habitat
   habit.recipes.direct_pooling_habitat

Precision screen
~~~~~~~~~~~~~~~~

Which already-extracted voxel features are repeatable and reproducible
enough to define habitats (Prior et al., Radiol Artif Intell
2024;6(2):e230118). Maps come from
:func:`~habit.voxel_features.extract_voxel_texture` /
:class:`~habit.voxel_features.VoxelRadiomicsFeatures`. See
:doc:`../examples/precise_features`.

.. autosummary::
   :toctree: generated

   habit.precision.perturb_image
   habit.precision.precision_panel
   habit.precision.identify_precise_features
   habit.recipes.identify_precise_voxel_features
   habit.recipes.voxel_radiomics_factory

Feature extraction (config-driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.recipes.extract_habitat_features
   habit.recipes.traditional_radiomics

Study utilities
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.recipes.run_from_yaml
   habit.recipes.preprocess_images
   habit.recipes.preprocess_subject
   habit.recipes.preprocess_image
   habit.recipes.icc_analysis
   habit.recipes.sort_dicom
   habit.recipes.dice
   habit.recipes.dicom_info
   habit.recipes.merge_tables

Recipe result types
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.recipes.StudyResult
   habit.recipes.ModelResult
   habit.recipes.CVResult
   habit.recipes.PredictionResult

.. _api-report:

Report (``habit.report``)
-------------------------

Run-scoped persistence and per-subject figures. Pass a
:class:`~habit.report.Report` to :meth:`~habit.recipes.Study.fit_predict`.
This is **not** part of :class:`~habit.spec.HabitatSpec` and does not
enter scientific fingerprints. ``figure_layout`` is ``"flat"`` or
``"by_subject"`` (:data:`~habit.report.FIGURE_LAYOUTS`). Figure-atom
catalog: :doc:`../examples/visualization`. See
:doc:`../examples/one_step_habitat`.

.. autosummary::
   :toctree: generated

   habit.report.Report
   habit.report.FigureAtom
   habit.report.SubjectContext
   habit.report.Overlay
   habit.report.VolumeFractions
   habit.report.MSI
   habit.report.ITH
   habit.report.ClusterValidation
   habit.report.GraphSlice
   habit.report.GraphNetwork2D

.. _api-contracts:

Data contracts (``habit.contracts``)
------------------------------------

The in-memory data model. Contracts are plain value objects with no IO and no
YAML knowledge; adapters and recipes move them to and from disk. See
:doc:`data_model` for the narrative guide.

Imaging
~~~~~~~

.. autosummary::
   :toctree: generated

   habit.contracts.Geometry
   habit.contracts.ImageRef
   habit.contracts.ArrayImageRef
   habit.contracts.ImageVolume
   habit.contracts.MaskVolume
   habit.contracts.Subject
   habit.contracts.Cohort
   habit.contracts.CohortFingerprint
   habit.contracts.cohort_from_directory

Provenance
~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.contracts.Provenance
   habit.contracts.RunManifest
   habit.contracts.software_fingerprint

Habitat artefacts
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.contracts.VoxelFeatureField
   habit.contracts.Supervoxelization
   habit.contracts.HabitatMap
   habit.contracts.HabitatModel

Tables and outcomes
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.contracts.FeatureTable
   habit.contracts.Outcome
   habit.contracts.BinaryOutcome
   habit.contracts.MulticlassOutcome
   habit.contracts.ContinuousOutcome
   habit.contracts.SurvivalOutcome
   habit.contracts.outcome_from_dict
   habit.contracts.outcome_to_dict

Operator protocols
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.contracts.SubjectOperator
   habit.contracts.CohortOperator
   habit.contracts.SubjectResult
   habit.contracts.ExecutionBackend
   habit.contracts.DataSource
   habit.contracts.ResultWriter

.. _api-spec:

Specifications (``habit.spec``)
-------------------------------

Specs are frozen, fingerprintable value objects describing *what* to run.
They are the in-Python counterpart of the YAML configuration files. See
:doc:`spec` for the narrative guide and YAML migration walkthrough.

.. autosummary::
   :toctree: generated

   habit.spec.Spec
   habit.spec.Stage
   habit.spec.HabitatSpec
   habit.spec.MLSpec
   habit.spec.RunPolicy

Serialisation
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.spec.load_habitat_spec
   habit.spec.save_habitat_spec
   habit.spec.load_run_policy
   habit.spec.save_run_policy
   habit.spec.build_habitat_document
   habit.spec.save_habitat_config
   habit.spec.load_habitat_config

YAML migration
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.spec.LegacyConfigAdapter
   habit.spec.LegacyTranslation
   habit.spec.MigrationReport
   habit.spec.detect_yaml_version
   habit.spec.migrate_yaml
   habit.spec.validate_v1_document

.. _api-domain:

Capability packages
-------------------

The pluggable operator surface. Every component is constructed either
directly (``SlicSupervoxelizer(n_supervoxels=50)``) or by name through its
registry (``SupervoxelizerRegistry.create("slic", ...)``). Importing
the v2 capability packages registers the built-ins listed here. See :doc:`domain` for
the narrative overview, :doc:`domain_habitat` and :doc:`domain_table` for the
two pipeline families, and :doc:`plugins` for runtime introspection.

Pipelines
~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.pipeline.SubjectPipeline
   habit.pipeline.TablePipeline

Habitat protocols
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.voxel_features.VoxelFeatureExtractor
   habit.supervoxel.Supervoxelizer
   habit.supervoxel.SupervoxelFeatureExtractor
   habit.habitat_model.HabitatModelFitter
   habit.habitat_model.HabitatAssigner
   habit.habitat_features.HabitatFeatureExtractor
   habit.combiners.Combiner
   habit._protocols.Seedable

Voxel feature extractors
~~~~~~~~~~~~~~~~~~~~~~~~

Registered domain ``voxel_feature_extractor``. ``raw`` / ``local_entropy``
/ ``voxel_radiomics`` describe each ROI voxel; ``concat`` / ``expression``
/ ``kinetic`` compose those families. :func:`~habit.voxel_features.extract_voxel_texture`
is the same ``voxel_radiomics`` pass on one ``ImageVolume`` + mask (no
``Subject``). Precise screening uses these maps; it does not extract them.

.. autosummary::
   :toctree: generated

   habit.voxel_features.extract_voxel_texture
   habit.voxel_features.RawVoxelFeatures
   habit.voxel_features.LocalEntropyVoxelFeatures
   habit.voxel_features.VoxelRadiomicsFeatures
   habit.voxel_features.ConcatVoxelFeatures
   habit.voxel_features.ExpressionVoxelFeatures
   habit.voxel_features.KineticVoxelFeatures
   habit.voxel_features.VoxelFeatureExtractorRegistry

Feature trees and combiners
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compose extractors into one node (``concat(raw("T1"), voxel_radiomics("T2"))``).

.. autosummary::
   :toctree: generated

   habit.voxel_features.VoxelFeatureTree
   habit.voxel_features.build_voxel_extractor
   habit.supervoxel.SupervoxelFeatureTree
   habit.supervoxel.build_supervoxel_extractor
   habit.habitat_features.HabitatFeatureTree
   habit.habitat_features.build_habitat_extractor
   habit.combiners.CombinerRegistry
   habit.combiners.ConcatCombiner
   habit.combiners.WeightedConcatCombiner
   habit.combiners.AverageCombiner
   habit.combiners.RatioCombiner
   habit.combiners.DifferenceCombiner
   habit.combiners.ExpressionCombiner
   habit.combiners.KineticCombiner

Supervoxelizers
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.supervoxel.SlicSupervoxelizer
   habit.supervoxel.KMeansSupervoxelizer
   habit.supervoxel.GmmSupervoxelizer

Supervoxel feature extractors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.supervoxel.MeanVoxelFeatures
   habit.supervoxel.MeanSupervoxelFeatures
   habit.supervoxel.StdSupervoxelFeatures
   habit.supervoxel.PercentileSupervoxelFeatures
   habit.supervoxel.SupervoxelRadiomicsFeatures
   habit.supervoxel.SupervoxelFeatureExtractorRegistry

Habitat model fitters
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.habitat_model.KMeansHabitatModelFitter
   habit.habitat_model.GmmHabitatModelFitter

Habitat assigners
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.habitat_model.NearestCentroidAssigner

Habitat feature extractors
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.habitat_features.HabitatVolumeFeatures
   habit.habitat_features.IthHabitatFeatures
   habit.habitat_features.MsiHabitatFeatures
   habit.habitat_features.GraphHabitatFeatures
   habit.habitat_features.NonRadiomicsHabitatFeatures
   habit.habitat_features.EachHabitatRadiomicsFeatures
   habit.habitat_features.to_habitat_feature_panel
   habit.habitat_features.compare_habitat_features
   habit.habitat_features.HabitatFeaturePanel
   habit.habitat_features.HabitatFeatureComparison
   habit.habitat_features.WholeHabitatRadiomicsFeatures
   habit.habitat_features.TraditionalRadiomicsHabitatFeatures

Voxel-feature preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subject-level preprocessing of voxel feature matrices, composable into
chains.

.. autosummary::
   :toctree: generated

   habit.feature_preprocessing.SubjectFeaturePreprocessor
   habit.feature_preprocessing.CohortFeaturePreprocessor
   habit.feature_preprocessing.SubjectPreprocessingChain
   habit.feature_preprocessing.CohortPreprocessingChain
   habit.feature_preprocessing.build_methods
   habit.feature_preprocessing.ZScoreScaling
   habit.feature_preprocessing.MinMaxScaling
   habit.feature_preprocessing.RobustScaling
   habit.feature_preprocessing.Winsorizing
   habit.feature_preprocessing.LogTransform
   habit.feature_preprocessing.Binning
   habit.feature_preprocessing.Impute
   habit.feature_preprocessing.VarianceFilter
   habit.feature_preprocessing.CorrelationFilter
   habit.feature_preprocessing.PreciseCorrelationFilter
   habit.feature_preprocessing.MaxAbsScaling
   habit.feature_preprocessing.QuantileTransform
   habit.feature_preprocessing.L2Normalizer
   habit.feature_preprocessing.FeatureWhitelist

Precision screen
~~~~~~~~~~~~~~~~

Simulated-retest perturbations and the ICC intersection that decides
which extracted voxel columns may define habitats. See
:doc:`../examples/precise_features`.

.. autosummary::
   :toctree: generated

   habit.precision.ImagePerturbation
   habit.precision.GaussianNoisePerturbation
   habit.precision.TranslationPerturbation
   habit.precision.RotationPerturbation
   habit.precision.RigidPerturbation
   habit.precision.BSplineDeformPerturbation
   habit.precision.MorphologicalPerturbation
   habit.precision.GradientWeightedPerturbation
   habit.precision.SliceExtentPerturbation
   habit.precision.perturb_image
   habit.precision.PerturbationChain
   habit.precision.prior2024_retest_perturbation
   habit.precision.PreciseFeatureSet
   habit.precision.precision_panel
   habit.precision.aggregate_panels
   habit.precision.identify_precise_features

Habitat label matching
~~~~~~~~~~~~~~~~~~~~~~

Match independently clustered habitat ids (e.g. observers vs patients).
See :doc:`../examples/habitat_label_match`.

.. autosummary::
   :toctree: generated

   habit.precision.align_habitat_map
   habit.precision.habitat_stability

Table protocols
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.table_preprocessing.TablePreprocessor
   habit.feature_selection.FeatureSelector
   habit.classification.Classifier
   habit.evaluation.Metric

Table preprocessors
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.table_preprocessing.ZScorePreprocessor
   habit.table_preprocessing.MinMaxPreprocessor
   habit.table_preprocessing.RobustPreprocessor
   habit.table_preprocessing.WinsorizePreprocessor
   habit.table_preprocessing.LogPreprocessor
   habit.table_preprocessing.BinningPreprocessor
   habit.table_preprocessing.VarianceFilterPreprocessor
   habit.table_preprocessing.CorrelationFilterPreprocessor

Feature selectors
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.feature_selection.VarianceSelector
   habit.feature_selection.CorrelationSelector
   habit.feature_selection.AnovaSelector
   habit.feature_selection.Chi2Selector
   habit.feature_selection.IccSelector
   habit.feature_selection.LassoSelector
   habit.feature_selection.MrmrSelector
   habit.feature_selection.RfecvSelector
   habit.feature_selection.StatisticalTestSelector
   habit.feature_selection.StepwiseSelector
   habit.feature_selection.UnivariateLogisticSelector
   habit.feature_selection.VifSelector

Classifiers
~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.classification.LogisticRegressionClassifier
   habit.classification.SvmClassifier
   habit.classification.SvcClassifier
   habit.classification.RandomForestClassifier
   habit.classification.GradientBoostingClassifier
   habit.classification.AdaboostClassifier
   habit.classification.DecisionTreeClassifier
   habit.classification.KnnClassifier
   habit.classification.MlpClassifier
   habit.classification.GaussianNbClassifier
   habit.classification.BernoulliNbClassifier
   habit.classification.MultinomialNbClassifier
   habit.classification.XgboostClassifier
   habit.classification.AutogluonTabularClassifier

Metrics
~~~~~~~

.. autosummary::
   :toctree: generated

   habit.evaluation.AccuracyMetric
   habit.evaluation.AucMetric
   habit.evaluation.F1ScoreMetric
   habit.evaluation.SensitivityMetric
   habit.evaluation.SpecificityMetric
   habit.evaluation.PpvMetric
   habit.evaluation.NpvMetric
   habit.evaluation.HosmerLemeshowPValueMetric
   habit.evaluation.SpiegelhalterZPValueMetric

Evaluation statistics
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.evaluation.auc_confidence_interval
   habit.evaluation.calibration_tests
   habit.evaluation.delong_test
   habit.evaluation.icc_analysis
   habit.evaluation.repeat_measurement_matrix
   habit.evaluation.AucConfidenceInterval
   habit.evaluation.CalibrationResult
   habit.evaluation.DelongResult

Component registries
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.voxel_features.VoxelFeatureExtractorRegistry
   habit.supervoxel.SupervoxelizerRegistry
   habit.supervoxel.SupervoxelFeatureExtractorRegistry
   habit.habitat_model.HabitatModelFitterRegistry
   habit.habitat_model.HabitatAssignerRegistry
   habit.habitat_features.HabitatFeatureExtractorRegistry
   habit.feature_preprocessing.FeaturePreprocessingMethodRegistry
   habit.precision.ImagePerturbationRegistry
   habit.table_preprocessing.TablePreprocessorRegistry
   habit.feature_selection.FeatureSelectorRegistry
   habit.classification.ClassifierRegistry
   habit.evaluation.MetricRegistry

.. _api-execution:

Execution (``habit.execution``)
-------------------------------

Execution backends control how subject-level operators are mapped over a
cohort: serially, in a process pool, with per-subject timeouts and
checkpoint/resume. Integrator chapter: :doc:`../tutorial/execution`.
Reference: :doc:`execution`.

.. autosummary::
   :toctree: generated

   habit.execution.SerialBackend
   habit.execution.ProcessPoolBackend
   habit.execution.SubjectTimeoutError
   habit.execution.CheckpointStore

.. _api-adapters:

Adapters (``habit.adapters``)
-----------------------------

Adapters connect the contracts to the filesystem: directory conventions for
cohorts, lazy file-backed image references, and result writers. See
:doc:`adapters` and :doc:`image_io`.

.. autosummary::
   :toctree: generated

   habit.adapters.DirectoryDataSource
   habit.adapters.DirectoryResultWriter
   habit.adapters.FileImageRef

.. _api-image-io:

Image I/O and low-level radiomics
---------------------------------

SimpleITK-backed read / geometry checks, plus pair-wise radiomics helpers.
Prefer contract volumes inside pipelines (:doc:`data_model`); see
:doc:`image_io`.

.. autosummary::
   :toctree: generated

   habit.api.image.GeometryPolicy
   habit.api.image.GeometryReport
   habit.api.image.ImageVolume
   habit.api.image.MaskVolume
   habit.api.image.ImageMaskPair
   habit.api.image.read_image
   habit.api.image.read_mask
   habit.api.image.validate_geometry
   habit.api.image.align_image_mask
   habit.api.radiomics.extract_features
   habit.api.radiomics.extract_batch
   habit.api.radiomics.FeatureResult
   habit.api.radiomics.FeatureTableResult

.. _api-datasets:

Datasets (``habit.datasets``)
-----------------------------

Official imaging pack (:func:`~habit.datasets.fetch_demo`, downloaded once) plus
deterministic synthetic builders for tests and in-memory API exploration.

.. autosummary::
   :toctree: generated

   habit.datasets.fetch_demo
   habit.datasets.get_data_home
   habit.datasets.inspect_preprocessed_root
   habit.datasets.PreprocessedInventory
   habit.datasets.make_synthetic_cohort
   habit.datasets.make_synthetic_feature_table

.. _api-registry:

Component registry (``habit.registry``)
---------------------------------------

The generic name → component mapping underlying every domain registry. See
:doc:`registry`.

.. autosummary::
   :toctree: generated

   habit.registry.ComponentRegistry
   habit.api.plugins.list_plugins
   habit.api.plugins.get_plugin_info
   habit.api.plugins.plugin_catalog
   habit.api.plugins.load_plugins
   habit.api.plugins.PluginInfo
   habit.api.plugins.PluginCatalogEntry
   habit.api.plugins.PluginLoadReport
   habit.api.utils.setup_logger
   habit.api.utils.is_available
   habit.api.utils.show_versions
   habit.api.utils.check_component

.. _api-kernels:

Numeric kernels (``habit.kernels``)
-----------------------------------

Pure NumPy/SciPy functions — no ``Subject``, no YAML, no IO. Call them the
way you would call a SciPy function. See :doc:`kernels`.

Model selection
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.kernels.score_direction
   habit.kernels.knee_index
   habit.kernels.best_index
   habit.kernels.vote_best_index
   habit.kernels.gap_statistic

The four selection-rule constants are documented at their canonical defining
module: module-level data does not carry a runtime docstring, so autodoc can
only pick up their documentation there.

.. autosummary::
   :toctree: generated

   habit.kernels.cluster_selection.SCORE_DIRECTIONS
   habit.kernels.cluster_selection.MAXIMIZE
   habit.kernels.cluster_selection.MINIMIZE
   habit.kernels.cluster_selection.KNEE

Habitat metrics
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.kernels.local_entropy_map
   habit.kernels.spatial_interaction_matrix
   habit.kernels.msi_features_from_matrix
   habit.kernels.habitat_volume_fractions
   habit.kernels.habitat_region_stats
   habit.kernels.habitat_ith_dispersion
   habit.kernels.ith_score
   habit.kernels.HabitatGraphFeatureOptions
   habit.kernels.extract_graph_features
   habit.kernels.extract_graph_features_for_labels
   habit.kernels.extract_habitat_nodes
   habit.kernels.build_centroid_distance_graph
   habit.kernels.build_min_distance_graph
   habit.kernels.build_adjacency_graph
   habit.kernels.pair_count

Classification and agreement statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.kernels.compute_midrank
   habit.kernels.fast_delong
   habit.kernels.delong_roc_variance
   habit.kernels.delong_roc_test
   habit.kernels.delong_roc_ci
   habit.kernels.hosmer_lemeshow_test
   habit.kernels.spiegelhalter_z_test
   habit.kernels.two_way_mean_squares
   habit.kernels.icc3_1
   habit.kernels.icc2_1

Image perturbation and voxel reliability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.kernels.estimate_noise_sigma
   habit.kernels.add_gaussian_noise
   habit.kernels.translate_image
   habit.kernels.rotate_image
   habit.kernels.rigid_transform_image
   habit.kernels.morphological_grow_shrink
   habit.kernels.boundary_band_mask
   habit.kernels.boundary_weighted_perturbation
   habit.kernels.slice_extent_perturbation
   habit.kernels.ICCEstimate
   habit.kernels.icc3a_1
   habit.kernels.icc3c_1

.. _api-compat:

Legacy compat (``habit.compat``)
--------------------------------

v2 removed external ecosystem adapters (scikit-learn / MONAI / nnU-Net
wrappers). Remaining submodules are frozen v0.1 helpers. See :doc:`compat`.

.. _api-viz:

Visualization (``habit.viz``)
-----------------------------

Publication figures. ``matplotlib`` is imported lazily inside each function,
so importing ``habit`` never pulls a plotting backend. All figure labels are
English-only.

Interactive habitat overlay (optional ``[view]`` extra; default
``habit view``): :func:`~habit.viz.view_habitat_napari`. Without napari the
CLI falls back to a PNG. Static PNG overlays:
:func:`~habit.viz.plot_habitat_overlay` (``habit view --backend matplotlib``;
panel aspect follows voxel spacing; ``display_convention`` defaults to
radiological — pass an ``ImageVolume`` so coronal/sagittal keep superior up).

Styles
~~~~~~

.. autosummary::
   :toctree: generated

   habit.viz.StyleSpec
   habit.viz.use_style
   habit.viz.get_style
   habit.viz.register_style
   habit.viz.available_styles
   habit.viz.sanitize_label

Survival
~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.viz.plot_kaplan_meier
   habit.viz.plot_risk_triptych
   habit.viz.plot_time_dependent_auc
   habit.viz.plot_survival_calibration
   habit.viz.plot_brier_curve
   habit.viz.plot_cox_forest

Regression and classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.viz.plot_predicted_vs_observed
   habit.viz.plot_residuals
   habit.viz.plot_residual_qq
   habit.viz.plot_bland_altman
   habit.viz.plot_coefficient_forest
   habit.viz.plot_roc
   habit.viz.plot_precision_recall
   habit.viz.plot_calibration
   habit.viz.plot_decision_curve
   habit.viz.plot_confusion_matrix
   habit.viz.plot_shap_summary
   habit.viz.plot_shap_bar
   habit.viz.plot_shap_violin
   habit.viz.plot_shap_heatmap
   habit.viz.plot_shap_dependence
   habit.viz.plot_shap_waterfall
   habit.viz.plot_shap_decision
   habit.viz.plot_shap_force
   habit.viz.plot_permutation_importance

Habitat clustering and overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   habit.viz.plot_habitat_clustering_pca_2d
   habit.viz.plot_habitat_clustering_pca_3d
   habit.viz.plot_habitat_clustering_pca_3d_interactive
   habit.viz.plot_habitat_overlay
   habit.viz.plot_cluster_validation_curves
   habit.viz.plot_cluster_validation_from_report
   habit.viz.plot_habitat_volume_fractions
   habit.viz.plot_msi_matrix
   habit.viz.plot_ith_summary
   habit.viz.plot_habitat_label_compare
   habit.viz.plot_partition_triptych
   habit.viz.plot_precision_icc
   habit.viz.plot_habitat_feature_heatmap
   habit.viz.plot_habitat_feature_effect
   habit.viz.plot_habitat_feature_components
   habit.viz.plot_habitat_feature_violin
   habit.viz.plot_habitat_feature_bars
   habit.viz.plot_habitat_graph_slice
   habit.viz.plot_habitat_graph_network_2d
   habit.viz.plot_graph_feature_heatmap
   habit.viz.render_habitat_graph_surface_3d
   habit.viz.render_habitat_graph_network_3d
   habit.viz.dense_voxel_feature_map
   habit.viz.plot_intensity_slice
   habit.viz.plot_voxel_texture_slice
   habit.viz.view_habitat_napari

.. _api-exceptions:

Exceptions (``habit.exceptions``)
---------------------------------

The canonical exception hierarchy. See :doc:`exceptions`.

.. autosummary::
   :toctree: generated

   habit.exceptions.HabitError
   habit.exceptions.HABITAPIError
   habit.exceptions.ConfigurationError
   habit.exceptions.DataFormatError
   habit.exceptions.GeometryError
   habit.exceptions.OptionalDependencyError
   habit.exceptions.ComponentNotFoundError
   habit.exceptions.CompatibilityError
   habit.exceptions.ProcessingError
   habit.exceptions.NotFittedError

.. _api-guides:

API guides
----------

Narrative companions to the reference tables above:

.. toctree::
   :maxdepth: 1

   python_api
   data_model
   adapters
   domain
   domain_habitat
   domain_table
   spec
   execution
   kernels
   compat
   plugins
   registry
   image_io
   exceptions
