# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Declarative v2 capability namespace registry.

The package root deliberately exposes no component aliases.  A public
component has exactly one canonical implementation namespace, recorded here
without importing that namespace so ``import habit`` remains lightweight.
"""

from __future__ import annotations

from typing import Dict, Tuple

PUBLIC_API_SYMBOLS: Tuple[str, ...] = ()

PUBLIC_NAMESPACES: Dict[str, Tuple[str, ...]] = {'habit.voxel_features': ('aligned_image', 'build_voxel_field', 'roi_voxels', 'ConcatVoxelFeatures', 'extract_voxel_texture', 'ExpressionVoxelFeatures', 'KineticVoxelFeatures', 'LocalEntropyVoxelFeatures', 'RawVoxelFeatures', 'load_cached_voxel_field', 'save_cached_voxel_field', 'voxel_radiomics_cache_key', 'voxel_volume_fingerprint', 'VoxelRadiomicsFeatures', 'VoxelFeatureExtractorRegistry', 'VoxelFeatureExtractor', 'VoxelFeatureTree', 'build_voxel_extractor'), 'habit.supervoxel': ('SlicSupervoxelizer', 'KMeansSupervoxelizer', 'GmmSupervoxelizer', 'SupervoxelizerRegistry', 'MeanSupervoxelFeatures', 'MeanVoxelFeatures', 'PercentileSupervoxelFeatures', 'StdSupervoxelFeatures', 'SupervoxelRadiomicsFeatures', 'SupervoxelFeatureExtractorRegistry', 'aggregate_voxel_means', 'Supervoxelizer', 'SupervoxelFeatureExtractor', 'SupervoxelFeatureTree', 'build_supervoxel_extractor'), 'habit.feature_preprocessing': ('CohortFeaturePreprocessor', 'SubjectFeaturePreprocessor', 'Binning', 'CohortPreprocessingChain', 'CorrelationFilter', 'PreciseCorrelationFilter', 'FeaturePreprocessingMethodRegistry', 'FeatureWhitelist', 'Impute', 'L2Normalizer', 'LogTransform', 'MaxAbsScaling', 'MinMaxScaling', 'QuantileTransform', 'RobustScaling', 'SubjectPreprocessingChain', 'VarianceFilter', 'Winsorizing', 'ZScoreScaling', 'build_methods'), 'habit.habitat_model': ('HabitatAssigner', 'HabitatModelFitter', 'GmmHabitatModelFitter', 'KMeansHabitatModelFitter', 'HabitatModelFitterRegistry', 'NearestCentroidAssigner', 'HabitatAssignerRegistry', 'ConnectedComponentPostprocess', 'build_connected_component_postprocess'), 'habit.habitat_features': ('GraphHabitatFeatures', 'GraphHabitatFeaturesParams', 'IthHabitatFeatures', 'IthHabitatFeaturesParams', 'MsiHabitatFeatures', 'MsiHabitatFeaturesParams', 'NonRadiomicsHabitatFeatures', 'NonRadiomicsHabitatFeaturesParams', 'TraditionalRadiomicsHabitatFeatures', 'TraditionalRadiomicsHabitatFeaturesParams', 'WholeHabitatRadiomicsFeatures', 'WholeHabitatRadiomicsFeaturesParams', 'EachHabitatRadiomicsFeatures', 'EachHabitatRadiomicsFeaturesParams', 'HabitatFeaturePanel', 'HabitatFeatureComparison', 'to_habitat_feature_panel', 'compare_habitat_features', 'HabitatVolumeFeatures', 'HabitatVolumeFeaturesParams', 'HabitatFeatureExtractorRegistry', 'HabitatFeatureExtractor', 'HabitatFeatureTree', 'build_habitat_extractor'), 'habit.combiners': ('Combiner', 'block_sources', 'check_blocks', 'concat_blocks', 'AverageCombiner', 'AverageCombinerParams', 'ConcatCombiner', 'ConcatCombinerParams', 'DifferenceCombiner', 'DifferenceCombinerParams', 'ExpressionCombiner', 'ExpressionCombinerParams', 'KineticCombiner', 'KineticCombinerParams', 'RatioCombiner', 'RatioCombinerParams', 'WeightedConcatCombiner', 'WeightedConcatCombinerParams', 'CombinerRegistry'), 'habit.pipeline': ('SubjectPipeline', 'TablePipeline', 'voxel_units', 'PooledUnits', 'fan_in', 'PoolMarker', 'PoolMarkerParams', 'PoolingRegistry'), 'habit.table_preprocessing': ('TablePreprocessor', 'TablePreprocessorRegistry', 'MinMaxPreprocessor', 'ZScorePreprocessor', 'RobustPreprocessor', 'MaxAbsPreprocessor', 'QuantilePreprocessor', 'L2Preprocessor', 'BinningPreprocessor', 'WinsorizePreprocessor', 'LogPreprocessor', 'VarianceFilterPreprocessor', 'CorrelationFilterPreprocessor', 'PreciseCorrelationFilterPreprocessor'), 'habit.feature_selection': ('FeatureSelector', 'FeatureSelectorRegistry', 'VarianceSelector', 'CorrelationSelector', 'VifSelector', 'AnovaSelector', 'Chi2Selector', 'StatisticalTestSelector', 'UnivariateLogisticSelector', 'UnivariateCoxSelector', 'StepwiseSelector', 'RfecvSelector', 'LassoSelector', 'IccSelector', 'MrmrSelector'), 'habit.classification': ('Classifier', 'ClassifierRegistry', 'DecisionTreeClassifier', 'KnnClassifier', 'SvmClassifier', 'SvcClassifier', 'MlpClassifier', 'LogisticRegressionClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'XgboostClassifier', 'AdaboostClassifier', 'GaussianNbClassifier', 'MultinomialNbClassifier', 'BernoulliNbClassifier', 'AutogluonTabularClassifier'), 'habit.regression': ('Regressor', 'RegressorRegistry', 'RidgeRegressor', 'LassoRegressor', 'ElasticNetRegressor', 'SvrRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor'), 'habit.survival': ('SurvivalModel', 'SurvivalModelRegistry', 'CoxPhSurvival', 'RandomSurvivalForest', 'GradientBoostingSurvival'), 'habit.evaluation': ('Metric', 'MetricRegistry', 'SurvivalMetricRegistry', 'RegressionMetricRegistry', 'CIndexMetric', 'IntegratedBrierScoreMetric', 'CumulativeDynamicAucMetric', 'R2Metric', 'MaeMetric', 'MseMetric', 'RmseMetric', 'AccuracyMetric', 'SensitivityMetric', 'SpecificityMetric', 'PpvMetric', 'NpvMetric', 'F1ScoreMetric', 'AucMetric', 'HosmerLemeshowPValueMetric', 'SpiegelhalterZPValueMetric', 'DelongResult', 'delong_test', 'AucConfidenceInterval', 'auc_confidence_interval', 'CalibrationResult', 'calibration_tests', 'repeat_measurement_matrix', 'icc_analysis', 'CleanedPredictions', 'clean_binary_predictions', 'compute_classification_metrics', 'MergedPredictions', 'PredictionSource', 'ComparisonResult', 'merge_prediction_frames', 'evaluate_comparison', 'pairwise_delong_report', 'resolve_training_group_name', 'metrics_at_threshold', 'youden_threshold_metrics', 'apply_youden_threshold', 'target_threshold_metrics', 'apply_target_threshold'), 'habit.image_preprocessing': ('Preprocessor', 'PreprocessorRegistry', 'ZScoreNormalization', 'Resample', 'Reorientation', 'N4Correction', 'HistogramStandardization', 'AdaptiveHistogramEqualization', 'Registration'), 'habit.precision': ('ImagePerturbation', 'BSplineDeformPerturbation', 'GaussianNoisePerturbation', 'GradientWeightedPerturbation', 'ImagePerturbationRegistry', 'MorphologicalPerturbation', 'PerturbationChain', 'PreciseFeatureSet', 'RigidPerturbation', 'RotationPerturbation', 'SliceExtentPerturbation', 'TranslationPerturbation', 'aggregate_panels', 'align_habitat_map', 'habitat_stability', 'identify_precise_features', 'perturb_image', 'precision_panel', 'prior2024_retest_perturbation')}
PUBLIC_NAMESPACES["habit.habitat_features"] = tuple(
    name
    for name in PUBLIC_NAMESPACES["habit.habitat_features"]
    if name
    not in {
        "GraphHabitatFeaturesParams",
        "IthHabitatFeaturesParams",
        "MsiHabitatFeaturesParams",
        "NonRadiomicsHabitatFeaturesParams",
        "TraditionalRadiomicsHabitatFeaturesParams",
        "WholeHabitatRadiomicsFeaturesParams",
        "EachHabitatRadiomicsFeaturesParams",
        "HabitatVolumeFeaturesParams",
    }
)
PUBLIC_NAMESPACES["habit.combiners"] = tuple(
    name
    for name in PUBLIC_NAMESPACES["habit.combiners"]
    if not name.endswith("Params")
)
PUBLIC_NAMESPACES["habit.pipeline"] = tuple(
    name
    for name in PUBLIC_NAMESPACES["habit.pipeline"]
    if not name.endswith("Params")
)

def build_lazy_exports() -> Dict[str, Tuple[str, str]]:
    """Return no root-level component exports in v2."""
    return {}
