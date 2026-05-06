"""Common constants used across the HABIT project.

This module provides a centralized location for all constants to ensure
consistency and avoid magic strings throughout the codebase.
"""

from enum import Enum


class DataColumns(str, Enum):
    """Standard column names for prediction data."""
    Y_TRUE = 'y_true'
    Y_PRED_PROBA = 'y_pred_proba'
    Y_PRED = 'y_pred'
    FILE_PATH = 'file_path'
    FOLD_ID = 'fold_id'


class MetricNames(str, Enum):
    """Standard metric names used throughout the project."""
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'
    SENSITIVITY = 'sensitivity'
    SPECIFICITY = 'specificity'
    PPV = 'ppv'
    NPV = 'npv'
    AUC = 'auc'
    AUPRC = 'auprc'
    YOUDEN_INDEX = 'youden_index'


class ThresholdTypes(str, Enum):
    """Types of thresholds used for binary classification."""
    YOUDEN = 'youden'
    TARGET = 'target'
    OPTIMAL = 'optimal'


class FileExtensions(str, Enum):
    """Standard file extensions."""
    CSV = '.csv'
    JSON = '.json'
    PKL = '.pkl'
    PDF = '.pdf'
    PNG = '.png'
    XLSX = '.xlsx'


class DefaultFileNames(str, Enum):
    """Default file names used throughout the project."""
    PREDICTIONS = 'predictions.csv'
    METRICS = 'metrics.json'
    RESULTS = 'results.json'
    SUMMARY = 'summary.csv'
    CONFIG = 'config.json'
    MODEL_BUNDLE = 'model_bundle.pkl'
    PERFORMANCE_TABLE = 'performance_table.csv'
    PERFORMANCE_DETAILED = 'performance_detailed.csv'
    MERGED_PREDICTIONS = 'merged_predictions.csv'
    DELONG_RESULTS = 'delong_results.json'


class PlotNames(str, Enum):
    """Standard plot names."""
    ROC_CURVE = 'roc_curve'
    PR_CURVE = 'pr_curve'
    CONFUSION_MATRIX = 'confusion_matrix'
    CALIBRATION_PLOT = 'calibration_plot'


class ModelFields(str, Enum):
    """Standard field names for model-related data."""
    MODEL_NAME = 'model_name'
    MODEL_ID = 'model_id'
    MODEL_TYPE = 'model_type'
    PRED_COL = 'pred_col'
    LABEL_COL = 'label_col'


class GroupNames(str, Enum):
    """Standard group names for organizing results."""
    ALL = 'all'
    TRAINING_SET = 'Training set'
    TEST_SET = 'Test set'
    VALIDATION_SET = 'Validation set'


class DataSections(str, Enum):
    """Standard section names for data organization."""
    METRICS = 'metrics'
    THRESHOLDS = 'thresholds'
    PREDICTIONS = 'predictions'
    PLOTS = 'plots'
    COMBINED_RESULTS = 'combined_results'


class ValidationMethods(str, Enum):
    """Validation method names."""
    K_FOLD = 'k_fold'
    LOO = 'leave_one_out'
    HOLDOUT = 'hold_out'


class ClusteringMethods(str, Enum):
    """Clustering algorithm names."""
    KMEANS = 'kmeans'
    HIERARCHICAL = 'hierarchical'
    DBSCAN = 'dbscan'
    MEAN_SHIFT = 'mean_shift'


class FeatureTypes(str, Enum):
    """Feature type names."""
    TEXTURE = 'texture'
    INTENSITY = 'intensity'
    SHAPE = 'shape'
    GLCM = 'glcm'
    GLRLM = 'glrlm'
    GLSZM = 'glszm'
    NGTDM = 'ngtdm'


class RegistryKeys(str, Enum):
    """Registry keys for various registries."""
    EXTRACTOR = 'extractor'
    CLUSTERING = 'clustering'
    METRIC = 'metric'


class OutputPrefixes(str, Enum):
    """Standard output file prefixes."""
    TRAIN = 'train_'
    TEST = 'test_'
    VALIDATION = 'validation_'
    COMBINED = 'combined_'


class CommonConstants:
    """Common numeric and string constants."""
    
    DEFAULT_NUM_WORKERS = 32
    
    DEFAULT_RANDOM_STATE = 42
    
    DEFAULT_TEST_SIZE = 0.2
    
    DEFAULT_VALIDATION_SIZE = 0.2
    
    DEFAULT_N_FOLDS = 5
    
    DEFAULT_CONFIDENCE_LEVEL = 0.95
    
    NAN_VALUE = float('nan')
    
    POSITIVE_LABEL = 1
    
    NEGATIVE_LABEL = 0


class PlotFormats(str, Enum):
    """Plot format constants."""
    PDF = 'pdf'
    PNG = 'png'
    SVG = 'svg'
    JPG = 'jpg'
    TIFF = 'tiff'


class ReportSections(str, Enum):
    """Report section names."""
    SUMMARY = 'summary'
    DETAILED = 'detailed'
    COMPARISON = 'comparison'
    STATISTICAL_TESTS = 'statistical_tests'


class ErrorMessages(str, Enum):
    """Common error messages."""
    SERVICE_NOT_REGISTERED = "Service '{name}' not registered in DI container"
    CONFIG_NOT_FOUND = "Configuration key '{key}' not found"
    INVALID_CONFIG_TYPE = "Config must be dict or Pydantic model, got {type}"
    MODEL_NOT_FOUND = "Model '{name}' not found"
    DATA_NOT_LOADED = "Data not loaded"
    INVALID_DATA_FORMAT = "Invalid data format"
