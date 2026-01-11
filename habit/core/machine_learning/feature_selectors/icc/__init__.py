"""
ICC and Reliability Metrics Module

This module provides comprehensive reliability analysis including:
    - Intraclass Correlation Coefficient (ICC) - all 6 types
    - Cohen's Kappa (2 raters)
    - Fleiss' Kappa (multiple raters)
    - Krippendorff's Alpha
    - Extensible framework for custom metrics

Example Usage:
    # Simple ICC calculation (backward compatible)
    >>> from habit.core.machine_learning.feature_selectors.icc import calculate_icc
    >>> results = calculate_icc(['rater1.csv', 'rater2.csv'], logger)
    
    # Multiple metrics with full results
    >>> from habit.core.machine_learning.feature_selectors.icc import (
    ...     calculate_reliability_metrics,
    ...     ICCType,
    ...     create_metric
    ... )
    >>> results = calculate_reliability_metrics(
    ...     ['rater1.csv', 'rater2.csv'],
    ...     logger,
    ...     metrics=['icc2', 'icc3', 'fleiss_kappa'],
    ...     return_full_results=True
    ... )
    
    # Direct metric usage
    >>> metric = create_metric('icc3')
    >>> result = metric.calculate(data, 'subject', 'rater', 'score')
    >>> print(f"ICC = {result.value:.3f}, 95% CI = [{result.ci95_lower:.3f}, {result.ci95_upper:.3f}]")
"""

# Import main analysis functions
from .icc import (
    calculate_icc,
    calculate_reliability_metrics,
    analyze_multiple_groups,
    read_file,
    configure_logger,
    parse_files_groups,
    parse_directories,
    parse_features,
    parse_metrics,
    DEFAULT_METRICS,
)

# Import reliability metric classes and utilities
from .reliability_metrics import (
    # Enums
    ICCType,
    KappaType,
    
    # Result container
    MetricResult,
    
    # Base class
    BaseReliabilityMetric,
    
    # Metric implementations
    ICCMetric,
    MultiICCMetric,
    CohenKappaMetric,
    FleissKappaMetric,
    KrippendorffAlphaMetric,
    
    # Factory and registry functions
    create_metric,
    calculate_reliability,
    get_available_metrics,
    register_metric,
)

__all__ = [
    # Main analysis functions
    'calculate_icc',
    'calculate_reliability_metrics',
    'analyze_multiple_groups',
    'read_file',
    'configure_logger',
    'parse_files_groups',
    'parse_directories',
    'parse_features',
    'parse_metrics',
    'DEFAULT_METRICS',
    
    # Enums
    'ICCType',
    'KappaType',
    
    # Result container
    'MetricResult',
    
    # Base class
    'BaseReliabilityMetric',
    
    # Metric implementations
    'ICCMetric',
    'MultiICCMetric',
    'CohenKappaMetric',
    'FleissKappaMetric',
    'KrippendorffAlphaMetric',
    
    # Factory and registry
    'create_metric',
    'calculate_reliability',
    'get_available_metrics',
    'register_metric',
]
