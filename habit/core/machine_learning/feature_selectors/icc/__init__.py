# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
ICC and Reliability Metrics Module

This module provides comprehensive reliability analysis including:
    - Intraclass Correlation Coefficient (ICC) - all 6 types
    - Cohen's Kappa (2 raters)
    - Fleiss' Kappa (multiple raters)
    - Krippendorff's Alpha
    - Unified interface for all calculations

Example Usage:
    # Simple analysis
    >>> from habit.core.machine_learning.feature_selectors.icc.icc_analyzer import (
    ...     analyze_features,
    ...     save_results,
    ...     print_summary
    ... )
    >>> results = analyze_features(['file1.csv', 'file2.csv'], metrics=['icc2', 'icc3'])
    >>> save_results(results, 'output.json')
"""

# Import main analysis functions and classes from unified module
from .icc_analyzer import (
    # Main high-level analysis function
    analyze_features,
    
    # Result handlers and helpers
    save_results,
    print_summary,
    print_statistics,
    
    # Core metric classes and factory
    create_metric,
    MetricResult,
    ICCType,
    ICCMetric,
    MultiICCMetric,
    CohenKappaMetric,
    FleissKappaMetric,
    KrippendorffAlphaMetric,

    # Low-level data helpers (optional to expose)
    load_and_merge_data,
    find_common_indices,
    find_common_columns,
    prepare_long_format,
)

__all__ = [
    # Main analysis functions
    'analyze_features',
    'save_results',
    'print_summary',
    'print_statistics',
    
    # Core metric classes and factory
    'create_metric',
    'MetricResult',
    'ICCType',
    'ICCMetric',
    'MultiICCMetric',
    'CohenKappaMetric',
    'FleissKappaMetric',
    'KrippendorffAlphaMetric',

    # Data processing functions
    'load_and_merge_data',
    'find_common_indices',
    'find_common_columns',
    'prepare_long_format',
]