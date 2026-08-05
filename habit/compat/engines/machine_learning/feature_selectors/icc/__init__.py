# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    >>> from habit.compat.engines.machine_learning.feature_selectors.icc.icc_analyzer import (
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