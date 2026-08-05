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
Feature-preprocessing package for habitat analysis.

All preprocessing uses a unified DataFrame pipeline backed by
``PreprocessingMethodFactory`` and ``@PreprocessingMethodFactory.register``. Built-in
methods (including variance / correlation filters) live in :mod:`builtin_methods`.
"""

from .base_preprocessing import (
    BaseFeaturePreprocessing,
    BaselineStats,
    PreprocessingMethodFactory,
)
from .builtin_methods import (
    apply_correlation_filter,
    apply_variance_filter,
    select_correlation_columns,
    select_variance_columns,
)


def __getattr__(name: str):
    if name == "PreprocessingState":
        from .preprocessing_state import PreprocessingState
        return PreprocessingState
    if name == "apply_preprocessing_pipeline":
        from .pipeline import apply_preprocessing_pipeline
        return apply_preprocessing_pipeline
    if name == "apply_stateless_preprocessing":
        from .pipeline import apply_stateless_preprocessing
        return apply_stateless_preprocessing
    if name == "handle_extreme_values":
        from .value_transforms import handle_extreme_values
        return handle_extreme_values
    if name == "create_discretizer":
        from .value_transforms import create_discretizer
        return create_discretizer
    if name == "preprocess_features":
        from .value_transforms import preprocess_features
        return preprocess_features
    if name == "process_features_pipeline":
        from .value_transforms import process_features_pipeline
        return process_features_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseFeaturePreprocessing",
    "BaselineStats",
    "PreprocessingMethodFactory",
    "PreprocessingState",
    "apply_correlation_filter",
    "apply_preprocessing_pipeline",
    "apply_stateless_preprocessing",
    "apply_variance_filter",
    "create_discretizer",
    "handle_extreme_values",
    "preprocess_features",
    "process_features_pipeline",
    "select_correlation_columns",
    "select_variance_columns",
]
