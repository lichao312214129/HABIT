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
"""Gap tests for optional-dependency graceful degradation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from habit.api.exceptions import OptionalDependencyError
from habit.classification import AutogluonTabularClassifier
from habit.viz.habitat_clustering import plot_habitat_clustering_pca_3d_interactive


def _make_binary_feature_table():
    """
    Build a minimal feature table for classifier optional-dependency tests.

    Returns:
        FeatureTable with two subjects and one signal column.
    """
    from habit.datasets.synthetic import make_synthetic_feature_table

    return make_synthetic_feature_table(n_rows=20, n_features=4, rng=42)


@pytest.mark.unit
def test_plotly_interactive_plot_raises_optional_dependency_error() -> None:
    """Without plotly, interactive 3D plots raise OptionalDependencyError."""
    if importlib.util.find_spec("plotly") is not None:
        pytest.skip("plotly is installed in this environment")
    features = np.random.default_rng(0).normal(size=(12, 3))
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int64)
    with pytest.raises(OptionalDependencyError, match="plotly"):
        plot_habitat_clustering_pca_3d_interactive(features, labels)


@pytest.mark.unit
def test_autogluon_classifier_raises_optional_dependency_error() -> None:
    """Without AutoGluon, fit fails with an actionable OptionalDependencyError."""
    if importlib.util.find_spec("autogluon") is not None:
        pytest.skip("AutoGluon is installed in this environment")
    classifier = AutogluonTabularClassifier()
    with pytest.raises(OptionalDependencyError, match="AutoGluon|automl"):
        classifier.fit(_make_binary_feature_table())

