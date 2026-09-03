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
"""Built-in regression evaluation metrics (domain ``regression_metric``)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from habit.evaluation.regression_registry import RegressionMetricRegistry
from habit.spec.specs import Spec

__all__ = [
    "R2Metric",
    "MaeMetric",
    "MseMetric",
    "RmseMetric",
]


class _SpecParamsMixin:
    """Build ``spec.params`` from the constructor-stored parameter mapping."""

    _spec_name: str = ""
    _params: Dict[str, Any] = {}

    @property
    def spec(self) -> Spec:
        """Return the metric specification."""
        return Spec(name=self._spec_name, params=dict(self._params))


@RegressionMetricRegistry.register("r2")
class R2Metric(_SpecParamsMixin):
    """Coefficient of determination; 1.0 is perfect, 0.0 predicts the mean."""

    greater_is_better = True
    _spec_name = "r2"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return R-squared between true and predicted responses."""
        from sklearn.metrics import r2_score

        return float(r2_score(np.asarray(y_true), np.asarray(y_pred)))


@RegressionMetricRegistry.register("mae")
class MaeMetric(_SpecParamsMixin):
    """Mean absolute error; robust to outliers, lower is better."""

    greater_is_better = False
    _spec_name = "mae"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the mean absolute error."""
        from sklearn.metrics import mean_absolute_error

        return float(mean_absolute_error(np.asarray(y_true), np.asarray(y_pred)))


@RegressionMetricRegistry.register("mse")
class MseMetric(_SpecParamsMixin):
    """Mean squared error; penalises large errors, lower is better."""

    greater_is_better = False
    _spec_name = "mse"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the mean squared error."""
        from sklearn.metrics import mean_squared_error

        return float(mean_squared_error(np.asarray(y_true), np.asarray(y_pred)))


@RegressionMetricRegistry.register("rmse")
class RmseMetric(_SpecParamsMixin):
    """Root mean squared error, in the response's own units; lower is better."""

    greater_is_better = False
    _spec_name = "rmse"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the root mean squared error."""
        from sklearn.metrics import root_mean_squared_error

        return float(root_mean_squared_error(np.asarray(y_true), np.asarray(y_pred)))


# --- Parameter-schema wiring -----------------------------------------------

