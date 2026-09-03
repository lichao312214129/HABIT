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
"""Built-in survival evaluation metrics (domain ``survival_metric``).

The discrimination and accuracy measures survival papers report, delegated to
scikit-survival -- the reference implementations -- with imports kept lazy so
the module loads without the optional ``analysis`` extra:

- ``c_index`` is Harrell's concordance index, the survival analogue of AUC
  for a single risk score;
- ``integrated_brier_score`` summarises calibration and discrimination of a
  predicted survival FUNCTION over the follow-up range;
- ``cumulative_dynamic_auc`` is Uno's time-dependent AUC over a time grid.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from habit.exceptions import HABITAPIError
from habit.evaluation.survival_registry import SurvivalMetricRegistry
from habit.spec.specs import Spec

__all__ = [
    "CIndexMetric",
    "IntegratedBrierScoreMetric",
    "CumulativeDynamicAucMetric",
]


def _sksurv_metrics(owner: str):
    """Import scikit-survival's metrics or raise with the install hint."""
    try:
        from sksurv import metrics as _m  # type: ignore
    except ImportError as exc:
        raise HABITAPIError(
            f"survival_metric.{owner} needs scikit-survival; install the "
            "'analysis' extra (pip install \"habitat-analysis[analysis]\")."
        ) from exc
    return _m


def _as_surv(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Rebuild scikit-survival's structured target from time + event."""
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.bool_)
    target = np.empty(time.size, dtype=[("event", np.bool_), ("time", np.float64)])
    target["event"] = event
    target["time"] = time
    return target


def _evaluation_grid(
    time: np.ndarray, event: np.ndarray, n_times: int
) -> np.ndarray:
    """
    Build a time grid strictly inside the observed follow-up range.

    scikit-survival requires every evaluation time to satisfy
    ``min(time) <= t < max(time)``: the tail beyond the last observed time is
    unidentifiable (no one is still at risk), and the IPCW estimate degrades
    at the exact upper boundary. The grid therefore runs from the smallest
    OBSERVED-EVENT time up to just below the largest follow-up time, which is
    the convention ``sksurv`` uses in its own examples.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.bool_)
    event_times = time[event]
    lower = float(event_times.min()) if event_times.size else float(time.min())
    # Step back one grid spacing from the top so the largest grid point stays
    # strictly below max(time), as _check_times requires.
    upper = float(time.max())
    span = upper - lower
    if span <= 0:
        return np.array([lower])
    step = span / max(n_times, 2)
    stop = upper - 0.5 * step
    return np.linspace(lower, stop, n_times)


class _SpecParamsMixin:
    """Build ``spec.params`` from the constructor-stored parameter mapping."""

    _spec_name: str = ""
    _params: Dict[str, Any] = {}

    @property
    def spec(self) -> Spec:
        """Return the metric specification."""
        return Spec(name=self._spec_name, params=dict(self._params))


@SurvivalMetricRegistry.register("c_index")
class CIndexMetric(_SpecParamsMixin):
    """
    Harrell's concordance index between the risk scores and the outcome.

    The proportion of comparable subject pairs whose predicted risk ordering
    agrees with the observed survival ordering; 0.5 is no discrimination, 1.0
    perfect. Consumes risk scores, so it needs no survival function.
    """

    needs_survival_function = False
    greater_is_better = True
    _spec_name = "c_index"

    def __call__(
        self,
        time: np.ndarray,
        event: np.ndarray,
        prediction: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> float:
        """Return Harrell's C-index of the risk scores."""
        metrics = _sksurv_metrics(self._spec_name)
        risk = np.asarray(prediction, dtype=np.float64)
        value, *_ = metrics.concordance_index_censored(
            np.asarray(event, dtype=np.bool_),
            np.asarray(time, dtype=np.float64),
            risk,
        )
        return float(value)




@SurvivalMetricRegistry.register("integrated_brier_score")
class IntegratedBrierScoreMetric(_SpecParamsMixin):
    """
    Integrated Brier score of the predicted survival function.

    Mean squared error between the predicted survival probability and the
    actual (censoring-adjusted) survival status, integrated over the observed
    follow-up range; lower is better, 0.25 is the reference of predicting the
    population-average survival. Needs S(t|x), hence
    ``needs_survival_function = True``.
    """

    needs_survival_function = True
    greater_is_better = False
    _spec_name = "integrated_brier_score"

    def __init__(self, n_times: int = 100) -> None:
        """Initialize numerical integration with at least two time points."""
        if isinstance(n_times, bool) or not isinstance(n_times, int) or n_times < 2:
            raise ValueError(f"n_times must be an integer >= 2; got {n_times!r}.")
        self.n_times = n_times
        self._params = {"n_times": n_times}

    def __call__(
        self,
        time: np.ndarray,
        event: np.ndarray,
        prediction: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> float:
        """Return the integrated Brier score of the survival probabilities."""
        metrics = _sksurv_metrics(self._spec_name)
        probability = np.asarray(prediction, dtype=np.float64)
        if probability.ndim != 2:
            raise HABITAPIError(
                "survival_metric.integrated_brier_score needs an "
                "(n_subjects, n_times) survival-probability matrix; got a "
                f"{probability.ndim}-D array. It must come from "
                "predict_survival_function, not predict_risk."
            )
        if times is None:
            raise HABITAPIError(
                "survival_metric.integrated_brier_score needs `times`: the "
                "evaluation times the probability columns correspond to. Pass "
                "the same grid predict_survival_function was given."
            )
        grid = np.asarray(times, dtype=np.float64)
        if grid.ndim != 1 or grid.size != probability.shape[1]:
            raise HABITAPIError(
                f"survival_metric.integrated_brier_score got {probability.shape[1]} "
                f"probability columns but {grid.size} times; they must match."
            )
        target = _as_surv(time, event)
        ibs = metrics.integrated_brier_score(target, target, probability, grid)
        return float(ibs)




@SurvivalMetricRegistry.register("cumulative_dynamic_auc")
class CumulativeDynamicAucMetric(_SpecParamsMixin):
    """
    Uno's cumulative/dynamic time-dependent AUC, averaged over the grid.

    Discrimination of the risk score as a function of follow-up time:
    at each t, the AUC between subjects who experienced the event by t and
    those still event-free at t. Consumes risk scores.
    """

    needs_survival_function = False
    greater_is_better = True
    _spec_name = "cumulative_dynamic_auc"

    def __init__(self, n_times: int = 100) -> None:
        """Initialize time-dependent AUC with at least two grid points."""
        if isinstance(n_times, bool) or not isinstance(n_times, int) or n_times < 2:
            raise ValueError(f"n_times must be an integer >= 2; got {n_times!r}.")
        self.n_times = n_times
        self._params = {"n_times": n_times}

    def __call__(
        self,
        time: np.ndarray,
        event: np.ndarray,
        prediction: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> float:
        """Return the mean cumulative/dynamic AUC over the time grid."""
        metrics = _sksurv_metrics(self._spec_name)
        risk = np.asarray(prediction, dtype=np.float64)
        grid = _evaluation_grid(
            np.asarray(time, dtype=np.float64),
            np.asarray(event, dtype=np.bool_),
            self.n_times,
        )
        target = _as_surv(time, event)
        _, mean_auc = metrics.cumulative_dynamic_auc(target, target, risk, grid)
        return float(mean_auc)


# --- Parameter-schema wiring -----------------------------------------------

