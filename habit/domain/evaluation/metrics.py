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
"""Built-in evaluation metrics (domain ``metric``).

The nine plugins migrate the v0.1
``habit.core.machine_learning.evaluation.metrics`` registry under their
original spellings, with numerics preserved exactly:

- the label-based metrics (``accuracy`` ... ``f1_score``) are computed from
  the same confusion-matrix convention sklearn uses (sorted label union),
  now in plain NumPy via :func:`habit.domain.evaluation._base.confusion_matrix`;
- ``auc`` keeps the v0.1 sklearn calls (binary, plus ``multi_class="ovr"``
  for probability matrices) and therefore imports sklearn lazily inside the
  call body (L3 layer rule);
- the two calibration p-values delegate to the L0 kernels
  (:mod:`habit.kernels.statistics`) and answer ``NaN`` wherever the v0.1
  implementations caught an error or saw a multi-class problem.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from habit.domain.evaluation._base import binary_class_scores, confusion_matrix
from habit.domain.evaluation.registry import MetricRegistry
from habit.kernels.statistics import hosmer_lemeshow_test, spiegelhalter_z_test
from habit.spec.specs import Spec

__all__ = [
    "AccuracyMetric",
    "AccuracyMetricParams",
    "SensitivityMetric",
    "SensitivityMetricParams",
    "SpecificityMetric",
    "SpecificityMetricParams",
    "PpvMetric",
    "PpvMetricParams",
    "NpvMetric",
    "NpvMetricParams",
    "F1ScoreMetric",
    "F1ScoreMetricParams",
    "AucMetric",
    "AucMetricParams",
    "HosmerLemeshowPValueMetric",
    "HosmerLemeshowPValueMetricParams",
    "SpiegelhalterZPValueMetric",
    "SpiegelhalterZPValueMetricParams",
]


class _SpecMixin:
    """Build ``spec`` from the registered name and the parameter mapping."""

    _spec_name: str = ""
    _params: Dict[str, Any] = {}

    @property
    def spec(self) -> Spec:
        """Return the metric specification."""
        return Spec(name=self._spec_name, params=dict(self._params))


# ---------------------------------------------------------------------------
# Confusion-matrix derived quantities (the v0.1 formulas, on one shared cm)
# ---------------------------------------------------------------------------


def _sensitivity_from_cm(cm: np.ndarray) -> float:
    """Recall: binary true-positive rate, macro per-class mean otherwise."""
    if cm.shape == (2, 2):
        denominator = cm[1, 1] + cm[1, 0]
        return float(cm[1, 1] / denominator) if denominator > 0 else 0.0
    recalls = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))


def _specificity_from_cm(cm: np.ndarray) -> float:
    """True negative rate: binary case, macro per-class mean otherwise."""
    if cm.shape == (2, 2):
        denominator = cm[0, 0] + cm[0, 1]
        return float(cm[0, 0] / denominator) if denominator > 0 else 0.0
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specificities))


def _ppv_from_cm(cm: np.ndarray) -> float:
    """Precision: binary PPV, macro per-class mean otherwise."""
    if cm.shape == (2, 2):
        denominator = cm[1, 1] + cm[0, 1]
        return float(cm[1, 1] / denominator) if denominator > 0 else 0.0
    precisions = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(precisions))


def _npv_from_cm(cm: np.ndarray) -> float:
    """Negative predictive value: binary case, macro per-class mean."""
    if cm.shape == (2, 2):
        denominator = cm[0, 0] + cm[1, 0]
        return float(cm[0, 0] / denominator) if denominator > 0 else 0.0
    npvs = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
    return float(np.mean(npvs))


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


class AccuracyMetricParams(BaseModel):
    """Constructor parameters for :class:`AccuracyMetric` (none)."""

    model_config = ConfigDict(extra="forbid")
@MetricRegistry.register("accuracy")
class AccuracyMetric(_SpecMixin):
    """Fraction of exactly matching labels."""

    needs_proba = False
    greater_is_better = True
    _spec_name = "accuracy"

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return the exact-match fraction between true and predicted labels."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# sensitivity / specificity / ppv / npv / f1_score
# ---------------------------------------------------------------------------


class SensitivityMetricParams(BaseModel):
    """Constructor parameters for :class:`SensitivityMetric` (none)."""

    model_config = ConfigDict(extra="forbid")
@MetricRegistry.register("sensitivity")
class SensitivityMetric(_SpecMixin):
    """Sensitivity (recall, true positive rate); macro mean when multi-class."""

    needs_proba = False
    greater_is_better = True
    _spec_name = "sensitivity"

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return sensitivity of the predictions."""
        cm, _ = confusion_matrix(y_true, y_pred)
        return _sensitivity_from_cm(cm)


class SpecificityMetricParams(BaseModel):
    """Constructor parameters for :class:`SpecificityMetric` (none)."""

    model_config = ConfigDict(extra="forbid")
@MetricRegistry.register("specificity")
class SpecificityMetric(_SpecMixin):
    """Specificity (true negative rate); macro mean when multi-class."""

    needs_proba = False
    greater_is_better = True
    _spec_name = "specificity"

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return specificity of the predictions."""
        cm, _ = confusion_matrix(y_true, y_pred)
        return _specificity_from_cm(cm)


class PpvMetricParams(BaseModel):
    """Constructor parameters for :class:`PpvMetric` (none)."""

    model_config = ConfigDict(extra="forbid")
@MetricRegistry.register("ppv")
class PpvMetric(_SpecMixin):
    """Positive predictive value (precision); macro mean when multi-class."""

    needs_proba = False
    greater_is_better = True
    _spec_name = "ppv"

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return the positive predictive value of the predictions."""
        cm, _ = confusion_matrix(y_true, y_pred)
        return _ppv_from_cm(cm)


class NpvMetricParams(BaseModel):
    """Constructor parameters for :class:`NpvMetric` (none)."""

    model_config = ConfigDict(extra="forbid")
@MetricRegistry.register("npv")
class NpvMetric(_SpecMixin):
    """Negative predictive value; macro per-class mean when multi-class."""

    needs_proba = False
    greater_is_better = True
    _spec_name = "npv"

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return the negative predictive value of the predictions."""
        cm, _ = confusion_matrix(y_true, y_pred)
        return _npv_from_cm(cm)


class F1ScoreMetricParams(BaseModel):
    """Constructor parameters for :class:`F1ScoreMetric` (none)."""

    model_config = ConfigDict(extra="forbid")
@MetricRegistry.register("f1_score")
class F1ScoreMetric(_SpecMixin):
    """Harmonic mean of PPV and sensitivity over one shared confusion matrix."""

    needs_proba = False
    greater_is_better = True
    _spec_name = "f1_score"

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return the F1 score (0 when precision and recall are both 0)."""
        cm, _ = confusion_matrix(y_true, y_pred)
        precision = _ppv_from_cm(cm)
        recall = _sensitivity_from_cm(cm)
        if (precision + recall) == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))


# ---------------------------------------------------------------------------
# auc
# ---------------------------------------------------------------------------


class AucMetricParams(BaseModel):
    """Constructor parameters for :class:`AucMetric` (none)."""

    model_config = ConfigDict(extra="forbid")
@MetricRegistry.register("auc")
class AucMetric(_SpecMixin):
    """
    ROC AUC over the positive-class scores.

    Binary problems call sklearn's ``roc_auc_score`` directly; an ``(n, k)``
    probability matrix triggers the v0.1 multi-class branch
    (``multi_class="ovr"``). sklearn is imported lazily inside the call so
    importing this module stays cheap (L3 layer rule).
    """

    needs_proba = True
    greater_is_better = True
    _spec_name = "auc"

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return the ROC AUC of the scores."""
        from sklearn.metrics import roc_auc_score

        raw = np.asarray(y_score) if y_score is not None else None
        if raw is not None and raw.ndim == 2 and raw.shape[1] > 1:
            # v0.1 multi-class branch: one-vs-rest over the probability matrix.
            return float(
                roc_auc_score(
                    y_true, raw.astype(np.float64), multi_class="ovr"
                )
            )
        scores = binary_class_scores(y_score, owner=self._spec_name)
        return float(roc_auc_score(y_true, scores))


# ---------------------------------------------------------------------------
# hosmer_lemeshow_p_value / spiegelhalter_z_p_value (L0 kernel delegates)
# ---------------------------------------------------------------------------


class HosmerLemeshowPValueMetricParams(BaseModel):
    """Constructor parameters for :class:`HosmerLemeshowPValueMetric`."""

    model_config = ConfigDict(extra="forbid")
    #: Number of quantile-based risk groups (classically 10, the decile test).
    n_groups: int = 10


@MetricRegistry.register("hosmer_lemeshow_p_value")
class HosmerLemeshowPValueMetric(_SpecMixin):
    """
    Hosmer-Lemeshow calibration p-value (binary outcomes only).

    A HIGH p-value means the calibration-null is not rejected, hence
    ``greater_is_better = True``. Multi-class problems and degenerate inputs
    (e.g. too many tied probabilities to form the risk groups) answer
    ``NaN``, mirroring the v0.1 fail-soft behaviour.
    """

    needs_proba = True
    greater_is_better = True
    _spec_name = "hosmer_lemeshow_p_value"

    def __init__(self, n_groups: int = 10) -> None:
        self._params = {"n_groups": n_groups}
        self._n_groups = int(n_groups)

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return the H-L p-value, or ``NaN`` where the test is undefined."""
        scores = binary_class_scores(y_score, owner=self._spec_name)
        if scores is None:
            return float("nan")
        try:
            _, p_value = hosmer_lemeshow_test(
                np.asarray(y_true, dtype=np.float64),
                scores,
                n_groups=self._n_groups,
            )
        except Exception:
            # v0.1 semantics: any failure of the test reads as "undefined".
            return float("nan")
        return float(p_value)


class SpiegelhalterZPValueMetricParams(BaseModel):
    """Constructor parameters for :class:`SpiegelhalterZPValueMetric` (none)."""

    model_config = ConfigDict(extra="forbid")
@MetricRegistry.register("spiegelhalter_z_p_value")
class SpiegelhalterZPValueMetric(_SpecMixin):
    """
    Spiegelhalter Z-test calibration p-value (binary outcomes only).

    A HIGH p-value means the calibration-null is not rejected, hence
    ``greater_is_better = True``. Multi-class problems and degenerate inputs
    answer ``NaN``, mirroring the v0.1 fail-soft behaviour.
    """

    needs_proba = True
    greater_is_better = True
    _spec_name = "spiegelhalter_z_p_value"

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        """Return the Spiegelhalter p-value, or ``NaN`` where undefined."""
        scores = binary_class_scores(y_score, owner=self._spec_name)
        if scores is None:
            return float("nan")
        try:
            _, p_value = spiegelhalter_z_test(
                np.asarray(y_true, dtype=np.float64),
                scores,
            )
        except Exception:
            # v0.1 semantics: any failure of the test reads as "undefined".
            return float("nan")
        return float(p_value)


# --- Parameter-schema wiring for introspection (`get_param_schema`) --------

MetricRegistry.register_params_model("accuracy", AccuracyMetricParams)
MetricRegistry.register_params_model("sensitivity", SensitivityMetricParams)
MetricRegistry.register_params_model("specificity", SpecificityMetricParams)
MetricRegistry.register_params_model("ppv", PpvMetricParams)
MetricRegistry.register_params_model("npv", NpvMetricParams)
MetricRegistry.register_params_model("f1_score", F1ScoreMetricParams)
MetricRegistry.register_params_model("auc", AucMetricParams)
MetricRegistry.register_params_model(
    "hosmer_lemeshow_p_value", HosmerLemeshowPValueMetricParams
)
MetricRegistry.register_params_model(
    "spiegelhalter_z_p_value", SpiegelhalterZPValueMetricParams
)
