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
"""Built-in evaluation metrics, the ``metric`` registry, and statistics wrappers."""

from __future__ import annotations

from habit.domain.evaluation.metrics import (
    AccuracyMetric,
    AccuracyMetricParams,
    AucMetric,
    AucMetricParams,
    F1ScoreMetric,
    F1ScoreMetricParams,
    HosmerLemeshowPValueMetric,
    HosmerLemeshowPValueMetricParams,
    NpvMetric,
    NpvMetricParams,
    PpvMetric,
    PpvMetricParams,
    SensitivityMetric,
    SensitivityMetricParams,
    SpecificityMetric,
    SpecificityMetricParams,
    SpiegelhalterZPValueMetric,
    SpiegelhalterZPValueMetricParams,
)
from habit.domain.evaluation.registry import MetricRegistry
from habit.domain.evaluation.regression_metrics import (
    MaeMetric,
    MaeMetricParams,
    MseMetric,
    MseMetricParams,
    R2Metric,
    R2MetricParams,
    RmseMetric,
    RmseMetricParams,
)
from habit.domain.evaluation.regression_registry import RegressionMetricRegistry
from habit.domain.evaluation.survival_metrics import (
    CIndexMetric,
    CIndexMetricParams,
    CumulativeDynamicAucMetric,
    CumulativeDynamicAucMetricParams,
    IntegratedBrierScoreMetric,
    IntegratedBrierScoreMetricParams,
)
from habit.domain.evaluation.survival_registry import SurvivalMetricRegistry
from habit.domain.evaluation.statistics import (
    AucConfidenceInterval,
    CalibrationResult,
    DelongResult,
    auc_confidence_interval,
    calibration_tests,
    delong_test,
    icc_analysis,
    repeat_measurement_matrix,
)
from habit.domain.evaluation.panel import (
    CleanedPredictions,
    clean_binary_predictions,
    compute_classification_metrics,
)
from habit.domain.evaluation.comparison import (
    ComparisonResult,
    MergedPredictions,
    PredictionSource,
    evaluate_comparison,
    merge_prediction_frames,
    pairwise_delong_report,
    resolve_training_group_name,
)
from habit.domain.evaluation.thresholds import (
    apply_target_threshold,
    apply_youden_threshold,
    metrics_at_threshold,
    target_threshold_metrics,
    youden_threshold_metrics,
)

__all__ = [
    "MetricRegistry",
    "SurvivalMetricRegistry",
    "RegressionMetricRegistry",
    "CIndexMetric",
    "CIndexMetricParams",
    "IntegratedBrierScoreMetric",
    "IntegratedBrierScoreMetricParams",
    "CumulativeDynamicAucMetric",
    "CumulativeDynamicAucMetricParams",
    "R2Metric",
    "R2MetricParams",
    "MaeMetric",
    "MaeMetricParams",
    "MseMetric",
    "MseMetricParams",
    "RmseMetric",
    "RmseMetricParams",
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
    "DelongResult",
    "delong_test",
    "AucConfidenceInterval",
    "auc_confidence_interval",
    "CalibrationResult",
    "calibration_tests",
    "repeat_measurement_matrix",
    "icc_analysis",
    "CleanedPredictions",
    "clean_binary_predictions",
    "compute_classification_metrics",
    "MergedPredictions",
    "PredictionSource",
    "ComparisonResult",
    "merge_prediction_frames",
    "evaluate_comparison",
    "pairwise_delong_report",
    "resolve_training_group_name",
    "metrics_at_threshold",
    "youden_threshold_metrics",
    "apply_youden_threshold",
    "target_threshold_metrics",
    "apply_target_threshold",
]
