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
"""Threshold search / transfer helpers for binary model comparison.

Pure arrays in / nested dicts out. Numerics mirror the v0.1 comparison
workflow (Youden index, multi-target search with Pareto+Youden selection,
and closest-threshold fallback) so train-derived cut-offs applied to test
remain scientifically comparable.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Union

import numpy as np

from habit.evaluation.panel import (
    clean_binary_predictions,
    compute_classification_metrics,
)

__all__ = [
    "metrics_at_threshold",
    "youden_threshold_metrics",
    "apply_youden_threshold",
    "target_threshold_metrics",
    "apply_target_threshold",
]

MetricsDict = Dict[str, float]
ThresholdReport = Dict[str, Union[float, MetricsDict, Dict, List, str, None]]


def metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> MetricsDict:
    """
    Hard-label at ``threshold`` and compute the standard metric panel.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Positive-class probabilities.
        threshold: Decision threshold in ``[0, 1]``.

    Returns:
        Mapping of metric name -> float.
    """
    cleaned = clean_binary_predictions(y_true, y_prob, threshold=float(threshold))
    y_hat = (cleaned.y_prob >= float(threshold)).astype(np.float64)
    return compute_classification_metrics(cleaned.y_true, y_hat, cleaned.y_prob)


def youden_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> ThresholdReport:
    """
    Find the Youden-optimal threshold and report metrics at that cut-off.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Positive-class probabilities.

    Returns:
        ``{threshold, youden_index, metrics}``.
    """
    from sklearn.metrics import roc_curve

    cleaned = clean_binary_predictions(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(cleaned.y_true, cleaned.y_prob)
    youden_indices = tpr + (1.0 - fpr) - 1.0
    optimal_idx = int(np.argmax(youden_indices))
    optimal_threshold = float(thresholds[optimal_idx])
    return {
        "threshold": optimal_threshold,
        "youden_index": float(youden_indices[optimal_idx]),
        "metrics": metrics_at_threshold(
            cleaned.y_true, cleaned.y_prob, optimal_threshold
        ),
    }


def apply_youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> ThresholdReport:
    """
    Evaluate Youden-style metrics at a pre-chosen threshold (e.g. from train).

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Positive-class probabilities.
        threshold: Threshold transferred from the training split.

    Returns:
        ``{threshold, youden_index, metrics}``.
    """
    cleaned = clean_binary_predictions(y_true, y_prob)
    panel = metrics_at_threshold(cleaned.y_true, cleaned.y_prob, float(threshold))
    youden_index = float(panel["sensitivity"] + panel["specificity"] - 1.0)
    return {
        "threshold": float(threshold),
        "youden_index": youden_index,
        "metrics": panel,
    }


def target_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    targets: Mapping[str, float],
    *,
    threshold_selection: str = "pareto+youden",
    fallback_to_closest: bool = True,
    distance_metric: str = "euclidean",
) -> ThresholdReport:
    """
    Search thresholds that meet clinical target operating points.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Positive-class probabilities.
        targets: Metric name -> minimum required value, e.g.
            ``{"sensitivity": 0.8, "specificity": 0.8}``.
        threshold_selection: ``first`` / ``youden`` / ``pareto+youden``.
        fallback_to_closest: When no threshold meets all targets, return the
            closest operating point.
        distance_metric: Distance used by the closest-point fallback.

    Returns:
        Report with per-target thresholds, combined candidates, and either
        ``best_threshold`` or ``closest_threshold``.
    """
    from sklearn.metrics import roc_curve

    cleaned = clean_binary_predictions(y_true, y_prob)
    yt = cleaned.y_true
    yp = cleaned.y_prob
    fpr, tpr, thresholds = roc_curve(yt, yp)
    order = np.argsort(thresholds)[::-1]
    thresholds = thresholds[order]
    fpr = fpr[order]
    tpr = tpr[order]

    target_thresholds: Dict[str, float] = {}
    metrics_at_thresholds: Dict[str, MetricsDict] = {}
    roc_metric_names = {"sensitivity", "specificity"}
    other_metrics = {k for k in targets if k not in roc_metric_names}

    for metric_name, target_value in targets.items():
        best_threshold: Optional[float] = None
        if metric_name == "sensitivity":
            for i, val in enumerate(tpr):
                if val >= target_value:
                    best_threshold = float(thresholds[i])
                    break
        elif metric_name == "specificity":
            for i in range(len(fpr) - 1, -1, -1):
                if (1.0 - fpr[i]) >= target_value:
                    best_threshold = float(thresholds[i])
                    break
        elif metric_name in {"ppv", "npv", "f1_score", "accuracy"}:
            for thresh in thresholds:
                panel = metrics_at_threshold(yt, yp, float(thresh))
                if panel.get(metric_name, 0.0) >= target_value:
                    best_threshold = float(thresh)
                    break
        if best_threshold is not None:
            target_thresholds[metric_name] = best_threshold
            metrics_at_thresholds[metric_name] = metrics_at_threshold(
                yt, yp, best_threshold
            )

    combined_results: Dict[str, Dict[float, MetricsDict]] = {}
    if targets:
        for i, thresh in enumerate(thresholds):
            meets_all = True
            if "sensitivity" in targets and tpr[i] < targets["sensitivity"]:
                meets_all = False
            if "specificity" in targets and (1.0 - fpr[i]) < targets["specificity"]:
                meets_all = False
            full_metrics: Optional[MetricsDict] = None
            if meets_all:
                full_metrics = metrics_at_threshold(yt, yp, float(thresh))
                for metric_name in other_metrics:
                    if full_metrics.get(metric_name, 0.0) < targets[metric_name]:
                        meets_all = False
                        break
            if meets_all and full_metrics is not None:
                combined_key = " & ".join(sorted(targets.keys()))
                bucket = combined_results.setdefault(combined_key, {})
                bucket[float(thresh)] = full_metrics

    best_info = (
        _select_best_threshold(combined_results, threshold_selection)
        if combined_results
        else None
    )
    closest_info = None
    if fallback_to_closest and not combined_results:
        closest_info = _find_closest_threshold(
            yt, yp, thresholds, dict(targets), distance_metric
        )

    return {
        "thresholds": target_thresholds,
        "metrics_at_thresholds": metrics_at_thresholds,
        "combined_results": combined_results,
        "best_threshold": best_info,
        "closest_threshold": closest_info,
    }


def apply_target_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> ThresholdReport:
    """
    Apply a train-selected target threshold on a new split.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Positive-class probabilities.
        threshold: Threshold transferred from the training split.

    Returns:
        ``{threshold, metrics}``.
    """
    cleaned = clean_binary_predictions(y_true, y_prob)
    return {
        "threshold": float(threshold),
        "metrics": metrics_at_threshold(
            cleaned.y_true, cleaned.y_prob, float(threshold)
        ),
    }


def _select_best_threshold(
    combined_results: Mapping[str, Mapping[float, MetricsDict]],
    strategy: str,
) -> Optional[Dict[str, Union[float, MetricsDict, str, int]]]:
    """Pick one threshold from the set that jointly meets all targets."""
    all_thresholds: Dict[float, MetricsDict] = {}
    for thresh_dict in combined_results.values():
        all_thresholds.update(thresh_dict)
    if not all_thresholds:
        return None

    if strategy == "first":
        first_thresh = next(iter(all_thresholds.keys()))
        return {
            "threshold": first_thresh,
            "metrics": all_thresholds[first_thresh],
            "strategy": "first",
        }

    if strategy == "youden":
        best_thresh, best_metrics = max(
            all_thresholds.items(),
            key=lambda item: item[1].get("sensitivity", 0.0)
            + item[1].get("specificity", 0.0)
            - 1.0,
        )
        return {
            "threshold": best_thresh,
            "metrics": best_metrics,
            "youden_index": best_metrics.get("sensitivity", 0.0)
            + best_metrics.get("specificity", 0.0)
            - 1.0,
            "strategy": "youden",
        }

    # Default / recommended: Pareto front, then max Youden.
    pareto = _find_pareto_optimal(all_thresholds)
    best_thresh, best_metrics = max(
        pareto.items(),
        key=lambda item: item[1].get("sensitivity", 0.0)
        + item[1].get("specificity", 0.0)
        - 1.0,
    )
    return {
        "threshold": best_thresh,
        "metrics": best_metrics,
        "youden_index": best_metrics.get("sensitivity", 0.0)
        + best_metrics.get("specificity", 0.0)
        - 1.0,
        "strategy": "pareto+youden",
        "pareto_optimal_count": len(pareto),
    }


def _find_pareto_optimal(
    thresholds_dict: Mapping[float, MetricsDict],
) -> Dict[float, MetricsDict]:
    """Keep thresholds not dominated in every reported metric."""
    pareto: Dict[float, MetricsDict] = {}
    for thresh1, metrics1 in thresholds_dict.items():
        dominated = False
        for thresh2, metrics2 in thresholds_dict.items():
            if thresh1 == thresh2:
                continue
            better_or_equal = all(
                metrics2.get(key, 0.0) >= metrics1.get(key, 0.0) for key in metrics1
            )
            strictly_better = any(
                metrics2.get(key, 0.0) > metrics1.get(key, 0.0) for key in metrics1
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto[thresh1] = metrics1
    return pareto


def _find_closest_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    targets: Mapping[str, float],
    distance_metric: str,
) -> Optional[Dict[str, Union[float, MetricsDict, str, List[str]]]]:
    """Fallback when no threshold jointly satisfies every target."""
    best_threshold: Optional[float] = None
    best_distance = float("inf")
    best_metrics: Optional[MetricsDict] = None
    satisfied_targets: List[str] = []

    for thresh in thresholds:
        panel = metrics_at_threshold(y_true, y_prob, float(thresh))
        distances: List[float] = []
        current_satisfied: List[str] = []
        for metric_name, target_value in targets.items():
            actual = panel.get(metric_name, 0.0)
            diff = float(actual) - float(target_value)
            if actual >= target_value:
                current_satisfied.append(metric_name)
            if distance_metric == "manhattan":
                distances.append(abs(diff))
            elif distance_metric == "max":
                distances.append(abs(diff))
            else:
                distances.append(diff**2)
        if distance_metric == "manhattan":
            distance = float(sum(distances))
        elif distance_metric == "max":
            distance = float(max(distances)) if distances else float("inf")
        else:
            distance = float(np.sqrt(sum(distances)))

        if len(current_satisfied) > len(satisfied_targets) or (
            len(current_satisfied) == len(satisfied_targets)
            and distance < best_distance
        ):
            best_distance = distance
            best_threshold = float(thresh)
            best_metrics = panel
            satisfied_targets = current_satisfied

    if best_threshold is None or best_metrics is None:
        return None
    return {
        "threshold": best_threshold,
        "metrics": best_metrics,
        "distance_to_target": best_distance,
        "distance_metric": distance_metric,
        "satisfied_targets": satisfied_targets,
        "unsatisfied_targets": [
            key for key in targets if key not in satisfied_targets
        ],
        "warning": (
            "No threshold satisfies all targets. This is the closest match."
        ),
    }
