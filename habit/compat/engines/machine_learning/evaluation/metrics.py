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
Metrics Calculation Module
Provides a registry-based system for calculating model evaluation metrics.

Optimizations:
- Confusion matrix caching for performance
- Extended target metrics support (PPV, NPV, F1, etc.)
- Fallback mechanism when no threshold satisfies all targets
- Pareto optimal threshold selection
- Category-based metric filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List, Callable, Any, Optional, Literal
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, f1_score as sklearn_f1_score
from scipy import stats
from habit.registry.base import CallableRegistry
from habit.kernels.statistics import (
    delong_roc_ci,
    delong_roc_test,
    delong_roc_variance,
    hosmer_lemeshow_test,
    spiegelhalter_z_test,
)

# --- Metric Registry System ---


class MetricRegistry(CallableRegistry[Callable]):
    """
    Registry of model evaluation metric functions.

    Uses the shared :class:`~habit.core.common.registry.CallableRegistry`
    contract (``register`` / ``get`` / ``available`` / ``get_entry`` /
    ``entries`` / ...). Register a metric with::

        @MetricRegistry.register("accuracy", display_name="Accuracy")
        def calc_accuracy(y_true, y_pred, y_prob, cm=None):
            ...

    Each entry stores ``func`` + ``display_name`` + ``category``
    (``'basic'`` / ``'statistical'`` / ``'clinical'``).
    """

    kind = "metric"

    #: Metrics default to the ``basic`` category unless overridden.
    default_metadata: Dict[str, Any] = {"category": "basic"}


# --- Confusion Matrix Cache Class ---

class MetricsCache:
    """
    Cache for confusion matrix and derived metrics to avoid repeated calculations.
    Provides ~8x performance improvement when calculating multiple metrics.
    """
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob)
        self._cm = None
        self._cached_metrics = {}
    
    @property
    def confusion_matrix(self) -> np.ndarray:
        """Lazy evaluation with caching of confusion matrix."""
        if self._cm is None:
            self._cm = metrics.confusion_matrix(self.y_true, self.y_pred)
        return self._cm
    
    def get_metric(self, metric_name: str, calculator: Callable) -> float:
        """Get cached metric or calculate and cache it."""
        if metric_name not in self._cached_metrics:
            self._cached_metrics[metric_name] = calculator(
                self.y_true, self.y_pred, self.y_prob, self.confusion_matrix
            )
        return self._cached_metrics[metric_name]

# --- Optimized Basic Metrics Implementation ---
# These now accept confusion matrix to avoid repeated calculation

@MetricRegistry.register('accuracy', display_name='Accuracy')
def calc_accuracy(y_true, y_pred, y_prob, cm=None):
    return accuracy_score(y_true, y_pred)

@MetricRegistry.register('sensitivity', display_name='Sensitivity')
def calc_sensitivity(y_true, y_pred, y_prob, cm=None):
    """Calculate sensitivity (recall, true positive rate)."""
    if cm is None:
        cm = metrics.confusion_matrix(y_true, y_pred)
    # Handle both binary and multi-class cases
    if cm.shape == (2, 2):
        return cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    else:
        # Multi-class: macro average
        recalls = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)
        return np.mean(recalls)

@MetricRegistry.register('specificity', display_name='Specificity')
def calc_specificity(y_true, y_pred, y_prob, cm=None):
    """Calculate specificity (true negative rate)."""
    if cm is None:
        cm = metrics.confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        return cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    else:
        # Multi-class: macro average
        specs = []
        for i in range(cm.shape[0]):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specs.append(spec)
        return np.mean(specs)

@MetricRegistry.register('ppv', display_name='PPV')
def calc_ppv(y_true, y_pred, y_prob, cm=None):
    """Calculate Positive Predictive Value (precision)."""
    if cm is None:
        cm = metrics.confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        return cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    else:
        # Multi-class: macro average precision
        return metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)

@MetricRegistry.register('npv', display_name='NPV')
def calc_npv(y_true, y_pred, y_prob, cm=None):
    """Calculate Negative Predictive Value."""
    if cm is None:
        cm = metrics.confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        return cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
    else:
        # Multi-class: calculate per-class and average
        npvs = []
        for i in range(cm.shape[0]):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fn = cm[i, :].sum() - cm[i, i]
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            npvs.append(npv)
        return np.mean(npvs)

@MetricRegistry.register('f1_score', display_name='F1-score')
def calc_f1(y_true, y_pred, y_prob, cm=None):
    """Calculate F1-score (harmonic mean of precision and recall)."""
    if cm is None:
        cm = metrics.confusion_matrix(y_true, y_pred)
    # Use cached confusion matrix for precision and recall
    precision = calc_ppv(y_true, y_pred, y_prob, cm)
    recall = calc_sensitivity(y_true, y_pred, y_prob, cm)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

@MetricRegistry.register('auc', display_name='AUC')
def calc_auc(y_true, y_pred, y_prob):
    # Support for multi-class AUC will be handled here if y_prob is 2D
    if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1:
        return roc_auc_score(y_true, y_prob, multi_class='ovr')
    return roc_auc_score(y_true, y_prob)

# --- Statistical Metrics Implementation ---

@MetricRegistry.register('hosmer_lemeshow_p_value', display_name='H-L P-value', category='statistical')
def calc_hl_p(y_true, y_pred, y_prob):
    try:
        # H-L test usually only for binary
        if y_prob.ndim > 1 and y_prob.shape[1] > 1: return np.nan
        _, p_value = hosmer_lemeshow_test(y_true, y_prob)
        return p_value
    except:
        return np.nan

@MetricRegistry.register('spiegelhalter_z_p_value', display_name='Spiegelhalter P-value', category='statistical')
def calc_spiegelhalter_p(y_true, y_pred, y_prob):
    try:
        if y_prob.ndim > 1 and y_prob.shape[1] > 1: return np.nan
        _, p_value = spiegelhalter_z_test(y_true, y_prob)
        return p_value
    except:
        return np.nan

from .prediction_container import PredictionContainer

# --- Main Interface ---

def calculate_metrics(y_true: Union[np.ndarray, PredictionContainer], 
                      y_pred: Optional[np.ndarray] = None, 
                      y_pred_proba: Optional[np.ndarray] = None,
                      categories: Optional[List[str]] = None,
                      use_cache: bool = True) -> Dict[str, float]:
    """
    Calculate all registered metrics using PredictionContainer.
    
    Args:
        y_true: True labels or PredictionContainer
        y_pred: Predicted labels (optional if using PredictionContainer)
        y_pred_proba: Predicted probabilities (optional if using PredictionContainer)
        categories: Filter metrics by category, e.g., ['basic', 'statistical']
                   If None, calculate all metrics
        use_cache: If True, use confusion matrix caching for better performance
    
    Returns:
        Dictionary of metric_name -> value
    """
    if isinstance(y_true, PredictionContainer):
        container = y_true
    else:
        container = PredictionContainer(y_true, y_pred_proba, y_pred)

    # Create cache if enabled
    cache = MetricsCache(container.y_true, container.y_pred, container.get_eval_probs()) if use_cache else None
    
    metrics_dict = {}
    for name, info in MetricRegistry.entries().items():
        # Filter by category if specified
        if categories is not None and info['category'] not in categories:
            continue
        
        try:
            if use_cache and cache is not None:
                # Use cache for metrics that benefit from it
                if name in ['sensitivity', 'specificity', 'ppv', 'npv', 'f1_score']:
                    metrics_dict[name] = cache.get_metric(
                        name,
                        lambda yt, yp, ypr, cm: info['func'](yt, yp, ypr, cm)
                    )
                else:
                    # Other metrics don't need cache
                    metrics_dict[name] = info['func'](
                        container.y_true, 
                        container.y_pred, 
                        container.get_eval_probs()
                    )
            else:
                # No cache
                metrics_dict[name] = info['func'](
                    container.y_true, 
                    container.y_pred, 
                    container.get_eval_probs()
                )
        except Exception as e:
            metrics_dict[name] = np.nan
    
    return metrics_dict

#: Metric categories that receive bootstrap confidence intervals by default.
#: Only performance metrics qualify: a confidence interval around a
#: goodness-of-fit p-value (Hosmer-Lemeshow, Spiegelhalter) has no
#: interpretation, and computing those tests 1000 times is the dominant cost.
BOOTSTRAP_METRIC_CATEGORIES: Tuple[str, ...] = ('basic',)


def bootstrap_metrics(
    container: PredictionContainer,
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    stratified: bool = True,
    random_state: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Estimate percentile bootstrap confidence intervals for registered metrics.

    The point estimate always comes from the observed sample, never from the
    mean of the bootstrap replicates: the replicate mean is a biased estimate
    of the reported performance, while the replicate percentiles are the
    interval of interest.

    Args:
        container: Predictions to resample (true labels, probabilities, labels).
        n_iterations: Number of bootstrap replicates.
        ci_level: Confidence level as a proportion, e.g. ``0.95``.
        stratified: Resample within each observed class, preserving class
            prevalence in every replicate. Required in practice for small or
            imbalanced cohorts, where unstratified replicates can contain a
            single class and leave AUC undefined.
        random_state: Seed for the replicate generator. Uses a local generator,
            so it never perturbs NumPy's global random state.
        categories: Metric categories to bootstrap. Defaults to
            :data:`BOOTSTRAP_METRIC_CATEGORIES`.

    Returns:
        Dict[str, Dict[str, float]]: Mapping of metric name to a summary with
        keys ``point``, ``mean``, ``se``, ``ci_lower``, ``ci_upper``,
        ``ci_level`` and ``n_valid`` (replicates where the metric was defined).
    """
    selected_categories = (
        list(BOOTSTRAP_METRIC_CATEGORIES) if categories is None else list(categories)
    )
    point_estimates = calculate_metrics(container, categories=selected_categories)

    y_true = np.asarray(container.y_true)
    y_prob = np.asarray(container.y_prob)
    y_pred = np.asarray(container.y_pred)

    rng = np.random.default_rng(random_state)
    index_pools = _bootstrap_index_pools(y_true, stratified=stratified)

    replicates: Dict[str, List[float]] = {name: [] for name in point_estimates}
    for _ in range(int(n_iterations)):
        indices = np.concatenate(
            [rng.choice(pool, size=pool.size, replace=True) for pool in index_pools]
        )
        replicate_container = PredictionContainer(
            y_true=y_true[indices],
            y_prob=y_prob[indices],
            y_pred=y_pred[indices],
        )
        replicate_metrics = calculate_metrics(
            replicate_container, categories=selected_categories
        )
        for name, value in replicate_metrics.items():
            if name in replicates:
                replicates[name].append(value)

    lower_percentile = (1.0 - ci_level) / 2.0 * 100.0
    upper_percentile = (1.0 + ci_level) / 2.0 * 100.0

    summary: Dict[str, Dict[str, float]] = {}
    for name, point_value in point_estimates.items():
        values = np.asarray(replicates[name], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            # The metric was undefined in every replicate; report the point
            # estimate with an explicitly empty interval rather than a fake one.
            summary[name] = {
                'point': float(point_value),
                'mean': float('nan'),
                'se': float('nan'),
                'ci_lower': float('nan'),
                'ci_upper': float('nan'),
                'ci_level': float(ci_level),
                'n_valid': 0,
            }
            continue
        summary[name] = {
            'point': float(point_value),
            'mean': float(np.mean(values)),
            'se': float(np.std(values, ddof=1)) if values.size > 1 else float('nan'),
            'ci_lower': float(np.percentile(values, lower_percentile)),
            'ci_upper': float(np.percentile(values, upper_percentile)),
            'ci_level': float(ci_level),
            'n_valid': int(values.size),
        }
    return summary


def _bootstrap_index_pools(
    y_true: np.ndarray,
    stratified: bool = True,
) -> List[np.ndarray]:
    """
    Build the index pools each bootstrap replicate is drawn from.

    Args:
        y_true: Observed labels defining the strata.
        stratified: When True, return one pool per class so a replicate keeps
            the observed class counts; when False, return a single pool over
            all samples.

    Returns:
        List[np.ndarray]: Index arrays sampled independently with replacement.
    """
    if not stratified:
        return [np.arange(len(y_true))]
    return [np.flatnonzero(y_true == label) for label in np.unique(y_true)]


def format_metric_ci(entry: Dict[str, float], digits: int = 3) -> str:
    """
    Render a bootstrap summary as a publication-style ``point (lower-upper)``.

    Args:
        entry: One value of the :func:`bootstrap_metrics` return mapping.
        digits: Number of decimals for all three numbers.

    Returns:
        str: Formatted interval, or the bare point estimate when no replicate
        produced a defined value.
    """
    point = entry.get('point', float('nan'))
    lower = entry.get('ci_lower', float('nan'))
    upper = entry.get('ci_upper', float('nan'))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} ({lower:.{digits}f}-{upper:.{digits}f})"


def apply_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Apply a given threshold to predicted probabilities and calculate metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return calculate_metrics(y_true, y_pred, y_pred_proba)

def calculate_metrics_youden(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate metrics based on the optimal Youden index.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    youden_indices = tpr + (1 - fpr) - 1
    optimal_idx = np.argmax(youden_indices)
    optimal_threshold = thresholds[optimal_idx]
    
    metrics_at_threshold = apply_threshold(y_true, y_pred_proba, optimal_threshold)
    
    return {
        'threshold': float(optimal_threshold),
        'youden_index': float(youden_indices[optimal_idx]),
        'metrics': metrics_at_threshold
    }

def apply_youden_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Apply a pre-determined Youden threshold.
    """
    metrics_at_threshold = apply_threshold(y_true, y_pred_proba, threshold)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba) # For youden index calculation if needed
    
    # Simple youden index at this specific threshold
    sens = metrics_at_threshold['sensitivity']
    spec = metrics_at_threshold['specificity']
    youden_index = sens + spec - 1
    
    return {
        'threshold': float(threshold),
        'youden_index': float(youden_index),
        'metrics': metrics_at_threshold
    }

def calculate_metrics_at_target(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               target_metrics: Dict[str, float],
                               threshold_selection: str = 'pareto+youden',
                               fallback_to_closest: bool = True,
                               distance_metric: str = 'euclidean') -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate metrics at thresholds that achieve target metric values.
    
    Enhanced version with:
    - Support for any metric (sensitivity, specificity, ppv, npv, f1_score, accuracy)
    - Fallback mechanism when no threshold satisfies all targets
    - Multiple threshold selection strategies
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        target_metrics: Target values, e.g., {'sensitivity': 0.91, 'specificity': 0.91}
        threshold_selection: Strategy for selecting best threshold from multiple candidates:
            - 'first': Use the first satisfying threshold
            - 'youden': Maximum Youden index among satisfying thresholds
            - 'pareto+youden': Pareto optimal with highest Youden (recommended)
        fallback_to_closest: If True, find closest threshold when no perfect match exists
        distance_metric: Distance metric for fallback ('euclidean', 'manhattan', 'max')
    
    Returns:
        Dictionary containing:
        - 'thresholds': Individual thresholds for each target metric
        - 'metrics_at_thresholds': Full metrics at each individual threshold
        - 'combined_results': Thresholds satisfying all targets
        - 'best_threshold': Selected best threshold (if multiple candidates exist)
        - 'closest_threshold': Fallback threshold (if no perfect match)
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Sort thresholds from highest to lowest
    sorted_indices = np.argsort(thresholds)[::-1]
    thresholds = thresholds[sorted_indices]
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    
    target_thresholds = {}
    metrics_at_thresholds = {}
    
    # Separate ROC-based metrics from others
    roc_metrics = {'sensitivity', 'specificity'}
    other_metrics = {k for k in target_metrics.keys() if k not in roc_metrics}
    
    # Find individual thresholds for each target
    for metric_name, target_value in target_metrics.items():
        best_threshold = None
        
        if metric_name == 'sensitivity':
            # Sensitivity = TPR (from ROC curve)
            for i, val in enumerate(tpr):
                if val >= target_value:
                    best_threshold = thresholds[i]
                    break
                    
        elif metric_name == 'specificity':
            # Specificity = 1 - FPR (from ROC curve)
            for i in range(len(fpr)-1, -1, -1):
                if (1 - fpr[i]) >= target_value:
                    best_threshold = thresholds[i]
                    break
        
        elif metric_name in ['ppv', 'npv', 'f1_score', 'accuracy']:
            # These metrics require full calculation at each threshold
            for thresh in thresholds:
                metrics_at_thresh = apply_threshold(y_true, y_pred_proba, thresh)
                if metrics_at_thresh.get(metric_name, 0) >= target_value:
                    best_threshold = thresh
                    break
        
        if best_threshold is not None:
            target_thresholds[metric_name] = best_threshold
            metrics_at_thresholds[metric_name] = apply_threshold(y_true, y_pred_proba, best_threshold)
    
    # Search for combined thresholds that meet all targets simultaneously
    combined_results = {}
    
    if len(target_metrics) > 0:
        # Iterate through all thresholds to find ones meeting all criteria
        for i, thresh in enumerate(thresholds):
            meets_all = True
            current_metrics_from_roc = {}
            
            # Check ROC-based targets first (fast)
            if 'sensitivity' in target_metrics:
                current_sensitivity = tpr[i]
                current_metrics_from_roc['sensitivity'] = float(current_sensitivity)
                if current_sensitivity < target_metrics['sensitivity']:
                    meets_all = False
            
            if 'specificity' in target_metrics:
                current_specificity = 1 - fpr[i]
                current_metrics_from_roc['specificity'] = float(current_specificity)
                if current_specificity < target_metrics['specificity']:
                    meets_all = False
            
            # If ROC-based checks pass, check other metrics (slower)
            if meets_all and len(other_metrics) > 0:
                full_metrics = apply_threshold(y_true, y_pred_proba, thresh)
                
                for metric_name in other_metrics:
                    if full_metrics.get(metric_name, 0) < target_metrics[metric_name]:
                        meets_all = False
                        break
            else:
                # Calculate full metrics only if ROC checks passed
                full_metrics = apply_threshold(y_true, y_pred_proba, thresh) if meets_all else None
            
            # If this threshold meets all targets, add to results
            if meets_all and full_metrics is not None:
                combined_key = ' & '.join(sorted(target_metrics.keys()))
                
                if combined_key not in combined_results:
                    combined_results[combined_key] = {}
                
                combined_results[combined_key][float(thresh)] = full_metrics
    
    # Select best threshold from candidates
    best_threshold_info = None
    if combined_results:
        best_threshold_info = _select_best_threshold(
            combined_results, 
            threshold_selection
        )
    
    # Fallback: find closest threshold if no perfect match
    closest_threshold_info = None
    if fallback_to_closest and not combined_results:
        closest_threshold_info = _find_closest_threshold_to_targets(
            y_true, y_pred_proba, thresholds, target_metrics, distance_metric
        )
    
    return {
        'thresholds': target_thresholds,
        'metrics_at_thresholds': metrics_at_thresholds,
        'combined_results': combined_results,
        'best_threshold': best_threshold_info,
        'closest_threshold': closest_threshold_info
    }


def _select_best_threshold(
    combined_results: Dict[str, Dict[float, Dict]],
    strategy: str = 'pareto+youden'
) -> Dict:
    """
    Select the best threshold from multiple candidates.
    
    Args:
        combined_results: Dictionary of thresholds and their metrics
        strategy: Selection strategy
    
    Returns:
        Dictionary with 'threshold' and 'metrics'
    """
    # Get all thresholds and metrics
    all_thresholds = {}
    for key, thresh_dict in combined_results.items():
        all_thresholds.update(thresh_dict)
    
    if not all_thresholds:
        return None
    
    if strategy == 'first':
        # Return first threshold
        first_thresh = next(iter(all_thresholds.keys()))
        return {
            'threshold': first_thresh,
            'metrics': all_thresholds[first_thresh],
            'strategy': 'first'
        }
    
    elif strategy == 'youden':
        # Maximum Youden index
        best_thresh, best_metrics = max(
            all_thresholds.items(),
            key=lambda x: x[1].get('sensitivity', 0) + x[1].get('specificity', 0) - 1
        )
        return {
            'threshold': best_thresh,
            'metrics': best_metrics,
            'youden_index': best_metrics.get('sensitivity', 0) + best_metrics.get('specificity', 0) - 1,
            'strategy': 'youden'
        }
    
    elif strategy == 'pareto+youden':
        # Find Pareto optimal thresholds, then select highest Youden
        pareto_thresholds = _find_pareto_optimal(all_thresholds)
        
        if len(pareto_thresholds) == 1:
            thresh = next(iter(pareto_thresholds.keys()))
            metrics_dict = pareto_thresholds[thresh]
            return {
                'threshold': thresh,
                'metrics': metrics_dict,
                'youden_index': metrics_dict.get('sensitivity', 0) + metrics_dict.get('specificity', 0) - 1,
                'strategy': 'pareto+youden',
                'pareto_optimal_count': 1
            }
        else:
            # Select highest Youden among Pareto optimal
            best_thresh, best_metrics = max(
                pareto_thresholds.items(),
                key=lambda x: x[1].get('sensitivity', 0) + x[1].get('specificity', 0) - 1
            )
            return {
                'threshold': best_thresh,
                'metrics': best_metrics,
                'youden_index': best_metrics.get('sensitivity', 0) + best_metrics.get('specificity', 0) - 1,
                'strategy': 'pareto+youden',
                'pareto_optimal_count': len(pareto_thresholds)
            }
    
    # Default: first
    first_thresh = next(iter(all_thresholds.keys()))
    return {
        'threshold': first_thresh,
        'metrics': all_thresholds[first_thresh],
        'strategy': 'default'
    }


def _find_pareto_optimal(thresholds_dict: Dict[float, Dict]) -> Dict[float, Dict]:
    """
    Find Pareto optimal thresholds.
    
    A threshold is Pareto optimal if no other threshold is strictly better 
    in all metrics.
    """
    pareto_optimal = {}
    
    for thresh1, metrics1 in thresholds_dict.items():
        is_dominated = False
        
        for thresh2, metrics2 in thresholds_dict.items():
            if thresh1 == thresh2:
                continue
            
            # Check if metrics2 dominates metrics1
            # (better or equal in all, strictly better in at least one)
            better_or_equal_in_all = all(
                metrics2.get(k, 0) >= metrics1.get(k, 0)
                for k in metrics1.keys()
            )
            strictly_better_in_one = any(
                metrics2.get(k, 0) > metrics1.get(k, 0)
                for k in metrics1.keys()
            )
            
            if better_or_equal_in_all and strictly_better_in_one:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_optimal[thresh1] = metrics1
    
    return pareto_optimal


def _find_closest_threshold_to_targets(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray,
    target_metrics: Dict[str, float],
    distance_metric: str = 'euclidean'
) -> Dict:
    """
    Find the threshold that gets closest to all target metrics.
    
    Used as fallback when no threshold satisfies all targets.
    """
    best_threshold = None
    best_distance = float('inf')
    best_metrics = None
    satisfied_targets = []
    
    for thresh in thresholds:
        metrics = apply_threshold(y_true, y_pred_proba, thresh)
        
        # Calculate distance to target
        distances = []
        current_satisfied = []
        
        for metric_name, target_value in target_metrics.items():
            actual_value = metrics.get(metric_name, 0)
            diff = actual_value - target_value
            
            if actual_value >= target_value:
                current_satisfied.append(metric_name)
            
            if distance_metric == 'euclidean':
                distances.append(diff ** 2)
            elif distance_metric == 'manhattan':
                distances.append(abs(diff))
            elif distance_metric == 'max':
                distances.append(abs(diff))
        
        if distance_metric == 'euclidean':
            distance = np.sqrt(sum(distances))
        elif distance_metric == 'manhattan':
            distance = sum(distances)
        elif distance_metric == 'max':
            distance = max(distances)
        
        # Update best if this is closer or satisfies more targets
        if len(current_satisfied) > len(satisfied_targets):
            # Prioritize thresholds that satisfy more targets
            best_distance = distance
            best_threshold = thresh
            best_metrics = metrics
            satisfied_targets = current_satisfied
        elif len(current_satisfied) == len(satisfied_targets) and distance < best_distance:
            # Same number of satisfied targets, choose closer one
            best_distance = distance
            best_threshold = thresh
            best_metrics = metrics
    
    if best_threshold is not None:
        return {
            'threshold': float(best_threshold),
            'metrics': best_metrics,
            'distance_to_target': float(best_distance),
            'distance_metric': distance_metric,
            'satisfied_targets': satisfied_targets,
            'unsatisfied_targets': [k for k in target_metrics.keys() if k not in satisfied_targets],
            'warning': 'No threshold satisfies all targets. This is the closest match.'
        }
    
    return None

def apply_target_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                           threshold: float) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Apply a pre-determined target threshold.
    """
    metrics_at_threshold = apply_threshold(y_true, y_pred_proba, threshold)
    return {
        'threshold': float(threshold),
        'metrics': metrics_at_threshold
    }

# --- Utility Functions (Legacy Support) ---

def calculate_net_benefit(y_true, y_pred_proba, threshold):
    """Used for DCA plotting."""
    if isinstance(threshold, list): threshold = threshold[0]
    if threshold >= 0.999: return 0.0
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    
    benefit = (true_positives / n) - (false_positives / n) * (threshold / (1 - threshold))
    return benefit if np.isfinite(benefit) else 0.0

# Add other functions like calculate_metrics_youden, calculate_metrics_at_target...
# For brevity, these are assumed to remain or be refactored similarly.
