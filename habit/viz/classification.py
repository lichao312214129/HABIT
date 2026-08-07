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
"""Binary-classification evaluation figures.

Pure functions: arrays (or a name -> (y_true, y_prob) mapping) in, a
matplotlib ``Figure`` out, no filesystem. All text is sanitised to ASCII via
:func:`~habit.viz.labels.sanitize_label`.

The panel covers the standard imaging-paper set for a binary classifier:
ROC, precision-recall, calibration, decision-curve analysis, and the
confusion matrix. SHAP summary rendering is available when the caller already
holds attribution values (computation stays outside this package).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.labels import sanitize_label

__all__ = [
    "plot_roc",
    "plot_precision_recall",
    "plot_calibration",
    "plot_decision_curve",
    "plot_confusion_matrix",
    "plot_shap_summary",
    "plot_shap_dependence",
    "plot_shap_waterfall",
    "plot_permutation_importance",
    "rank_shap_feature_indices",
    "select_representative_sample_indices",
    "net_benefit",
]

#: One named series: (y_true, y_prob) of equal length.
CurveSeries = Tuple[np.ndarray, np.ndarray]
#: One or more named series for overlay curves.
CurvePanel = Mapping[str, CurveSeries]


#: What habit.viz needs matplotlib for.
_VIZ_PURPOSE = "classification figures (ROC, calibration, SHAP, ...)"


def _plt():
    """
    Return the pyplot module with the Agg canvas guaranteed headless.

    matplotlib is an OPTIONAL dependency (habitat-analysis[viz]); it is
    imported here rather than at module scope so ``import habit.viz`` stays
    free of it, and it goes through ``require`` so a missing install names
    the extra instead of raising a bare ModuleNotFoundError.

    Returns:
        The ``matplotlib.pyplot`` module, with a non-interactive backend
        already active.

    Raises:
        OptionalDependencyError: When matplotlib is not installed.
    """
    matplotlib = require("matplotlib", extra="viz", purpose=_VIZ_PURPOSE)

    if matplotlib.get_backend().lower() not in (
        "agg",
        "module://matplotlib_inline.backend_inline",
    ):
        matplotlib.use("Agg")

    return require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)


def _as_panel(
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    *,
    curves: Optional[CurvePanel] = None,
    owner: str,
    model_name: str = "model",
) -> CurvePanel:
    """
    Normalise the single-series and multi-series call shapes.

    Args:
        y_true: Binary labels for a single series.
        y_prob: Positive-class probabilities for a single series.
        curves: Named ``(y_true, y_prob)`` mapping for overlays.
        owner: Calling function name (for error messages).
        model_name: Default series key when only arrays are given.

    Returns:
        A non-empty ``CurvePanel``.

    Raises:
        HABITAPIError: When neither form is provided or arrays are invalid.
    """
    if curves is not None:
        if not curves:
            raise HABITAPIError(f"habit.viz.{owner}: curves mapping is empty.")
        panel: dict[str, CurveSeries] = {}
        for name, pair in curves.items():
            panel[str(name)] = _check_binary_pair(pair[0], pair[1], owner)
        return panel
    if y_true is None or y_prob is None:
        raise HABITAPIError(
            f"habit.viz.{owner}: provide y_true/y_prob or a curves mapping."
        )
    return {model_name: _check_binary_pair(y_true, y_prob, owner)}


def _check_binary_pair(
    y_true: np.ndarray, y_prob: np.ndarray, owner: str
) -> CurveSeries:
    """Validate one (label, probability) series."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true_arr.shape != y_prob_arr.shape:
        raise HABITAPIError(
            f"habit.viz.{owner}: y_true and y_prob must have the same length; "
            f"got {y_true_arr.shape[0]} and {y_prob_arr.shape[0]}."
        )
    if y_true_arr.size < 2:
        raise HABITAPIError(f"habit.viz.{owner}: need at least two samples.")
    # Labels may arrive as strings ("0"/"1"); coerce to float for metrics.
    try:
        y_true_num = y_true_arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise HABITAPIError(
            f"habit.viz.{owner}: y_true must be numeric binary labels."
        ) from exc
    return y_true_num, y_prob_arr


def _palette(n: int) -> Sequence[str]:
    """Colour-blind-friendly cycle matching :class:`~habit.viz.style.StyleSpec`."""
    base = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    )
    return [base[i % len(base)] for i in range(n)]


def net_benefit(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> float:
    """
    Decision-curve net benefit at one threshold probability.

    Args:
        y_true: Binary labels (0/1).
        y_prob: Predicted positive-class probabilities in ``[0, 1]``.
        threshold: Probability threshold in ``[0, 1)``.

    Returns:
        Net benefit at ``threshold`` (0.0 when undefined).
    """
    if threshold >= 0.999:
        return 0.0
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y_hat = (y_prob_arr >= threshold).astype(int)
    n = float(len(y_true_arr))
    if n == 0.0:
        return 0.0
    tp = float(np.sum((y_hat == 1) & (y_true_arr == 1)))
    fp = float(np.sum((y_hat == 1) & (y_true_arr == 0)))
    benefit = (tp / n) - (fp / n) * (threshold / (1.0 - threshold))
    return float(benefit) if np.isfinite(benefit) else 0.0


def plot_roc(
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    *,
    curves: Optional[CurvePanel] = None,
    model_name: str = "model",
    title: str = "ROC",
):
    """
    Receiver-operating-characteristic curve(s) with AUC in the legend.

    Args:
        y_true: Binary labels for a single series.
        y_prob: Positive-class probabilities for a single series.
        curves: Optional multi-model overlay panel.
        model_name: Legend name when using the single-series form.
        title: Figure title (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    from sklearn.metrics import auc, roc_curve

    plt = _plt()
    panel = _as_panel(
        y_true, y_prob, curves=curves, owner="plot_roc", model_name=model_name
    )
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    colors = _palette(len(panel))
    for (name, (yt, yp)), color in zip(panel.items(), colors):
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=1.8,
            label=f"{sanitize_label(name)} (AUC = {auc(fpr, tpr):.2f})",
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", lw=1.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(sanitize_label(title))
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_precision_recall(
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    *,
    curves: Optional[CurvePanel] = None,
    model_name: str = "model",
    title: str = "Precision-Recall",
):
    """
    Precision-recall curve(s) with average precision in the legend.

    Args:
        y_true: Binary labels for a single series.
        y_prob: Positive-class probabilities for a single series.
        curves: Optional multi-model overlay panel.
        model_name: Legend name when using the single-series form.
        title: Figure title (sanitised).

    Returns:
        The matplotlib ``Figure``.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    plt = _plt()
    panel = _as_panel(
        y_true,
        y_prob,
        curves=curves,
        owner="plot_precision_recall",
        model_name=model_name,
    )
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    colors = _palette(len(panel))
    for (name, (yt, yp)), color in zip(panel.items(), colors):
        precision, recall, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ax.plot(
            recall,
            precision,
            color=color,
            lw=1.8,
            label=f"{sanitize_label(name)} (AP = {ap:.2f})",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(sanitize_label(title))
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_calibration(
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    *,
    curves: Optional[CurvePanel] = None,
    model_name: str = "model",
    title: str = "Calibration",
    n_bins: int = 10,
):
    """
    Reliability diagram (calibration curve) against the identity line.

    Args:
        y_true: Binary labels for a single series.
        y_prob: Positive-class probabilities for a single series.
        curves: Optional multi-model overlay panel.
        model_name: Legend name when using the single-series form.
        title: Figure title (sanitised).
        n_bins: Number of probability bins.

    Returns:
        The matplotlib ``Figure``.
    """
    from sklearn.calibration import calibration_curve

    plt = _plt()
    panel = _as_panel(
        y_true,
        y_prob,
        curves=curves,
        owner="plot_calibration",
        model_name=model_name,
    )
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", lw=1.2, label="Perfectly calibrated")
    colors = _palette(len(panel))
    for (name, (yt, yp)), color in zip(panel.items(), colors):
        frac_pos, mean_pred = calibration_curve(
            yt, yp, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            color=color,
            lw=1.8,
            label=sanitize_label(name),
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(sanitize_label(title))
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_decision_curve(
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    *,
    curves: Optional[CurvePanel] = None,
    model_name: str = "model",
    title: str = "Decision Curve",
    n_thresholds: int = 100,
):
    """
    Decision-curve analysis with Treat-All / Treat-None references.

    Args:
        y_true: Binary labels for a single series.
        y_prob: Positive-class probabilities for a single series.
        curves: Optional multi-model overlay panel.
        model_name: Legend name when using the single-series form.
        title: Figure title (sanitised).
        n_thresholds: Number of threshold points on ``[0, 1]``.

    Returns:
        The matplotlib ``Figure``.
    """
    plt = _plt()
    panel = _as_panel(
        y_true,
        y_prob,
        curves=curves,
        owner="plot_decision_curve",
        model_name=model_name,
    )
    thresholds = np.linspace(0.0, 1.0, int(n_thresholds))
    # Use the first series' labels for the reference strategies.
    ref_y = next(iter(panel.values()))[0]
    treat_all = np.asarray(
        [net_benefit(ref_y, np.ones_like(ref_y), t) for t in thresholds]
    )
    treat_none = np.asarray(
        [net_benefit(ref_y, np.zeros_like(ref_y), t) for t in thresholds]
    )

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(thresholds, treat_all, "k--", lw=1.5, label="Treat All")
    ax.plot(thresholds, treat_none, "k-", lw=1.5, label="Treat None")
    colors = _palette(len(panel))
    for (name, (yt, yp)), color in zip(panel.items(), colors):
        nb = np.asarray([net_benefit(yt, yp, t) for t in thresholds])
        ax.plot(thresholds, nb, color=color, lw=1.8, label=sanitize_label(name))

    y_min = -0.05
    y_max = 0.5
    finite_none = treat_none[np.isfinite(treat_none)]
    finite_all = treat_all[np.isfinite(treat_all)]
    if finite_none.size:
        y_min = min(y_min, float(np.min(finite_none)))
    if finite_all.size:
        y_max = max(y_max, float(np.max(finite_all)) + 0.1)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(sanitize_label(title))
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Confusion Matrix",
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
):
    """
    Confusion-matrix heatmap for discrete class predictions.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels (same length as ``y_true``).
        title: Figure title (sanitised).
        class_names: Optional tick labels; defaults to sorted unique labels.
        normalize: When True, row-normalise to proportions.

    Returns:
        The matplotlib ``Figure``.
    """
    from sklearn.metrics import confusion_matrix

    plt = _plt()
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = np.asarray(y_pred).reshape(-1)
    if y_true_arr.shape != y_pred_arr.shape:
        raise HABITAPIError(
            "habit.viz.plot_confusion_matrix: y_true and y_pred must have "
            f"the same length; got {y_true_arr.shape[0]} and {y_pred_arr.shape[0]}."
        )
    if y_true_arr.size < 1:
        raise HABITAPIError(
            "habit.viz.plot_confusion_matrix: need at least one sample."
        )
    unique_labels = sorted(
        np.unique(np.concatenate([y_true_arr, y_pred_arr])), key=str
    )
    cm_labels = list(class_names) if class_names is not None else list(unique_labels)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=cm_labels)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
        row_sum[row_sum == 0.0] = 1.0
        cm_display = cm.astype(np.float64) / row_sum
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_labels = [sanitize_label(name) for name in cm_labels]
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(sanitize_label(title))
    thresh = float(np.max(cm_display)) / 2.0 if cm_display.size else 0.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if float(value) > thresh else "black",
            )
    fig.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    features: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "SHAP summary",
    max_display: int = 20,
):
    """
    Beeswarm-style SHAP summary for one attribution matrix.

    Args:
        shap_values: Attribution matrix of shape ``(n_samples, n_features)``.
        features: Feature matrix aligned with ``shap_values``.
        feature_names: Optional column names (ASCII-sanitised on draw).
        title: Figure title (sanitised).
        max_display: Maximum number of features to show.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        HABITAPIError: When shapes disagree or ``shap`` is not installed.
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        from habit.exceptions import OptionalDependencyError

        raise OptionalDependencyError(
            "plot_shap_summary requires the optional 'shap' package. "
            'Install it with: pip install "habitat-analysis[explain]" '
            "(or pip install shap)."
        ) from exc

    values = np.asarray(shap_values, dtype=np.float64)
    feats = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or feats.ndim != 2:
        raise HABITAPIError(
            "habit.viz.plot_shap_summary: shap_values and features must be 2-D."
        )
    if values.shape != feats.shape:
        raise HABITAPIError(
            "habit.viz.plot_shap_summary: shap_values and features must share "
            f"shape; got {values.shape} and {feats.shape}."
        )
    names = (
        [sanitize_label(name) for name in feature_names]
        if feature_names is not None
        else [f"f{i}" for i in range(values.shape[1])]
    )
    plt = _plt()
    # shap.summary_plot draws on the current axes; capture that figure.
    shap.summary_plot(
        values,
        feats,
        feature_names=names,
        max_display=int(max_display),
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(sanitize_label(title))
    fig.tight_layout()
    return fig


def rank_shap_feature_indices(
    shap_values: np.ndarray,
    *,
    top_k: int = 3,
) -> np.ndarray:
    """
    Rank features by mean absolute SHAP attribution (descending).

    Args:
        shap_values: Attribution matrix ``(n_samples, n_features)``.
        top_k: Number of leading indices to return.

    Returns:
        Integer index array of length ``min(top_k, n_features)``.
    """
    values = np.asarray(shap_values, dtype=np.float64)
    if values.ndim != 2:
        raise HABITAPIError(
            "habit.viz.rank_shap_feature_indices: shap_values must be 2-D."
        )
    k = max(min(int(top_k), values.shape[1]), 0)
    if k == 0:
        return np.asarray([], dtype=int)
    mean_abs = np.abs(values).mean(axis=0)
    return np.argsort(mean_abs)[::-1][:k]


def select_representative_sample_indices(
    scores: np.ndarray,
    *,
    n_samples: int = 3,
) -> list[int]:
    """
    Pick sample indices spanning the score range (low / mid / high).

    Args:
        scores: Per-sample scalar used for ranking (e.g. sum of SHAP values).
        n_samples: Number of indices to return.

    Returns:
        Distinct indices evenly spaced over the ranked scores.
    """
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    total = int(arr.size)
    n = max(min(int(n_samples), total), 0)
    if n == 0:
        return []
    order = np.argsort(arr)
    if n == 1:
        return [int(order[total // 2])]
    positions = np.linspace(0, total - 1, n).round().astype(int)
    return [int(order[position]) for position in positions]


def plot_shap_dependence(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_index: int,
    *,
    feature_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
):
    """
    SHAP dependence plot for one feature (pure; no filesystem).

    Args:
        shap_values: Attribution matrix ``(n_samples, n_features)``.
        features: Feature matrix aligned with ``shap_values``.
        feature_index: Column index to explain.
        feature_names: Optional column names.
        title: Optional figure title; defaults to the feature name.

    Returns:
        The matplotlib ``Figure``.
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        from habit.exceptions import OptionalDependencyError

        raise OptionalDependencyError(
            "plot_shap_dependence requires the optional 'shap' package. "
            'Install it with: pip install "habitat-analysis[explain]" '
            "(or pip install shap)."
        ) from exc

    values = np.asarray(shap_values, dtype=np.float64)
    feats = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or feats.ndim != 2 or values.shape != feats.shape:
        raise HABITAPIError(
            "habit.viz.plot_shap_dependence: shap_values and features must "
            "be 2-D arrays of equal shape."
        )
    index = int(feature_index)
    if index < 0 or index >= values.shape[1]:
        raise HABITAPIError(
            f"habit.viz.plot_shap_dependence: feature_index {index} out of "
            f"range for {values.shape[1]} features."
        )
    names = (
        [sanitize_label(name) for name in feature_names]
        if feature_names is not None
        else [f"f{i}" for i in range(values.shape[1])]
    )
    plt = _plt()
    plt.figure(figsize=(5.0, 4.0))
    shap.dependence_plot(
        index,
        values,
        feats,
        feature_names=names,
        interaction_index="auto",
        show=False,
    )
    fig = plt.gcf()
    resolved_title = title if title is not None else f"SHAP Dependence: {names[index]}"
    fig.axes[0].set_title(sanitize_label(resolved_title))
    fig.tight_layout()
    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    features: np.ndarray,
    sample_index: int,
    *,
    feature_names: Optional[Sequence[str]] = None,
    base_value: float = 0.0,
    title: Optional[str] = None,
):
    """
    Per-sample SHAP waterfall explanation (pure; no filesystem).

    Args:
        shap_values: Attribution matrix ``(n_samples, n_features)``.
        features: Feature matrix aligned with ``shap_values``.
        sample_index: Row to explain.
        feature_names: Optional column names.
        base_value: Explainer base value (positive class).
        title: Optional figure title.

    Returns:
        The matplotlib ``Figure``.
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        from habit.exceptions import OptionalDependencyError

        raise OptionalDependencyError(
            "plot_shap_waterfall requires the optional 'shap' package. "
            'Install it with: pip install "habitat-analysis[explain]" '
            "(or pip install shap)."
        ) from exc

    values = np.asarray(shap_values, dtype=np.float64)
    feats = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or feats.ndim != 2 or values.shape != feats.shape:
        raise HABITAPIError(
            "habit.viz.plot_shap_waterfall: shap_values and features must "
            "be 2-D arrays of equal shape."
        )
    row = int(sample_index)
    if row < 0 or row >= values.shape[0]:
        raise HABITAPIError(
            f"habit.viz.plot_shap_waterfall: sample_index {row} out of "
            f"range for {values.shape[0]} samples."
        )
    names = (
        [sanitize_label(name) for name in feature_names]
        if feature_names is not None
        else [f"f{i}" for i in range(values.shape[1])]
    )
    explanation = shap.Explanation(
        values=values[row],
        base_values=float(base_value),
        data=feats[row],
        feature_names=list(names),
    )
    plt = _plt()
    plt.figure(figsize=(6.0, 5.0))
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    resolved = title if title is not None else f"SHAP Explanation: Sample {row}"
    fig.suptitle(sanitize_label(resolved))
    fig.tight_layout()
    return fig


def plot_permutation_importance(
    feature_names: Sequence[str],
    importance_mean: np.ndarray,
    *,
    importance_std: Optional[np.ndarray] = None,
    title: str = "Permutation Importance",
    top_k: int = 20,
):
    """
    Horizontal bar chart of permutation importances (pure; no filesystem).

    Args:
        feature_names: Feature labels aligned with the importance vectors.
        importance_mean: Mean importance per feature.
        importance_std: Optional standard deviation for error bars.
        title: Figure title (sanitised).
        top_k: Maximum number of features to display.

    Returns:
        The matplotlib ``Figure``.
    """
    names = [sanitize_label(name) for name in feature_names]
    means = np.asarray(importance_mean, dtype=np.float64).reshape(-1)
    if means.size != len(names):
        raise HABITAPIError(
            "habit.viz.plot_permutation_importance: feature_names and "
            "importance_mean must have the same length."
        )
    if means.size == 0:
        raise HABITAPIError(
            "habit.viz.plot_permutation_importance: need at least one feature."
        )
    stds = (
        np.asarray(importance_std, dtype=np.float64).reshape(-1)
        if importance_std is not None
        else np.zeros_like(means)
    )
    if stds.size != means.size:
        raise HABITAPIError(
            "habit.viz.plot_permutation_importance: importance_std length "
            "must match importance_mean."
        )
    order = np.argsort(means)[::-1][: max(int(top_k), 1)]
    # Reverse for barh so the strongest feature is at the top.
    order = order[::-1]
    plt = _plt()
    height = max(3.0, 0.28 * len(order) + 1.2)
    fig, ax = plt.subplots(figsize=(6.0, height))
    ax.barh(
        [names[i] for i in order],
        means[order],
        xerr=stds[order],
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.5,
        error_kw={"ecolor": "#444444", "elinewidth": 0.8, "capsize": 2},
    )
    ax.set_xlabel("Importance (score decrease)")
    ax.set_title(sanitize_label(title))
    fig.tight_layout()
    return fig
