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
"""In-memory multi-model comparison (merge, splits, metrics, DeLong).

Arrays / DataFrames in, nested dicts out. No filesystem. The L4 compare
recipe loads prediction CSVs, calls these helpers, then writes plots and
JSON via ``habit.viz`` / adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.evaluation.panel import (
    clean_binary_predictions,
    compute_classification_metrics,
)
from habit.evaluation.statistics import auc_confidence_interval, delong_test
from habit.evaluation.thresholds import (
    apply_target_threshold,
    apply_youden_threshold,
    target_threshold_metrics,
    youden_threshold_metrics,
)
from habit.exceptions import HABITAPIError

__all__ = [
    "PredictionSource",
    "MergedPredictions",
    "ModelArrays",
    "ComparisonResult",
    "merge_prediction_frames",
    "model_arrays_from_frame",
    "split_model_arrays",
    "resolve_training_group_name",
    "pairwise_delong_report",
    "compute_basic_metrics_bundle",
    "compute_youden_metrics_bundle",
    "compute_target_metrics_bundle",
    "evaluate_comparison",
]

#: (y_true, y_prob, optional y_pred)
ModelArrays = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]


@dataclass(frozen=True)
class PredictionSource:
    """Column mapping for one prediction table already loaded in memory."""

    model_name: str
    subject_id_col: str
    label_col: str
    prob_col: str
    pred_col: Optional[str] = None
    split_col: Optional[str] = None


@dataclass(frozen=True)
class MergedPredictions:
    """Subject-aligned multi-model prediction table."""

    #: Columns: ``subject_id``, ``label``, ``{model}_prob``, optional
    #: ``{model}_pred``, optional split column.
    frame: pd.DataFrame
    model_names: Tuple[str, ...]
    split_column: Optional[str]


@dataclass(frozen=True)
class ComparisonResult:
    """In-memory outcome of a multi-model comparison evaluation.

    Pure data: the L4 recipe / reporting helpers persist CSVs, figures, and
    JSON from this object. Notebooks can consume it without touching disk.
    """

    #: Merged multi-model prediction table.
    merged: MergedPredictions
    #: Split group (or ``"all"``) -> model -> prediction arrays.
    groups: Mapping[Any, Mapping[str, ModelArrays]]
    #: Nested metrics store (group -> model -> metric family).
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]
    #: DeLong pairwise reports keyed by group name (empty when disabled).
    delong_by_group: Mapping[str, Tuple[Mapping[str, Any], ...]]
    #: Resolved training group label used for threshold transfer, if any.
    training_group: Optional[Any]


def merge_prediction_frames(
    sources: Sequence[Tuple[PredictionSource, pd.DataFrame]],
) -> MergedPredictions:
    """
    Outer-join prediction tables on ``(subject_id, label)``.

    Args:
        sources: Ordered ``(spec, frame)`` pairs. Duplicate model names are
            disambiguated with ``_2``, ``_3``, ...

    Returns:
        Merged table plus resolved model / split column names.

    Raises:
        HABITAPIError: When ``sources`` is empty or a required column is missing.
    """
    if not sources:
        raise HABITAPIError("merge_prediction_frames requires at least one source.")

    used_names: set[str] = set()
    standardized: List[Tuple[str, pd.DataFrame]] = []
    # Collect subject_id -> split value from every source that declares one.
    split_by_subject: Dict[str, Any] = {}
    split_column: Optional[str] = None
    for spec, raw in sources:
        model_name = _unique_model_name(spec.model_name, used_names)
        for col in (spec.subject_id_col, spec.label_col, spec.prob_col):
            if col not in raw.columns:
                raise HABITAPIError(
                    f"merge_prediction_frames: column {col!r} missing for "
                    f"model {model_name!r}."
                )
        subject_ids = raw[spec.subject_id_col].astype(str)
        piece = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "label": raw[spec.label_col],
                f"{model_name}_prob": raw[spec.prob_col],
            }
        )
        if spec.pred_col and spec.pred_col in raw.columns:
            piece[f"{model_name}_pred"] = raw[spec.pred_col]
        if spec.split_col and spec.split_col in raw.columns:
            if split_column is None:
                split_column = spec.split_col
            for subj_id, split_val in zip(subject_ids, raw[spec.split_col]):
                # Canonical train label used by the v0.1 comparison workflow.
                normalized = (
                    "train"
                    if str(split_val).strip().lower() == "train"
                    else split_val
                )
                key = str(subj_id)
                if key in split_by_subject and split_by_subject[key] != normalized:
                    raise HABITAPIError(
                        f"merge_prediction_frames: subject {key!r} has "
                        f"conflicting split values "
                        f"({split_by_subject[key]!r} vs {normalized!r})."
                    )
                split_by_subject[key] = normalized
        standardized.append((model_name, piece))

    merged: Optional[pd.DataFrame] = None
    model_names: List[str] = []
    for model_name, piece in standardized:
        model_names.append(model_name)
        if merged is None:
            merged = piece.copy()
        else:
            keep = ["subject_id", "label"] + [
                c for c in piece.columns if c.endswith("_prob") or c.endswith("_pred")
            ]
            merged = merged.merge(piece[keep], on=["subject_id", "label"], how="outer")

    assert merged is not None
    if split_column is not None and split_by_subject:
        merged[split_column] = merged["subject_id"].map(split_by_subject)

    return MergedPredictions(
        frame=merged,
        model_names=tuple(model_names),
        split_column=split_column,
    )


def model_arrays_from_frame(
    frame: pd.DataFrame,
    model_names: Sequence[str],
) -> Dict[str, ModelArrays]:
    """
    Extract ``(y_true, y_prob, y_pred|None)`` per model from a merged frame.

    Args:
        frame: Merged prediction table with a ``label`` column.
        model_names: Model names whose ``{name}_prob`` columns to read.

    Returns:
        Mapping model name -> prediction arrays (row-aligned to ``frame``).
    """
    if "label" not in frame.columns:
        raise HABITAPIError("model_arrays_from_frame: frame lacks a 'label' column.")
    y_true = frame["label"].to_numpy()
    out: Dict[str, ModelArrays] = {}
    for name in model_names:
        prob_col = f"{name}_prob"
        if prob_col not in frame.columns:
            continue
        y_prob = frame[prob_col].to_numpy()
        pred_col = f"{name}_pred"
        y_pred = frame[pred_col].to_numpy() if pred_col in frame.columns else None
        out[name] = (y_true, y_prob, y_pred)
    return out


def split_model_arrays(
    frame: pd.DataFrame,
    model_names: Sequence[str],
    split_column: str,
) -> Dict[Any, Dict[str, ModelArrays]]:
    """
    Partition model arrays by the values of ``split_column``.

    Args:
        frame: Merged prediction table.
        model_names: Models to extract.
        split_column: Column holding train/test (or other) group labels.

    Returns:
        Mapping group label -> model -> arrays.
    """
    if split_column not in frame.columns:
        raise HABITAPIError(
            f"split_model_arrays: split column {split_column!r} not in frame."
        )
    groups: Dict[Any, Dict[str, ModelArrays]] = {}
    for group_name, group_df in frame.groupby(split_column, dropna=True):
        groups[group_name] = model_arrays_from_frame(group_df, model_names)
    return groups


def resolve_training_group_name(group_names: Iterable[Any]) -> Optional[Any]:
    """
    Resolve the training split label from available group names.

    Accepts common HABIT / AutoGluon exports (``train``, ``Training set``,
    ``training_set``, ...). Matching is case-insensitive.

    Args:
        group_names: Split group labels present in the merged table.

    Returns:
        The original group name that represents training, or ``None``.
    """
    names = list(group_names)
    if not names:
        return None
    preferred = {
        "train",
        "training",
        "training set",
        "train set",
        "trainset",
        "trainingset",
    }
    for name in names:
        if _normalize_split_label(name) in preferred:
            return name
    for name in names:
        normalized = _normalize_split_label(name)
        if normalized.startswith("train") and "test" not in normalized:
            return name
    return None


def pairwise_delong_report(
    models_data: Mapping[str, ModelArrays],
) -> List[Dict[str, Any]]:
    """
    Pairwise DeLong AUC comparisons with CI, p-value, and conclusion.

    Rows with NaN labels/probs for either model are dropped before testing,
    matching the v0.1 multifile evaluator.

    Args:
        models_data: Model name -> ``(y_true, y_prob, y_pred?)``.

    Returns:
        List of comparison dicts ready for ``delong_results.json``.
    """
    names = list(models_data.keys())
    results: List[Dict[str, Any]] = []
    if len(names) < 2:
        return results

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            model_a = names[i]
            model_b = names[j]
            y_true_a, y_prob_a, _ = models_data[model_a]
            y_true_b, y_prob_b, _ = models_data[model_b]
            # Prefer a shared label vector; fall back to model_a's labels.
            y_true = np.asarray(y_true_a, dtype=np.float64).reshape(-1)
            if np.asarray(y_true_b).shape == y_true.shape:
                # Keep finite rows where both models have scores.
                pass
            ya = np.asarray(y_prob_a, dtype=np.float64).reshape(-1)
            yb = np.asarray(y_prob_b, dtype=np.float64).reshape(-1)
            if y_true.shape[0] != ya.shape[0] or y_true.shape[0] != yb.shape[0]:
                raise HABITAPIError(
                    f"pairwise_delong_report: length mismatch for "
                    f"{model_a!r} vs {model_b!r}."
                )
            mask = np.isfinite(y_true) & np.isfinite(ya) & np.isfinite(yb)
            yt = y_true[mask]
            sa = ya[mask]
            sb = yb[mask]
            if yt.size == 0 or np.unique(yt).size < 2:
                continue
            delong = delong_test(yt, sa, sb)
            ci_a = auc_confidence_interval(yt, sa)
            ci_b = auc_confidence_interval(yt, sb)
            significant = bool(delong.p_value < 0.05)
            if significant:
                conclusion = (
                    f"{model_a} and {model_b} have significantly different "
                    "AUCs (p<0.05)"
                )
            else:
                conclusion = (
                    f"{model_a} and {model_b} do not have significantly "
                    "different AUCs (p≥0.05)"
                )
            results.append(
                {
                    "comparison": f"{model_a} vs {model_b}",
                    f"{model_a}_auc": float(ci_a.auc),
                    f"{model_a}_ci_lower": float(ci_a.lower),
                    f"{model_a}_ci_upper": float(ci_a.upper),
                    f"{model_b}_auc": float(ci_b.auc),
                    f"{model_b}_ci_lower": float(ci_b.lower),
                    f"{model_b}_ci_upper": float(ci_b.upper),
                    "p_value": float(delong.p_value),
                    "significant_difference": significant,
                    "conclusion": conclusion,
                }
            )
    return results


def compute_basic_metrics_bundle(
    groups: Mapping[Any, Mapping[str, ModelArrays]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute basic metric panels for every group / model.

    Args:
        groups: Split group (or ``{"all": ...}``) -> model -> arrays.

    Returns:
        Nested metrics store fragment keyed by group / model /
        ``basic_metrics``.
    """
    store: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for group_name, models in groups.items():
        for model_name, (y_true, y_prob, y_pred) in models.items():
            cleaned = clean_binary_predictions(y_true, y_prob, y_pred)
            if cleaned.y_true.size == 0:
                continue
            panel = compute_classification_metrics(
                cleaned.y_true, cleaned.y_pred, cleaned.y_prob
            )
            _ensure_model(store, str(group_name), model_name)["basic_metrics"] = panel
    return store


def compute_youden_metrics_bundle(
    groups: Mapping[Any, Mapping[str, ModelArrays]],
    *,
    training_group: Optional[Any] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Youden metrics with train-threshold transfer when a training group exists.

    Args:
        groups: Split groups (or a single ``all`` group).
        training_group: Group used to fix the threshold; when ``None`` and
            multiple groups are present, resolved automatically. When the
            table is unsplit, thresholds are estimated per group independently.

    Returns:
        Nested metrics store fragment including ``youden_metrics`` and
        ``thresholds.youden``.
    """
    store: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if len(groups) == 1 and next(iter(groups.keys())) == "all":
        models = groups["all"]
        for model_name, (y_true, y_prob, _) in models.items():
            report = youden_threshold_metrics(y_true, y_prob)
            model_store = _ensure_model(store, "all", model_name)
            model_store["youden_metrics"] = report
            model_store.setdefault("thresholds", {})["youden"] = report["threshold"]
        return store

    train_name = training_group
    if train_name is None:
        train_name = resolve_training_group_name(groups.keys())
    if train_name is None or train_name not in groups:
        return store

    train_thresholds: Dict[str, float] = {}
    for model_name, (y_true, y_prob, _) in groups[train_name].items():
        report = youden_threshold_metrics(y_true, y_prob)
        threshold = float(report["threshold"])
        train_thresholds[model_name] = threshold
        model_store = _ensure_model(store, str(train_name), model_name)
        model_store["youden_metrics"] = report
        model_store.setdefault("thresholds", {})["youden"] = threshold

    for group_name, models in groups.items():
        for model_name, (y_true, y_prob, _) in models.items():
            if model_name not in train_thresholds:
                continue
            threshold = train_thresholds[model_name]
            applied = apply_youden_threshold(y_true, y_prob, threshold)
            model_store = _ensure_model(store, str(group_name), model_name)
            model_store["youden_metrics"] = applied
            model_store.setdefault("thresholds", {})["youden"] = threshold
    return store


def compute_target_metrics_bundle(
    groups: Mapping[Any, Mapping[str, ModelArrays]],
    targets: Mapping[str, float],
    *,
    training_group: Optional[Any] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Target operating-point metrics with train-threshold transfer.

    Args:
        groups: Split groups (or a single ``all`` group).
        targets: Metric name -> minimum required value.
        training_group: Optional explicit training group name.

    Returns:
        Nested metrics store fragment including ``target_metrics`` and
        ``thresholds.target`` (when a threshold could be fixed).
    """
    store: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not targets:
        return store

    if len(groups) == 1 and next(iter(groups.keys())) == "all":
        models = groups["all"]
        for model_name, (y_true, y_prob, _) in models.items():
            report = target_threshold_metrics(y_true, y_prob, targets)
            model_store = _ensure_model(store, "all", model_name)
            model_store["target_metrics"] = report
            threshold = _extract_target_threshold(report)
            if threshold is not None:
                model_store.setdefault("thresholds", {})["target"] = threshold
        return store

    train_name = training_group
    if train_name is None:
        train_name = resolve_training_group_name(groups.keys())
    if train_name is None or train_name not in groups:
        return store

    train_thresholds: Dict[str, float] = {}
    for model_name, (y_true, y_prob, _) in groups[train_name].items():
        report = target_threshold_metrics(y_true, y_prob, targets)
        model_store = _ensure_model(store, str(train_name), model_name)
        model_store["target_metrics"] = report
        threshold = _extract_target_threshold(report)
        if threshold is not None:
            train_thresholds[model_name] = threshold
            model_store.setdefault("thresholds", {})["target"] = threshold

    for group_name, models in groups.items():
        if group_name == train_name:
            continue
        for model_name, (y_true, y_prob, _) in models.items():
            if model_name not in train_thresholds:
                # No transferable threshold: compute independently on this split.
                report = target_threshold_metrics(y_true, y_prob, targets)
                _ensure_model(store, str(group_name), model_name)[
                    "target_metrics"
                ] = report
                continue
            threshold = train_thresholds[model_name]
            applied = apply_target_threshold(y_true, y_prob, threshold)
            model_store = _ensure_model(store, str(group_name), model_name)
            model_store["target_metrics"] = applied
            model_store.setdefault("thresholds", {})["target"] = threshold
    return store


def evaluate_comparison(
    merged: MergedPredictions,
    *,
    split_enabled: bool = False,
    basic_metrics: bool = True,
    youden_metrics: bool = True,
    target_metrics: bool = False,
    targets: Optional[Mapping[str, float]] = None,
    delong_test: bool = True,
) -> ComparisonResult:
    """
    Run the in-memory comparison evaluation for one merged prediction table.

    Args:
        merged: Output of :func:`merge_prediction_frames`.
        split_enabled: When True and a split column exists, evaluate per group.
        basic_metrics: Compute the standard classification panel.
        youden_metrics: Compute Youden metrics (train threshold -> other splits).
        target_metrics: Compute target operating-point metrics.
        targets: Target metric floors used when ``target_metrics`` is True.
        delong_test: Compute pairwise DeLong reports per group.

    Returns:
        Frozen :class:`ComparisonResult` ready for L4 persistence / notebooks.
    """
    if split_enabled and merged.split_column:
        groups: Dict[Any, Dict[str, ModelArrays]] = split_model_arrays(
            merged.frame, merged.model_names, merged.split_column
        )
        if not groups:
            groups = {
                "all": model_arrays_from_frame(merged.frame, merged.model_names)
            }
    else:
        groups = {"all": model_arrays_from_frame(merged.frame, merged.model_names)}

    train_name = resolve_training_group_name(groups.keys())
    metrics_store: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if basic_metrics:
        _merge_metric_store(metrics_store, compute_basic_metrics_bundle(groups))
    if youden_metrics:
        _merge_metric_store(
            metrics_store,
            compute_youden_metrics_bundle(groups, training_group=train_name),
        )
    if target_metrics:
        _merge_metric_store(
            metrics_store,
            compute_target_metrics_bundle(
                groups, dict(targets or {}), training_group=train_name
            ),
        )

    delong_by_group: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
    if delong_test:
        for group_name, models_data in groups.items():
            if len(models_data) < 2:
                continue
            rows = pairwise_delong_report(models_data)
            delong_by_group[str(group_name)] = tuple(rows)

    return ComparisonResult(
        merged=merged,
        groups=groups,
        metrics=metrics_store,
        delong_by_group=delong_by_group,
        training_group=train_name,
    )


def _merge_metric_store(
    base: MutableMapping[str, Dict[str, Dict[str, Any]]],
    incoming: Mapping[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Merge nested metrics fragments into ``base`` in place."""
    for group_name, models in incoming.items():
        group = base.setdefault(group_name, {})
        for model_name, payload in models.items():
            cell = group.setdefault(model_name, {})
            for key, value in payload.items():
                if key == "thresholds" and isinstance(value, Mapping):
                    cell.setdefault("thresholds", {}).update(value)
                else:
                    cell[key] = value


def _extract_target_threshold(report: Mapping[str, Any]) -> Optional[float]:
    """Prefer ``best_threshold``, then ``closest_threshold``."""
    best = report.get("best_threshold")
    if isinstance(best, Mapping) and best.get("threshold") is not None:
        return float(best["threshold"])
    closest = report.get("closest_threshold")
    if isinstance(closest, Mapping) and closest.get("threshold") is not None:
        return float(closest["threshold"])
    return None


def _ensure_model(
    store: MutableMapping[str, Dict[str, Dict[str, Any]]],
    group_name: str,
    model_name: str,
) -> Dict[str, Any]:
    """Return the mutable metrics dict for one group/model cell."""
    group = store.setdefault(group_name, {})
    return group.setdefault(model_name, {})


def _unique_model_name(model_name: str, used: set[str]) -> str:
    """Disambiguate colliding model names (``name``, ``name_2``, ...)."""
    base = str(model_name).strip() or "model"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _normalize_split_label(label: Any) -> str:
    """Lowercase / collapse whitespace for robust train-group matching."""
    normalized = str(label).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())
