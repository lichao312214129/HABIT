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
"""ML figure reporting helpers at the recipe / CLI boundary (L4).

Pure plotters live in :mod:`habit.viz.classification`. This module owns the
filesystem side-effect: it turns prediction arrays into published figures
under a caller-chosen directory and returns the written paths. CLI commands
and YAML runners call these helpers after :func:`train_model` /
:func:`cross_validate`; nothing here reads YAML.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.contracts.table import FeatureTable
from habit.exceptions import HABITAPIError, OptionalDependencyError
from habit.utils.optional_deps import require
from habit.recipes.modeling import (
    CVResult,
    ModelResult,
    predict_model,
    _row_ids,
    _select_rows,
)

__all__ = [
    "write_classification_figures",
    "write_holdout_model_figures",
    "write_cv_model_figures",
    "write_ml_figures_from_config",
    "visualization_enabled",
]

#: matplotlib is an OPTIONAL dependency (habitat-analysis[viz]) and this whole
#: module is figure reporting, but it is imported from the modelling recipes,
#: so the gates stay inside the functions that actually draw.
_VIZ_PURPOSE = "machine-learning report figures"

_CURVE_TYPES = frozenset({"roc", "dca", "calibration", "pr"})
_LABEL_TYPES = frozenset({"confusion"})
_SHAP_TYPES = frozenset({"shap", "shap_dependence", "shap_waterfall"})
_KNOWN_TYPES = _CURVE_TYPES | _LABEL_TYPES | _SHAP_TYPES | frozenset({"permutation"})


def write_ml_figures_from_config(
    result: Union[ModelResult, CVResult],
    table: FeatureTable,
    config: Any,
    *,
    destination: Union[str, Path],
    mode: str,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """
    Config-driven figure write used by ``habit model`` / ``habit cv``.

    Reads ``is_visualize`` and ``visualization`` from a validated MLConfig
    (or compatible namespace). Does not import CLI modules.

    Args:
        result: Train or CV recipe outcome.
        table: Feature table used for scoring / OOF reconstruction.
        config: Object with ``is_visualize``, ``visualization``, and for CV
            ``n_splits`` / ``random_state``.
        destination: Directory for figure files.
        mode: ``"holdout"`` or ``"cv"``.
        logger: Optional logger.

    Returns:
        Written figure paths (empty when visualization is disabled).
    """
    log = logger or logging.getLogger(__name__)
    viz = getattr(config, "visualization", None)
    if not visualization_enabled(
        is_visualize=bool(getattr(config, "is_visualize", False)),
        visualization=viz,
    ):
        log.info("Visualization disabled (is_visualize / visualization.enabled).")
        return []

    plot_types = list(getattr(viz, "plot_types", ["roc", "dca", "calibration"]))
    dpi = int(getattr(viz, "dpi", 600))
    image_format = str(getattr(viz, "format", "pdf"))
    explainability = getattr(viz, "explainability", None)
    if mode == "holdout":
        if not isinstance(result, ModelResult):
            raise HABITAPIError(
                "write_ml_figures_from_config(mode='holdout') needs a ModelResult."
            )
        return write_holdout_model_figures(
            result,
            table,
            destination=destination,
            plot_types=plot_types,
            dpi=dpi,
            image_format=image_format,
            explainability=explainability,
            logger=log,
        )
    if mode == "cv":
        if not isinstance(result, CVResult):
            raise HABITAPIError(
                "write_ml_figures_from_config(mode='cv') needs a CVResult."
            )
        return write_cv_model_figures(
            result,
            table,
            destination=destination,
            plot_types=plot_types,
            n_splits=int(getattr(config, "n_splits", result.n_splits)),
            seed=int(getattr(config, "random_state", 0)),
            dpi=dpi,
            image_format=image_format,
            explainability=explainability,
            logger=log,
        )
    raise HABITAPIError(
        f"write_ml_figures_from_config: unknown mode {mode!r} "
        "(expected 'holdout' or 'cv')."
    )


def visualization_enabled(
    *,
    is_visualize: bool,
    visualization: Optional[Any] = None,
) -> bool:
    """
    Resolve the v0.1 ``is_visualize`` / ``visualization.enabled`` pair.

    Args:
        is_visualize: Top-level MLConfig flag.
        visualization: Optional VisualizationConfig (or mapping) with
            ``enabled``.

    Returns:
        ``True`` when figures should be produced.
    """
    if not bool(is_visualize):
        return False
    if visualization is None:
        return True
    enabled = getattr(visualization, "enabled", None)
    if enabled is None and isinstance(visualization, Mapping):
        enabled = visualization.get("enabled", True)
    return True if enabled is None else bool(enabled)


def write_classification_figures(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    destination: Union[str, Path],
    plot_types: Sequence[str],
    y_pred: Optional[np.ndarray] = None,
    model_name: str = "model",
    prefix: str = "",
    dpi: int = 600,
    image_format: str = "pdf",
    feature_matrix: Optional[Any] = None,
    predict_fn: Optional[Any] = None,
    explainability: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """
    Render configured classification figures and write them to disk.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Positive-class predicted probabilities.
        destination: Directory for figure files (created when missing).
        plot_types: Subset of ``roc`` / ``dca`` / ``calibration`` / ``pr`` /
            ``confusion`` / ``shap`` / ``shap_dependence`` /
            ``shap_waterfall`` / ``permutation``.
        y_pred: Hard class predictions; required for ``confusion``.
        model_name: Series / filename stem used in titles and SHAP files.
        prefix: Filename prefix (e.g. ``train_`` / ``test_`` / ``cv_``).
        dpi: Raster DPI for non-PDF formats.
        image_format: File extension without a leading dot.
        feature_matrix: Optional feature frame/array for SHAP / permutation.
        predict_fn: Optional ``callable(X) -> probability`` for SHAP /
            permutation.
        explainability: Optional VisualizationConfig.explainability block.
        logger: Optional logger for skip / soft-failure messages.

    Returns:
        Paths of figures that were successfully written.

    Raises:
        HABITAPIError: When a required array for an enabled plot type is
            missing or malformed.
    """
    log = logger or logging.getLogger(__name__)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    ext = image_format.lstrip(".").lower() or "pdf"
    requested = [str(name).lower() for name in plot_types]
    unknown = sorted({name for name in requested if name not in _KNOWN_TYPES})
    if unknown:
        log.warning("Ignoring unknown plot_types: %s", unknown)

    written: List[Path] = []
    curves = {model_name: (np.asarray(y_true), np.asarray(y_prob))}

    from habit.viz import use_style
    from habit.viz.classification import (
        plot_calibration,
        plot_confusion_matrix,
        plot_decision_curve,
        plot_precision_recall,
        plot_roc,
    )

    plt = require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)

    with use_style("radiology"):
        if "roc" in requested:
            path = root / f"{prefix}roc_curve.{ext}"
            fig = plot_roc(curves=curves, title=_title(prefix, "ROC"))
            _savefig(fig, path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        if "dca" in requested:
            path = root / f"{prefix}decision_curve.{ext}"
            fig = plot_decision_curve(
                curves=curves, title=_title(prefix, "Decision Curve")
            )
            _savefig(fig, path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        if "calibration" in requested:
            path = root / f"{prefix}calibration_curve.{ext}"
            fig = plot_calibration(
                curves=curves, title=_title(prefix, "Calibration")
            )
            _savefig(fig, path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        if "pr" in requested:
            path = root / f"{prefix}pr_curve.{ext}"
            fig = plot_precision_recall(
                curves=curves, title=_title(prefix, "Precision-Recall")
            )
            _savefig(fig, path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        if "confusion" in requested:
            if y_pred is None:
                raise HABITAPIError(
                    "write_classification_figures: plot_types includes "
                    "'confusion' but y_pred was not provided."
                )
            path = root / f"{prefix}{model_name}_confusion_matrix.{ext}"
            fig = plot_confusion_matrix(
                np.asarray(y_true),
                np.asarray(y_pred),
                title=f"{model_name} Confusion Matrix",
            )
            _savefig(fig, path, dpi=dpi)
            plt.close(fig)
            written.append(path)

    shap_requested = [name for name in requested if name in _SHAP_TYPES]
    if shap_requested:
        written.extend(
            _try_write_shap(
                destination=root,
                prefix=prefix,
                model_name=model_name,
                ext=ext,
                dpi=dpi,
                plot_types=shap_requested,
                feature_matrix=feature_matrix,
                predict_fn=predict_fn,
                explainability=explainability,
                logger=log,
            )
        )

    if "permutation" in requested:
        written.extend(
            _try_write_permutation(
                destination=root,
                prefix=prefix,
                model_name=model_name,
                ext=ext,
                dpi=dpi,
                y_true=np.asarray(y_true),
                feature_matrix=feature_matrix,
                predict_fn=predict_fn,
                explainability=explainability,
                logger=log,
            )
        )

    return written


def write_holdout_model_figures(
    result: ModelResult,
    table: FeatureTable,
    *,
    destination: Union[str, Path],
    plot_types: Sequence[str],
    dpi: int = 600,
    image_format: str = "pdf",
    explainability: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """
    Write train- and test-set evaluation figures for one hold-out train run.

    Mirrors the v0.1 PlotComposer coverage (both splits) while staying on the
    v1 ``habit.viz`` path. Prefixes are ``train_`` and ``test_``.

    Args:
        result: Outcome of :func:`habit.recipes.modeling.train_model`.
        table: Table the recipe trained on.
        destination: Directory for figure files.
        plot_types: Enabled figure names.
        dpi: Raster DPI for non-PDF formats.
        image_format: File extension without a leading dot.
        explainability: Optional VisualizationConfig.explainability block.
        logger: Optional logger.

    Returns:
        Written figure paths (empty when no scorable rows are available).
    """
    log = logger or logging.getLogger(__name__)
    model_name = result.pipeline.model.spec.name
    want_explain = bool(
        _SHAP_TYPES.intersection(str(p).lower() for p in plot_types)
        or "permutation" in {str(p).lower() for p in plot_types}
    )
    splits: List[Tuple[str, Sequence[str]]] = [
        ("train_", result.train_row_ids),
        ("test_", result.test_row_ids),
    ]
    # No hold-out split: score the full table once under the train_ prefix.
    if not result.train_row_ids and not result.test_row_ids:
        splits = [("train_", tuple(_row_ids(table)))]

    written: List[Path] = []
    for prefix, row_ids in splits:
        arrays = _score_split_arrays(result, table, row_ids=row_ids)
        if arrays is None:
            log.warning(
                "Skipping %s figures: no scorable rows for this split "
                "(check train/test_row_ids and outcome column).",
                prefix.rstrip("_"),
            )
            continue
        y_true, y_prob, y_pred, feature_frame = arrays
        predict_fn = (
            _make_pipeline_predict_fn(
                result.pipeline, table, feature_frame.columns
            )
            if want_explain
            else None
        )
        written.extend(
            write_classification_figures(
                y_true=y_true,
                y_prob=y_prob,
                y_pred=y_pred,
                destination=destination,
                plot_types=plot_types,
                model_name=model_name,
                prefix=prefix,
                dpi=dpi,
                image_format=image_format,
                feature_matrix=feature_frame,
                predict_fn=predict_fn,
                explainability=explainability,
                logger=log,
            )
        )
    if not written:
        log.warning(
            "No hold-out ML figures were written. Enable is_visualize and "
            "ensure the train run produced train/test row ids."
        )
    return written


def write_cv_model_figures(
    result: CVResult,
    table: FeatureTable,
    *,
    destination: Union[str, Path],
    plot_types: Sequence[str],
    n_splits: int,
    seed: Optional[int],
    dpi: int = 600,
    image_format: str = "pdf",
    explainability: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """
    Write out-of-fold pooled evaluation figures for a CV run.

    Fold indices are regenerated with the same ``n_splits`` / ``seed`` used
    by :func:`cross_validate`, then each stored fold pipeline scores its
    validation rows. The pooled OOF predictions drive the summary ROC/DCA/etc.
    SHAP / permutation on CV use the pooled OOF feature rows when requested.

    Args:
        result: Outcome of :func:`habit.recipes.modeling.cross_validate`.
        table: Table the recipe evaluated.
        destination: Directory for figure files.
        plot_types: Enabled figure names.
        n_splits: Fold count (must match the CV run).
        seed: Random seed used for fold shuffling (must match the CV run).
        dpi: Raster DPI for non-PDF formats.
        image_format: File extension without a leading dot.
        explainability: Optional VisualizationConfig.explainability block.
        logger: Optional logger.

    Returns:
        Written figure paths (empty when pipelines are unavailable).
    """
    log = logger or logging.getLogger(__name__)
    if not result.pipelines:
        log.warning("CV result has no fold pipelines; skipping ML figures.")
        return []
    y_true, y_prob, y_pred, feature_frame = _cv_oof_prediction_arrays(
        result, table, n_splits=n_splits, seed=seed
    )
    model_name = str(result.manifest.spec_payload["classifier"]["name"])
    want_explain = bool(
        _SHAP_TYPES.intersection(str(p).lower() for p in plot_types)
        or "permutation" in {str(p).lower() for p in plot_types}
    )
    # OOF SHAP uses the last fold pipeline as a representative scorer on the
    # pooled feature matrix (same columns). This is an approximation; per-fold
    # SHAP is deferred.
    predict_fn = None
    if want_explain and feature_frame is not None:
        predict_fn = _make_pipeline_predict_fn(
            result.pipelines[-1], table, feature_frame.columns
        )
        log.info(
            "CV SHAP/permutation uses the last fold pipeline on pooled OOF "
            "rows (not per-fold attribution)."
        )
    return write_classification_figures(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        destination=destination,
        plot_types=plot_types,
        model_name=model_name,
        prefix="cv_",
        dpi=dpi,
        image_format=image_format,
        feature_matrix=feature_frame,
        predict_fn=predict_fn,
        explainability=explainability,
        logger=log,
    )


def _score_split_arrays(
    result: ModelResult,
    table: FeatureTable,
    *,
    row_ids: Sequence[str],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    Score one hold-out side (train or test) of a ModelResult.

    Returns:
        ``(y_true, y_prob, y_pred, feature_frame)`` or ``None`` when the split
        is empty or the table has no outcome column.
    """
    if not row_ids:
        return None
    if table.outcome is None:
        return None
    all_ids = _row_ids(table)
    id_to_index = {rid: idx for idx, rid in enumerate(all_ids)}
    missing = [rid for rid in row_ids if rid not in id_to_index]
    if missing:
        raise HABITAPIError(
            "write_holdout_model_figures: row ids missing from the feature "
            f"table: {missing[:5]}{'...' if len(missing) > 5 else ''}."
        )
    indices = [id_to_index[rid] for rid in row_ids]
    split_table = _select_rows(table, indices)
    prediction = predict_model(result.pipeline, split_table)
    label_col = table.outcome.column
    y_true = split_table.frame[label_col].to_numpy()
    y_pred = prediction.predictions.to_numpy()
    y_prob = _positive_proba(prediction.probabilities)
    feature_frame = split_table.frame.loc[
        :, list(split_table.feature_columns)
    ].copy()
    return y_true, y_prob, y_pred, feature_frame


def _cv_oof_prediction_arrays(
    result: CVResult,
    table: FeatureTable,
    *,
    n_splits: int,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Pool out-of-fold predictions aligned with ``result.pipelines``.

    Returns:
        ``(y_true, y_prob, y_pred, feature_frame)`` concatenated in fold
        validation order.
    """
    from habit.domain.split import kfold_indices, stratify_labels

    if table.outcome is None:
        raise HABITAPIError(
            "write_cv_model_figures requires a labelled FeatureTable."
        )
    labels = stratify_labels(table.outcome, table.frame)
    label_col = table.outcome.column
    y_true_parts: List[np.ndarray] = []
    y_prob_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []
    feature_parts: List[pd.DataFrame] = []
    folds = list(
        kfold_indices(
            len(table.frame),
            n_splits=n_splits,
            labels=labels,
            seed=seed,
        )
    )
    if len(folds) != len(result.pipelines):
        raise HABITAPIError(
            "write_cv_model_figures: fold count mismatch between stored "
            f"pipelines ({len(result.pipelines)}) and regenerated splits "
            f"({len(folds)})."
        )
    for pipeline, (_train_idx, val_idx) in zip(result.pipelines, folds):
        val_table = _select_rows(table, val_idx)
        prediction = predict_model(pipeline, val_table)
        y_true_parts.append(val_table.frame[label_col].to_numpy())
        y_pred_parts.append(prediction.predictions.to_numpy())
        y_prob_parts.append(_positive_proba(prediction.probabilities))
        feature_parts.append(
            val_table.frame.loc[:, list(val_table.feature_columns)].copy()
        )
    return (
        np.concatenate(y_true_parts),
        np.concatenate(y_prob_parts),
        np.concatenate(y_pred_parts),
        pd.concat(feature_parts, axis=0, ignore_index=True),
    )


def _make_pipeline_predict_fn(
    pipeline: Any,
    table: FeatureTable,
    feature_columns: Sequence[str],
) -> Any:
    """
    Build a SHAP-compatible ``f(X) -> positive probability`` callable.

    Args:
        pipeline: Fitted :class:`~habit.domain.pipeline.TablePipeline`.
        table: Source table (provides identifier column names).
        feature_columns: Raw feature column order expected by the pipeline.

    Returns:
        Callable mapping a 2-D feature array to positive-class probabilities.
    """
    id_col = table.id_columns[0]
    columns = list(feature_columns)

    def _predict_fn(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(data, dtype=np.float64), columns=columns)
        frame.insert(0, id_col, [f"shap_row_{i}" for i in range(len(frame))])
        mini = FeatureTable(
            frame=frame,
            id_columns=(id_col,),
            feature_columns=tuple(columns),
            outcome=None,
            provenance=None,
        )
        proba = pipeline.predict_proba(mini)
        return _positive_proba(proba)

    return _predict_fn


def _positive_proba(probabilities: Optional[pd.DataFrame]) -> np.ndarray:
    """Extract the positive-class probability column from a proba frame."""
    if probabilities is None or probabilities.shape[1] == 0:
        raise HABITAPIError(
            "Classification figures require predict_proba output."
        )
    columns = list(probabilities.columns)
    positive = "1" if "1" in columns else columns[-1]
    return probabilities[positive].to_numpy(dtype=np.float64)


def _title(prefix: str, default: str) -> str:
    """Build a short English title from an optional filename prefix."""
    cleaned = prefix.strip("_ ").strip()
    return f"{cleaned} {default}".strip() if cleaned else default


def _savefig(fig: Any, path: Path, *, dpi: int) -> None:
    """Write one figure with PDF-friendly defaults."""
    if path.suffix.lower() == ".pdf":
        fig.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _explainability_option(
    explainability: Optional[Any], name: str, default: Any
) -> Any:
    """Read one field from VisualizationConfig.explainability (or mapping)."""
    if explainability is None:
        return default
    if isinstance(explainability, Mapping):
        return explainability.get(name, default)
    return getattr(explainability, name, default)


def _feature_matrix_parts(
    feature_matrix: Any,
) -> Tuple[np.ndarray, List[str]]:
    """Split a frame/array into ``(x_arr, feature_names)``."""
    if isinstance(feature_matrix, pd.DataFrame):
        return (
            feature_matrix.to_numpy(dtype=np.float64),
            [str(c) for c in feature_matrix.columns],
        )
    x_arr = np.asarray(feature_matrix, dtype=np.float64)
    return x_arr, [f"f{i}" for i in range(x_arr.shape[1])]


def _sanitize_filename_part(name: str) -> str:
    """Keep one path segment ASCII-safe and short."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name)
    )
    return cleaned[:48] or "feature"


def _try_write_shap(
    *,
    destination: Path,
    prefix: str,
    model_name: str,
    ext: str,
    dpi: int,
    plot_types: Sequence[str],
    feature_matrix: Optional[Any],
    predict_fn: Optional[Any],
    explainability: Optional[Any],
    logger: logging.Logger,
) -> List[Path]:
    """Best-effort SHAP figure export; never abort the parent reporting run."""
    if feature_matrix is None or predict_fn is None:
        logger.warning(
            "Skipping SHAP figures (%s): feature_matrix/predict_fn not provided. "
            "Hold-out/CV reporting must pass scored feature rows.",
            ",".join(plot_types),
        )
        return []

    try:
        import shap  # type: ignore
    except ImportError:
        logger.warning(
            "Skipping SHAP figures: optional dependency 'shap' is not installed. "
            'Install with: pip install shap  (or pip install "habitat-analysis[analysis]").'
        )
        return []

    try:
        x_arr, feature_names = _feature_matrix_parts(feature_matrix)

        def _proba(data: np.ndarray) -> np.ndarray:
            raw = predict_fn(data)
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1]
            return arr.reshape(-1)

        n_bg = min(50, x_arr.shape[0])
        background = shap.sample(x_arr, n_bg) if x_arr.shape[0] > n_bg else x_arr
        explainer = shap.Explainer(_proba, background)
        explanation = explainer(x_arr)
        values = np.asarray(explanation.values, dtype=np.float64)
        if values.ndim == 3:
            values = values[:, :, -1]
        base_values = getattr(explanation, "base_values", 0.0)
        if isinstance(base_values, (list, tuple, np.ndarray)):
            base_arr = np.asarray(base_values, dtype=np.float64).reshape(-1)
            base_value = float(base_arr[-1]) if base_arr.size else 0.0
        else:
            base_value = float(base_values)
    except OptionalDependencyError as exc:
        logger.warning("Skipping SHAP figures: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001 - soft-fail reporting path
        logger.warning("Could not compute SHAP values (figures skipped): %s", exc)
        return []

    from habit.viz.classification import (
        plot_shap_dependence,
        plot_shap_summary,
        plot_shap_waterfall,
        rank_shap_feature_indices,
        select_representative_sample_indices,
    )

    plt = require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)

    written: List[Path] = []
    if "shap" in plot_types:
        try:
            path = destination / f"{prefix}{model_name}_shap.{ext}"
            fig = plot_shap_summary(
                values,
                x_arr,
                feature_names=feature_names,
                title=f"{model_name} SHAP summary",
            )
            _savefig(fig, path, dpi=dpi)
            plt.close(fig)
            written.append(path)
        except OptionalDependencyError as exc:
            logger.warning("Skipping SHAP summary: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not render SHAP summary: %s", exc)

    if "shap_dependence" in plot_types:
        top_k = int(
            _explainability_option(explainability, "shap_dependence_top_k", 3)
        )
        for rank, feature_index in enumerate(
            rank_shap_feature_indices(values, top_k=top_k), start=1
        ):
            try:
                fname = feature_names[int(feature_index)]
                safe = _sanitize_filename_part(fname)
                path = (
                    destination
                    / f"{prefix}{model_name}_shap_dependence_{rank}_{safe}.{ext}"
                )
                fig = plot_shap_dependence(
                    values,
                    x_arr,
                    int(feature_index),
                    feature_names=feature_names,
                )
                _savefig(fig, path, dpi=dpi)
                plt.close(fig)
                written.append(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not render SHAP dependence for feature %s: %s",
                    feature_index,
                    exc,
                )

    if "shap_waterfall" in plot_types:
        n_samples = int(
            _explainability_option(explainability, "shap_waterfall_samples", 3)
        )
        sample_indices = select_representative_sample_indices(
            values.sum(axis=1), n_samples=n_samples
        )
        for sample_index in sample_indices:
            try:
                path = (
                    destination
                    / f"{prefix}{model_name}_shap_waterfall_sample{sample_index}.{ext}"
                )
                fig = plot_shap_waterfall(
                    values,
                    x_arr,
                    int(sample_index),
                    feature_names=feature_names,
                    base_value=base_value,
                )
                _savefig(fig, path, dpi=dpi)
                plt.close(fig)
                written.append(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not render SHAP waterfall for sample %s: %s",
                    sample_index,
                    exc,
                )

    return written


def _try_write_permutation(
    *,
    destination: Path,
    prefix: str,
    model_name: str,
    ext: str,
    dpi: int,
    y_true: np.ndarray,
    feature_matrix: Optional[Any],
    predict_fn: Optional[Any],
    explainability: Optional[Any],
    logger: logging.Logger,
) -> List[Path]:
    """Best-effort permutation-importance figure on raw input columns."""
    if feature_matrix is None or predict_fn is None:
        logger.warning(
            "Skipping permutation importance: feature_matrix/predict_fn not "
            "provided."
        )
        return []

    x_arr, feature_names = _feature_matrix_parts(feature_matrix)
    y = np.asarray(y_true).reshape(-1)
    if x_arr.shape[0] != y.shape[0]:
        logger.warning(
            "Skipping permutation importance: feature rows (%d) != labels (%d).",
            x_arr.shape[0],
            y.shape[0],
        )
        return []
    if len(np.unique(y)) < 2:
        logger.warning(
            "Skipping permutation importance: need both classes in y_true."
        )
        return []

    n_repeats = int(_explainability_option(explainability, "permutation_repeats", 10))
    scoring = str(
        _explainability_option(explainability, "permutation_scoring", "roc_auc")
    )
    top_k = int(_explainability_option(explainability, "permutation_top_k", 20))
    seed = _explainability_option(explainability, "permutation_random_state", None)
    rng = np.random.RandomState(None if seed is None else int(seed))

    try:
        from sklearn.metrics import get_scorer

        # Build a tiny sklearn-compatible wrapper around predict_fn.
        class _ProbaEstimator:
            def fit(self, X: Any, y: Any = None) -> Any:
                return self

            def predict_proba(self, X: Any) -> np.ndarray:
                proba = np.asarray(predict_fn(np.asarray(X, dtype=np.float64)))
                proba = proba.reshape(-1)
                return np.column_stack([1.0 - proba, proba])

            def predict(self, X: Any) -> np.ndarray:
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        estimator = _ProbaEstimator()
        scorer = get_scorer(scoring)
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            estimator,
            x_arr,
            y,
            scoring=scorer,
            n_repeats=max(n_repeats, 1),
            random_state=rng,
            n_jobs=1,
        )
        means = np.asarray(result.importances_mean, dtype=np.float64)
        stds = np.asarray(result.importances_std, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not compute permutation importance (figure skipped): %s", exc
        )
        return []

    try:
        from habit.viz.classification import plot_permutation_importance

        plt = require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)

        path = destination / f"{prefix}{model_name}_permutation_importance.{ext}"
        fig = plot_permutation_importance(
            feature_names,
            means,
            importance_std=stds,
            title=f"{model_name} Permutation Importance",
            top_k=top_k,
        )
        _savefig(fig, path, dpi=dpi)
        plt.close(fig)
        csv_path = destination / f"{prefix}{model_name}_permutation_importance.csv"
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": means,
                "importance_std": stds,
            }
        ).sort_values("importance_mean", ascending=False).to_csv(
            csv_path, index=False
        )
        return [path]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not render permutation importance figure: %s", exc
        )
        return []
