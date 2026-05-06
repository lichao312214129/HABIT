"""
Structured result objects for machine-learning workflows.

The result hierarchy is intentionally symmetric across execution modes so the
reporting layer (``ModelStore``, ``ReportWriter``, ``PlotComposer``) can work
against a single contract:

* :class:`ModelResult`           - one fitted estimator on a holdout split.
* :class:`KFoldModelResult`      - per-fold breakdown for one model.
* :class:`AggregatedModelResult` - cross-fold aggregation for one model.
* :class:`RunResult`             - full holdout run output.
* :class:`KFoldRunResult`        - full K-Fold run output.
* :class:`InferenceResult`       - prediction-only output (no training).

All result objects are frozen dataclasses and provide ``to_legacy_results``
helpers that return dictionary payloads compatible with the historical
output shape.  The legacy adapters give the visualisation/report writers a
gradual migration path while the new structured fields enable strict
unit testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .dataset import DatasetSnapshot
from .plan import WorkflowPlan


# ---------------------------------------------------------------------------
# Per-model results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelResult:
    """
    Per-model holdout result payload.

    Attributes
    ----------
    model_name:
        Name defined in ML config.
    train:
        Serialised prediction payload on the training split.
    test:
        Serialised prediction payload on the test split.
    train_metrics:
        Metric dictionary computed on the training split.
    test_metrics:
        Metric dictionary computed on the test split.
    fitted_estimator:
        Trained sklearn-compatible estimator/pipeline.
    feature_names:
        Ordered input feature names used for training.
    train_subject_ids:
        Subject identifiers (``DataFrame.index`` values) of training samples.
        Stored on the result so reports can join predictions back to subjects
        without keeping the original feature matrix around.
    test_subject_ids:
        Subject identifiers of test samples.
    """

    model_name: str
    train: Dict[str, Any]
    test: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    fitted_estimator: Any
    feature_names: Tuple[str, ...]
    train_subject_ids: Tuple[Any, ...] = ()
    test_subject_ids: Tuple[Any, ...] = ()

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy workflow result format (compatible with PlotManager).
        """
        train_payload: Dict[str, Any] = dict(self.train)
        test_payload: Dict[str, Any] = dict(self.test)
        train_payload["metrics"] = self.train_metrics
        test_payload["metrics"] = self.test_metrics
        return {
            "train": train_payload,
            "test": test_payload,
            "pipeline": self.fitted_estimator,
            "features": list(self.feature_names),
        }


@dataclass(frozen=True)
class KFoldModelResult:
    """
    Per-fold breakdown for a single model under K-Fold cross-validation.

    Attributes
    ----------
    model_name:
        Name defined in ML config.
    folds:
        Ordered list of fold-level prediction payloads.  Each entry contains
        ``y_true``, ``y_prob``, ``y_pred`` and ``metrics`` keys (matching the
        historical structure consumed by reporting code).
    fold_estimators:
        Trained estimator for each fold, in the same order as ``folds``.
    """

    model_name: str
    folds: Tuple[Dict[str, Any], ...]
    fold_estimators: Tuple[Any, ...]

    def to_legacy_fold_payloads(self) -> List[Dict[str, Any]]:
        """Return fold payloads as a plain list (legacy report format)."""
        return [dict(fold) for fold in self.folds]


@dataclass(frozen=True)
class AggregatedModelResult:
    """
    Cross-fold aggregation for a single model.

    Attributes
    ----------
    model_name:
        Name defined in ML config.
    raw:
        Concatenated ``y_true``/``y_prob``/``y_pred`` across all folds, kept
        as a dict so the legacy plot pipeline (which expects a ``raw`` key)
        works without translation.
    overall_metrics:
        Metrics computed on the concatenated raw arrays.
    auc_mean:
        Mean of per-fold AUCs (``None`` when AUC is not available).
    auc_std:
        Standard deviation of per-fold AUCs.
    """

    model_name: str
    raw: Dict[str, Any]
    overall_metrics: Dict[str, float]
    auc_mean: Optional[float] = None
    auc_std: Optional[float] = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Build a payload matching the historical ``aggregated[name]`` shape.

        Older readers expect:
        ``{ 'y_true': ..., 'y_prob': ..., 'y_pred': ..., 'metrics': ...,
        'auc_mean': ..., 'auc_std': ... }`` where the prediction arrays live
        at the top level.  Some plot helpers also look for a nested ``raw``
        key, so it is exposed for forward compatibility.
        """
        payload: Dict[str, Any] = dict(self.raw)
        payload["metrics"] = dict(self.overall_metrics)
        payload["raw"] = dict(self.raw)
        if self.auc_mean is not None:
            payload["auc_mean"] = self.auc_mean
        if self.auc_std is not None:
            payload["auc_std"] = self.auc_std
        return payload


# ---------------------------------------------------------------------------
# Workflow-level results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """
    Full holdout workflow output package.

    Attributes
    ----------
    plan:
        Workflow plan snapshot.
    models:
        Mapping from model name to model result object.
    summary_rows:
        Summary rows used for summary CSV generation.
    dataset:
        Snapshot of the train/test data used by the run.  Predictions can
        still be joined to subject identifiers via the embedded indices.
    created_at:
        UTC timestamp in ISO8601 format.
    """

    plan: WorkflowPlan
    models: Dict[str, ModelResult]
    summary_rows: List[Dict[str, Any]]
    dataset: DatasetSnapshot
    created_at: str

    @classmethod
    def create(
        cls,
        plan: WorkflowPlan,
        models: Dict[str, ModelResult],
        summary_rows: List[Dict[str, Any]],
        dataset: DatasetSnapshot,
    ) -> "RunResult":
        """Build a run result with auto timestamp."""
        return cls(
            plan=plan,
            models=models,
            summary_rows=summary_rows,
            dataset=dataset,
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

    # ------------------------------------------------------------------
    # Backward-compatible accessors
    # ------------------------------------------------------------------

    @property
    def x_train(self) -> Optional[pd.DataFrame]:
        """Backward-compat accessor mirroring the old direct attribute."""
        return self.dataset.x_train

    @property
    def x_test(self) -> Optional[pd.DataFrame]:
        """Backward-compat accessor mirroring the old direct attribute."""
        return self.dataset.x_test

    @property
    def y_train(self) -> Optional[pd.Series]:
        """Backward-compat accessor mirroring the old direct attribute."""
        return self.dataset.y_train

    @property
    def y_test(self) -> Optional[pd.Series]:
        """Backward-compat accessor mirroring the old direct attribute."""
        return self.dataset.y_test

    @property
    def label_col(self) -> str:
        """Backward-compat accessor mirroring the old direct attribute."""
        return self.dataset.label_col

    def to_legacy_results(self) -> Dict[str, Dict[str, Any]]:
        """Convert to the historical ``workflow.results`` format."""
        return {name: model.to_legacy_dict() for name, model in self.models.items()}


@dataclass(frozen=True)
class KFoldRunResult:
    """
    Full K-Fold workflow output package.

    The structured fields (``models`` / ``aggregated``) are the canonical
    contract; ``results`` is exposed as a property that re-materialises the
    historical dict payload so existing reports keep working unchanged.

    Attributes
    ----------
    plan:
        Workflow plan snapshot.
    models:
        Mapping from model name to per-fold breakdown object.
    aggregated:
        Mapping from model name to cross-fold aggregation object.
    summary_rows:
        Summary rows used for summary CSV generation.
    created_at:
        UTC timestamp in ISO8601 format.
    """

    plan: WorkflowPlan
    models: Dict[str, KFoldModelResult]
    aggregated: Dict[str, AggregatedModelResult]
    summary_rows: List[Dict[str, Any]]
    created_at: str

    @classmethod
    def create(
        cls,
        plan: WorkflowPlan,
        models: Dict[str, KFoldModelResult],
        aggregated: Dict[str, AggregatedModelResult],
        summary_rows: List[Dict[str, Any]],
    ) -> "KFoldRunResult":
        """Build a K-Fold run result with auto timestamp."""
        return cls(
            plan=plan,
            models=models,
            aggregated=aggregated,
            summary_rows=summary_rows,
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

    # ------------------------------------------------------------------
    # Backward-compatible accessors
    # ------------------------------------------------------------------

    @property
    def fold_pipelines(self) -> Dict[str, List[Any]]:
        """Backward-compat accessor: model -> list of fitted estimators."""
        return {
            name: list(model.fold_estimators)
            for name, model in self.models.items()
        }

    @property
    def results(self) -> Dict[str, Any]:
        """Backward-compat accessor returning the legacy payload shape."""
        return self.to_legacy_results()

    def to_legacy_results(self) -> Dict[str, Any]:
        """
        Convert to the historical ``workflow.results`` format.

        Output shape::

            {
              'folds': [
                  {'fold': 1, 'models': {model_name: <fold payload>, ...}},
                  ...
              ],
              'aggregated': {model_name: <aggregated payload>, ...}
            }
        """
        # Walk fold payloads keyed by fold index across all models.
        folds: List[Dict[str, Any]] = []
        if self.models:
            n_folds = max(len(model.folds) for model in self.models.values())
            for fold_idx in range(n_folds):
                fold_models: Dict[str, Any] = {}
                for model_name, model in self.models.items():
                    if fold_idx < len(model.folds):
                        fold_models[model_name] = dict(model.folds[fold_idx])
                folds.append({"fold": fold_idx + 1, "models": fold_models})

        aggregated_payload: Dict[str, Any] = {
            name: agg.to_legacy_dict() for name, agg in self.aggregated.items()
        }

        return {"folds": folds, "aggregated": aggregated_payload}


@dataclass(frozen=True)
class InferenceResult:
    """
    Output of a prediction-only run.

    Attributes
    ----------
    plan:
        Workflow plan snapshot.
    pipeline_path:
        Filesystem path of the loaded pipeline file.
    predictions:
        DataFrame to be written as ``prediction_results.csv`` (already
        contains predicted labels and probabilities).
    metrics:
        Optional evaluation metrics when ground-truth labels are available.
    label_col:
        Resolved ground-truth column name, ``None`` when evaluation was
        skipped.
    summary_rows:
        Summary rows for unified reporting (empty for inference).
    created_at:
        UTC timestamp in ISO8601 format.
    """

    plan: WorkflowPlan
    pipeline_path: str
    predictions: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)
    label_col: Optional[str] = None
    summary_rows: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""

    @classmethod
    def create(
        cls,
        plan: WorkflowPlan,
        pipeline_path: str,
        predictions: pd.DataFrame,
        metrics: Optional[Dict[str, float]] = None,
        label_col: Optional[str] = None,
    ) -> "InferenceResult":
        """Build an inference result with auto timestamp."""
        return cls(
            plan=plan,
            pipeline_path=pipeline_path,
            predictions=predictions,
            metrics=dict(metrics or {}),
            label_col=label_col,
            summary_rows=[],
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

    def to_legacy_results(self) -> Dict[str, Any]:
        """Inference payload exposed as a dict for parity with other results."""
        return {
            "predictions": self.predictions,
            "metrics": dict(self.metrics),
            "label_col": self.label_col,
            "pipeline_path": self.pipeline_path,
        }
