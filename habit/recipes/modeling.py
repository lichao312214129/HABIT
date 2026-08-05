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
"""Tabular machine-learning recipes, as assembly functions (L4).

Like the habitat designs in :mod:`habit.recipes.habitat`, each function here
is a WIRING DIAGRAM, not an engine: it reads a
:class:`~habit.spec.specs.MLSpec`, builds the declared components through the
domain registries (in :mod:`habit.domain.assembly`, the single construction
site), runs them through :class:`~habit.domain.pipeline.TablePipeline`, and
packs the outcome into a minimal result container. Orchestration concerns --
parallelism, resume, persistence -- stay outside: the fitted pipeline is
returned in memory and a caller that wants it on disk uses
:meth:`TablePipeline.save`.

Three recipes exist because three scientific acts exist:

* :func:`train_model` -- fit ONE pipeline on ONE table and score it. With no
  split arguments the score is the training-set readout of a final model;
  with ``test_size`` (random/stratified hold-out) or explicit
  ``train_ids``/``test_ids`` (custom hold-out, v0.1's id-file split) the
  pipeline is fitted on the training rows only and scored on the held-out
  rows as well, so the validation design -- which the ``MLSpec`` deliberately
  does not carry -- is stated at the call site that owns it.
* :func:`cross_validate` -- estimate generalisation by refitting the SAME
  definition on K stratified folds; every fold builds a fresh pipeline from
  the spec, so no fitted state can leak across folds.
* :func:`predict_model` -- apply one FITTED pipeline to new rows. This is
  the inference half of the train/predict contract: preprocessing and
  selection replay their TRAINING state, never refit on the new data.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.manifest import RunManifest
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.table import FeatureTable
from habit.domain.assembly import build_ml_metrics, build_table_pipeline
from habit.domain.pipeline import TablePipeline
from habit.domain.split import kfold_indices, stratify_labels, train_test_indices
from habit.domain.table_protocols import Classifier
from habit.spec.specs import MLSpec

__all__ = [
    "ModelResult",
    "CVResult",
    "PredictionResult",
    "train_model",
    "cross_validate",
    "predict_model",
]

#: Metric panel a recipe falls back on when the spec declares none. Kept
#: deliberately small: the panel a paper reports belongs in the spec.
_DEFAULT_METRIC_NAMES: Tuple[str, ...] = ("accuracy", "auc")


@dataclass(frozen=True, eq=False)
class ModelResult:
    """
    Outcome of :func:`train_model`, entirely in memory.

    Attributes:
        pipeline: The fitted pipeline -- the publishable artefact, persisted
            with :meth:`TablePipeline.save` when the caller wants a file.
            Under a hold-out split it is fitted on the TRAINING rows only,
            mirroring the v0.1 ``*_final_pipeline.pkl`` semantics.
        train_metrics: Metric panel scored on the training rows. Named
            ``train_`` so nobody mistakes an in-sample number for a
            generalisation estimate.
        manifest: Record of what ran: the effective spec payload, the
            provenance chain, and every row's outcome.
        test_metrics: Metric panel scored on the held-out rows, or ``None``
            when the recipe ran without a hold-out split.
        train_row_ids: Row ids of the training side of the hold-out split,
            in split order; empty when no split ran. Recording the ids makes
            the split itself reproducible from the result object alone.
        test_row_ids: Row ids of the held-out side; empty when no split ran.
    """

    pipeline: TablePipeline
    train_metrics: Mapping[str, float]
    manifest: RunManifest
    test_metrics: Optional[Mapping[str, float]] = None
    train_row_ids: Tuple[str, ...] = ()
    test_row_ids: Tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class CVResult:
    """
    Outcome of :func:`cross_validate`, entirely in memory.

    Attributes:
        fold_metrics: Metric panel per fold, in fold order.
        mean_metrics: Panel averaged across folds (NaN-safe).
        std_metrics: Panel standard deviation across folds (NaN-safe).
        n_splits: Number of folds actually run.
        pipelines: The per-fold fitted pipelines, in fold order. Each was
            fitted on its own training rows only; keeping them lets a caller
            inspect fold-level selection stability without refitting.
        manifest: Record of what ran: the effective spec payload, the
            provenance chain, and every row's outcome.
    """

    fold_metrics: Tuple[Mapping[str, float], ...]
    mean_metrics: Mapping[str, float]
    std_metrics: Mapping[str, float]
    n_splits: int
    manifest: RunManifest
    pipelines: Tuple[TablePipeline, ...] = field(default=())


def train_model(
    table: FeatureTable,
    spec: MLSpec,
    *,
    seed: Optional[int] = None,
    test_size: Optional[float] = None,
    stratify: bool = True,
    train_ids: Optional[Sequence[str]] = None,
    test_ids: Optional[Sequence[str]] = None,
) -> ModelResult:
    """
    Fit one pipeline on a table and score it, with an optional hold-out split.

    Without split arguments the pipeline is fitted on every row and scored
    on those same rows. With ``test_size`` the rows are first split into a
    training and a held-out side (stratified on the outcome unless
    ``stratify=False``, the v0.1 ``random`` method); with ``train_ids`` /
    ``test_ids`` the split follows the given row ids exactly (the v0.1
    ``custom`` method). Under a split the pipeline sees the training rows
    ONLY, so preprocessing statistics and feature selection can never leak
    in from the held-out rows, and both sides are scored.

    Args:
        table: Feature table with a declared outcome.
        spec: The modelling definition to fit.
        seed: Optional seed override, folded into the spec (and therefore
            into the split shuffling, the component seeding, and the
            manifest) before anything runs.
        test_size: Fraction of rows assigned to the held-out side; ``None``
            keeps the no-split behaviour. Mutually exclusive with the id
            lists.
        stratify: Stratify the ``test_size`` split on the outcome when the
            endpoint family has strata; ignored for id-list splits and
            continuous endpoints (which have no strata).
        train_ids: Row ids (identifier columns joined as in
            :func:`_row_ids`) forming the training side of a custom split.
            Must be given together with ``test_ids``.
        test_ids: Row ids forming the held-out side of a custom split.

    Returns:
        The fitted pipeline, the training-set panel, and -- under a split --
        the held-out panel plus both sides' row ids.

    Raises:
        HABITAPIError: If the table declares no outcome, the split arguments
            are contradictory, an id list is empty or names rows the table
            does not have, or the two id lists overlap.

    Examples:
        >>> from habit import MLSpec, Spec, make_synthetic_feature_table
        >>> import habit.recipes as recipes
        >>> table = make_synthetic_feature_table(n_rows=60, n_features=8, rng=42)
        >>> spec = MLSpec(
        ...     name="demo",
        ...     table_preprocessors=(Spec("zscore"),),
        ...     classifier=Spec("LogisticRegression", {"max_iter": 500}),
        ...     metrics=(Spec("accuracy"), Spec("auc")),
        ... )
        >>> result = recipes.train_model(table, spec, test_size=0.25, seed=42)
        >>> sorted(result.test_metrics)
        ['accuracy', 'auc']
    """
    _require_outcome(table, owner="train_model")
    effective = _effective_spec(spec, seed)
    started_at = _now()
    split = _resolve_holdout(
        table,
        effective,
        test_size=test_size,
        stratify=stratify,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    metrics = build_ml_metrics(effective, default_names=_DEFAULT_METRIC_NAMES)
    pipeline = build_table_pipeline(effective)
    if split is None:
        pipeline.fit(table)
        train_metrics = pipeline.evaluate(table, metrics)
        return ModelResult(
            pipeline=pipeline,
            train_metrics=train_metrics,
            manifest=_manifest("train_model", effective, table, started_at),
        )

    fit_table = _select_rows(table, split[0])
    holdout_table = _select_rows(table, split[1])
    pipeline.fit(fit_table)
    train_metrics = pipeline.evaluate(fit_table, metrics)
    test_metrics = pipeline.evaluate(holdout_table, metrics)
    row_ids = _row_ids(table)
    return ModelResult(
        pipeline=pipeline,
        train_metrics=train_metrics,
        manifest=_manifest("train_model", effective, table, started_at),
        test_metrics=test_metrics,
        train_row_ids=tuple(row_ids[int(index)] for index in split[0]),
        test_row_ids=tuple(row_ids[int(index)] for index in split[1]),
    )


@dataclass(frozen=True, eq=False)
class PredictionResult:
    """
    Outcome of :func:`predict_model`, entirely in memory.

    Attributes:
        predictions: The terminal model's output per row (class labels for a
            classifier, values for a regressor, risk scores for a survival
            model), indexed by the table's identifier columns.
        probabilities: Per-class probability frame when the terminal model
            is a classifier; ``None`` for regressors and survival models,
            which have no class-probability output.
        manifest: Record of what ran: the fitted pipeline's composed spec,
            the provenance chain, and every row's outcome.
    """

    predictions: pd.Series
    probabilities: Optional[pd.DataFrame]
    manifest: RunManifest


def predict_model(pipeline: TablePipeline, table: FeatureTable) -> PredictionResult:
    """
    Apply one FITTED pipeline to new rows (the inference recipe).

    The pipeline's preprocessing and feature-selection steps replay the state
    they learned on the training table -- the new rows are normalised with
    the TRAINING statistics and reduced with the TRAINING selection, never
    refitted. The table needs no outcome column: inference is legitimate on
    unlabelled rows, and scoring stays with :meth:`TablePipeline.evaluate`.

    Args:
        pipeline: A fitted pipeline, e.g. the ``pipeline`` of a
            :class:`ModelResult` or one reloaded with
            :meth:`TablePipeline.load`.
        table: Rows to predict, carrying the feature columns seen at fit
            time. An outcome column may be present (external-validation
            tables) but is never read.

    Returns:
        The per-row predictions, class probabilities when the terminal model
        is a classifier, and the run manifest.

    Raises:
        HABITAPIError: If the pipeline is not fitted or the table lacks a
            feature column seen at fit time.

    Examples:
        >>> from habit import MLSpec, Spec, make_synthetic_feature_table
        >>> import habit.recipes as recipes
        >>> table = make_synthetic_feature_table(n_rows=60, n_features=8, rng=42)
        >>> spec = MLSpec(
        ...     name="demo",
        ...     table_preprocessors=(Spec("zscore"),),
        ...     classifier=Spec("LogisticRegression", {"max_iter": 500}),
        ...     metrics=(Spec("accuracy"),),
        ... )
        >>> fitted = recipes.train_model(table, spec, seed=42)
        >>> prediction = recipes.predict_model(fitted.pipeline, table)
        >>> len(prediction.predictions) == len(table.frame)
        True
    """
    started_at = _now()
    predictions = pipeline.predict(table)
    probabilities = (
        pipeline.predict_proba(table)
        if isinstance(pipeline.model, Classifier)
        else None
    )
    return PredictionResult(
        predictions=predictions,
        probabilities=probabilities,
        manifest=_prediction_manifest(pipeline, table, started_at),
    )


def cross_validate(
    table: FeatureTable,
    spec: MLSpec,
    *,
    n_splits: int = 5,
    seed: Optional[int] = None,
) -> CVResult:
    """
    Estimate generalisation with stratified K-fold cross-validation.

    Every fold builds a FRESH pipeline from the spec and fits it on that
    fold's training rows only, so preprocessing statistics and feature
    selection can never leak in from the validation rows. Folds are
    stratified on the outcome when the endpoint family has strata (binary,
    multiclass, survival).

    Args:
        table: Feature table with a declared outcome.
        spec: The modelling definition to evaluate.
        n_splits: Number of folds; must be at least 2.
        seed: Optional seed override, folded into the spec (and therefore
            into the fold shuffling, the component seeding, and the
            manifest) before anything runs.

    Returns:
        Per-fold panels, their across-fold summary, and the per-fold fitted
        pipelines.

    Raises:
        HABITAPIError: If the table declares no outcome, or ``n_splits``
            is below 2, or a stratum is smaller than ``n_splits``.

    Examples:
        >>> from habit import MLSpec, Spec, make_synthetic_feature_table
        >>> import habit.recipes as recipes
        >>> table = make_synthetic_feature_table(n_rows=60, n_features=8, rng=42)
        >>> spec = MLSpec(
        ...     name="demo",
        ...     table_preprocessors=(Spec("zscore"),),
        ...     classifier=Spec("LogisticRegression", {"max_iter": 500}),
        ...     metrics=(Spec("accuracy"), Spec("auc")),
        ... )
        >>> result = recipes.cross_validate(table, spec, n_splits=3, seed=42)
        >>> result.n_splits
        3
        >>> sorted(result.mean_metrics)
        ['accuracy', 'auc']
    """
    _require_outcome(table, owner="cross_validate")
    if n_splits < 2:
        raise HABITAPIError(
            f"cross_validate needs at least 2 folds; got n_splits={n_splits}."
        )
    effective = _effective_spec(spec, seed)
    started_at = _now()
    labels = stratify_labels(table.outcome, table.frame)
    metrics = build_ml_metrics(effective, default_names=_DEFAULT_METRIC_NAMES)
    fold_metrics: list[Mapping[str, float]] = []
    pipelines: list[TablePipeline] = []
    for train_index, validation_index in kfold_indices(
        len(table.frame),
        n_splits=n_splits,
        labels=labels,
        seed=effective.random_seed,
    ):
        pipeline = build_table_pipeline(effective)
        pipeline.fit(_select_rows(table, train_index))
        fold_metrics.append(
            pipeline.evaluate(_select_rows(table, validation_index), metrics)
        )
        pipelines.append(pipeline)
    names = tuple(fold_metrics[0])
    mean_metrics = {
        name: float(np.nanmean([fold[name] for fold in fold_metrics]))
        for name in names
    }
    std_metrics = {
        name: float(np.nanstd([fold[name] for fold in fold_metrics]))
        for name in names
    }
    return CVResult(
        fold_metrics=tuple(fold_metrics),
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        n_splits=n_splits,
        manifest=_manifest("cross_validate", effective, table, started_at),
        pipelines=tuple(pipelines),
    )


def _resolve_holdout(
    table: FeatureTable,
    spec: MLSpec,
    *,
    test_size: Optional[float],
    stratify: bool,
    train_ids: Optional[Sequence[str]],
    test_ids: Optional[Sequence[str]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Turn the call-site split arguments into (train, held-out) row indices.

    Args:
        table: The full table being split.
        spec: The effective spec; its seed drives the shuffled split.
        test_size: Held-out fraction for a random split, or ``None``.
        stratify: Stratify the random split on the outcome when possible.
        train_ids: Custom training-side row ids, or ``None``.
        test_ids: Custom held-out-side row ids, or ``None``.

    Returns:
        ``(train_indices, test_indices)`` row positions, or ``None`` when no
        split was requested.

    Raises:
        HABITAPIError: On contradictory arguments (``test_size`` together
            with id lists, or only one id list given).
    """
    if test_size is not None and (train_ids is not None or test_ids is not None):
        raise HABITAPIError(
            "train_model received both test_size and train/test id lists; "
            "a hold-out split is either random (test_size) or custom "
            "(train_ids + test_ids), never both."
        )
    if (train_ids is None) != (test_ids is None):
        raise HABITAPIError(
            "a custom hold-out split needs both train_ids and test_ids; "
            f"got train_ids={'given' if train_ids is not None else 'None'}, "
            f"test_ids={'given' if test_ids is not None else 'None'}."
        )
    if train_ids is not None and test_ids is not None:
        return _custom_holdout_indices(table, train_ids, test_ids)
    if test_size is None:
        return None
    labels = stratify_labels(table.outcome, table.frame) if stratify else None
    return train_test_indices(
        len(table.frame),
        test_size=test_size,
        labels=labels,
        seed=spec.random_seed,
    )


def _custom_holdout_indices(
    table: FeatureTable,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve custom id lists into (train, held-out) row indices.

    The id lists define the split exactly: a row not named by either side is
    left OUT of the fit, mirroring v0.1 where rows absent from both id files
    simply never entered training or evaluation.

    Args:
        table: The full table being split.
        train_ids: Row ids (as in :func:`_row_ids`) of the training side.
        test_ids: Row ids of the held-out side.

    Returns:
        ``(train_indices, test_indices)`` in the order the ids were given.

    Raises:
        HABITAPIError: If a side is empty, names a row the table does not
            have, or the two sides overlap.
    """
    if not train_ids or not test_ids:
        raise HABITAPIError(
            "a custom hold-out split needs non-empty train_ids and test_ids."
        )
    position = {row_id: index for index, row_id in enumerate(_row_ids(table))}
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise HABITAPIError(
            "custom hold-out train/test id lists overlap; a row cannot be "
            f"both fitted on and held out. Overlapping ids: {overlap[:5]}"
        )

    def _resolve(ids: Sequence[str], *, side: str) -> np.ndarray:
        missing = [row_id for row_id in ids if row_id not in position]
        if missing:
            raise HABITAPIError(
                f"custom hold-out {side}_ids name {len(missing)} row(s) the "
                f"table does not have: {missing[:5]}"
            )
        return np.asarray([position[row_id] for row_id in ids], dtype=int)

    return (
        _resolve(train_ids, side="train"),
        _resolve(test_ids, side="test"),
    )


def _prediction_manifest(
    pipeline: TablePipeline,
    table: FeatureTable,
    started_at: str,
) -> RunManifest:
    """
    Record an inference run.

    The recorded spec payload is the FITTED pipeline's composed spec -- the
    definition that actually produced the predictions -- and the row
    outcomes cover the table that was predicted.

    Args:
        pipeline: The fitted pipeline that produced the predictions.
        table: The table that was predicted.
        started_at: ISO-8601 timestamp taken before the run.

    Returns:
        The manifest for the prediction run.
    """
    spec = pipeline.spec
    provenance = Provenance(
        produced_by="recipes.modeling.predict_model",
        spec_fingerprint=spec.fingerprint(),
        inputs=(table.provenance,) if table.provenance is not None else (),
        software=software_fingerprint(),
        # The composed pipeline Spec carries no top-level seed; whatever seed
        # the components were fitted with is already baked into their specs.
        random_seed=None,
    )
    return RunManifest(
        spec_payload=spec.to_dict(),
        provenance=provenance,
        subject_outcomes={row_id: "success" for row_id in _row_ids(table)},
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


def _effective_spec(spec: MLSpec, seed: Optional[int]) -> MLSpec:
    """
    Fold a call-site seed into the spec.

    Args:
        spec: The declared modelling definition.
        seed: Overriding seed, or ``None`` to keep ``spec.random_seed``.

    Returns:
        The spec that will actually run -- the one recorded in the manifest,
        so the record never disagrees with the execution.
    """
    if seed is None:
        return spec
    return dataclasses.replace(spec, random_seed=int(seed))


def _require_outcome(table: FeatureTable, *, owner: str) -> None:
    """Fail loudly when a modelling recipe gets an unlabelled table."""
    if table.outcome is None:
        raise HABITAPIError(
            f"{owner} requires a FeatureTable with a declared outcome; "
            "an unlabelled table has nothing to fit or score against."
        )


def _select_rows(table: FeatureTable, indices: np.ndarray) -> FeatureTable:
    """
    Return the table restricted to the given row positions.

    ``FeatureTable`` deliberately has no row-subsetting verb of its own (the
    contract is column semantics, not frame algebra), so the fold split --
    the one place rows are ever partitioned -- lives here next to its only
    caller.

    Args:
        table: The table to subset.
        indices: Integer row positions to keep.

    Returns:
        A new table with the same column semantics; provenance derives from
        the input's when present.
    """
    frame = table.frame.iloc[np.asarray(indices)].reset_index(drop=True)
    provenance = None
    if table.provenance is not None:
        provenance = table.provenance.derive(
            produced_by="recipes.modeling.fold_split",
            spec_fingerprint="",
        )
    return FeatureTable(
        frame=frame,
        id_columns=table.id_columns,
        feature_columns=table.feature_columns,
        outcome=table.outcome,
        provenance=provenance,
    )


def _row_ids(table: FeatureTable) -> Tuple[str, ...]:
    """
    Return one string id per row, joining multi-column keys with ``|``.

    Args:
        table: The table whose identifier columns name the rows.

    Returns:
        Row ids in frame order.
    """
    id_columns = list(table.id_columns)
    if len(id_columns) == 1:
        return tuple(str(value) for value in table.frame[id_columns[0]])
    return tuple(
        "|".join(str(value) for value in row)
        for row in table.frame[id_columns].itertuples(index=False)
    )


def _manifest(
    recipe: str,
    spec: MLSpec,
    table: FeatureTable,
    started_at: str,
) -> RunManifest:
    """
    Record what actually ran.

    Args:
        recipe: Recipe name, e.g. ``"train_model"``.
        spec: The effective spec (seed overrides already folded in).
        table: The table the recipe ran on; its provenance becomes the
            manifest's input.
        started_at: ISO-8601 timestamp taken before the run.

    Returns:
        The manifest. Every row of the input table appears as a success:
        these recipes raise on failure, so a manifest exists only for a
        complete run.
    """
    provenance = Provenance(
        produced_by=f"recipes.modeling.{recipe}",
        spec_fingerprint=spec.fingerprint(),
        inputs=(table.provenance,) if table.provenance is not None else (),
        software=software_fingerprint(),
        random_seed=spec.random_seed,
    )
    return RunManifest(
        spec_payload=spec.to_dict(),
        provenance=provenance,
        subject_outcomes={row_id: "success" for row_id in _row_ids(table)},
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
