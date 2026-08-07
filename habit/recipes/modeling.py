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

Four recipes exist because four scientific acts exist:

* :func:`train_model` -- fit ONE pipeline on ONE table and score it. With no
  split arguments the score is the training-set readout of a final model;
  with ``test_size`` (random/stratified hold-out) or explicit
  ``train_ids``/``test_ids`` (custom hold-out, v0.1's id-file split) the
  pipeline is fitted on the training rows only and scored on the held-out
  rows as well, so the validation design -- which the ``MLSpec`` deliberately
  does not carry -- is stated at the call site that owns it.
* :func:`cross_validate` -- estimate generalisation by refitting the SAME
  definition on K stratified folds; every fold builds a fresh pipeline from
  the spec, so no fitted state can leak across folds. With ``inner_cv`` and
  a ``param_grid`` it becomes NESTED cross-validation: the hyperparameters
  are re-tuned inside every outer fold's training rows, so the outer score
  estimates the whole TUNING PROCEDURE rather than one lucky parameter set.
* :func:`search_hyperparameters` -- choose hyperparameters by K-fold search
  and write the winners back into the ``MLSpec``. Writing them back (rather
  than into a fitted object) is what keeps the provenance chain intact: the
  tuned definition is a spec like any other, with its own fingerprint, and
  it can be saved, re-run and published.
* :func:`predict_model` -- apply one FITTED pipeline to new rows. This is
  the inference half of the train/predict contract: preprocessing and
  selection replay their TRAINING state, never refit on the new data.

The search recipes assemble scikit-learn's own ``GridSearchCV`` /
``RandomizedSearchCV`` drivers around :class:`TablePipeline` (itself an
``sklearn.pipeline.Pipeline``); no search algorithm is implemented here, in
keeping with a recipe's job of wiring existing operators together.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.manifest import RunManifest
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.table import FeatureTable
from habit.domain.assembly import build_ml_metrics, build_table_pipeline
from habit.domain.pipeline import TablePipeline
from habit.domain.sklearn_interop import FrameToTable
from habit.domain.split import kfold_indices, stratify_labels, train_test_indices
from habit.domain.table_protocols import Classifier
from habit.spec.specs import MLSpec, Spec

__all__ = [
    "ModelResult",
    "CVResult",
    "PredictionResult",
    "SearchResult",
    "train_model",
    "cross_validate",
    "search_hyperparameters",
    "predict_model",
]

#: Metric panel a recipe falls back on when the spec declares none. Kept
#: deliberately small: the panel a paper reports belongs in the spec.
_DEFAULT_METRIC_NAMES: Tuple[str, ...] = ("accuracy", "auc")

#: Search strategies :func:`search_hyperparameters` accepts. ``"grid"`` is an
#: exhaustive ``GridSearchCV``, ``"random"`` an ``n_iter``-budget
#: ``RandomizedSearchCV``. Deliberately no third entry: a Bayesian /
#: evolutionary backend would be a new hard dependency, and HABIT's dependency
#: policy keeps those optional and explicit rather than smuggled in behind a
#: string.
_SEARCH_STRATEGIES: Tuple[str, ...] = ("grid", "random")

#: Objective metric used when neither the caller nor the spec names one. AUC
#: rather than accuracy: a search that maximises accuracy on the imbalanced
#: endpoints radiomics studies usually carry can win by predicting the
#: majority class. Stated here rather than inferred from the data, so that two
#: runs of the same spec always optimise the same quantity.
_DEFAULT_OBJECTIVE: str = "auc"

#: Type of a scikit-learn scorer callable: ``scorer(estimator, X, y)``.
Scorer = Callable[..., float]


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
            provenance chain, and every row's outcome. Under nested
            cross-validation the recorded spec is the UNTUNED declaration --
            the protocol being evaluated -- because there is no single tuned
            spec: each outer fold selected its own (see
            :attr:`fold_best_params`).
        fold_best_params: Winning hyperparameters of each outer fold's inner
            search, in fold order; empty for a plain (non-nested) run. Their
            SPREAD is the number a nested-CV report needs: parameters that
            change from fold to fold say the search is fitting noise.
    """

    fold_metrics: Tuple[Mapping[str, float], ...]
    mean_metrics: Mapping[str, float]
    std_metrics: Mapping[str, float]
    n_splits: int
    manifest: RunManifest
    pipelines: Tuple[TablePipeline, ...] = field(default=())
    fold_best_params: Tuple[Mapping[str, Any], ...] = field(default=())


@dataclass(frozen=True, eq=False)
class SearchResult:
    """
    Outcome of :func:`search_hyperparameters`, entirely in memory.

    Attributes:
        spec: The TUNED modelling definition -- the input spec with every
            searched parameter replaced by its winning value, in the same
            field layout it was declared in. This is the publishable
            artefact of a search: it fingerprints, serialises back to YAML
            and re-runs, so the tuning step never breaks the provenance
            chain the way a fitted-object-only result would.
        best_params: Winning parameters, keyed exactly as the grid was
            (``"model__component__C"``), so a caller can compare them against
            what it asked for without re-deriving the key syntax.
        best_score: Cross-validated score of the winning candidate, in the
            objective metric's OWN direction (higher is better for ``auc``,
            lower for ``mae``) -- never scikit-learn's internally negated
            form, which would silently flip the sign of a reported number.
        objective: Name of the metric that was optimised.
        trials: One record per evaluated candidate: ``params``,
            ``mean_score``, ``std_score`` (across the search folds) and
            ``rank``. This is the tuning table a methods section reports.
        manifest: Record of what ran, fingerprinting the TUNED spec.
        model: The final model, refitted on the whole table with
            :attr:`spec`; ``None`` when the caller asked for the tuned spec
            only (``refit=False``), which is what nested cross-validation
            does since it refits on its own outer training rows.
    """

    spec: MLSpec
    best_params: Mapping[str, Any]
    best_score: float
    objective: str
    trials: Tuple[Mapping[str, Any], ...]
    manifest: RunManifest
    model: Optional[ModelResult] = None


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
        ...     steps=(Spec("zscore"),),
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
        ...     steps=(Spec("zscore"),),
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
    inner_cv: Optional[int] = None,
    param_grid: Optional[Union[Mapping[str, Sequence[Any]], Sequence[Mapping[str, Sequence[Any]]]]] = None,
    strategy: str = "grid",
    n_iter: int = 10,
    objective: Optional[str] = None,
) -> CVResult:
    """
    Estimate generalisation with stratified K-fold cross-validation.

    Every fold builds a FRESH pipeline from the spec and fits it on that
    fold's training rows only, so preprocessing statistics and feature
    selection can never leak in from the validation rows. Folds are
    stratified on the outcome when the endpoint family has strata (binary,
    multiclass, survival).

    **Nested cross-validation.** Passing ``inner_cv`` together with a
    ``param_grid`` re-tunes the hyperparameters inside every outer fold, on
    that fold's TRAINING rows only, and scores the winner on the untouched
    validation rows. The reported panel then estimates the whole tuning
    PROCEDURE, which is the quantity a reviewer asks for: tuning once on all
    the data and cross-validating afterwards reuses the validation rows for
    selection and reports an optimistically biased number. The two arguments
    are required together -- a grid without ``inner_cv`` would have to tune
    on the outer validation rows, and ``inner_cv`` without a grid would tune
    nothing.

    Args:
        table: Feature table with a declared outcome.
        spec: The modelling definition to evaluate. Under nested CV this is
            the definition MINUS the tuned parameters: each outer fold
            overwrites the searched ones with its own winners.
        n_splits: Number of OUTER folds; must be at least 2.
        seed: Optional seed override, folded into the spec (and therefore
            into the fold shuffling, the component seeding, and the
            manifest) before anything runs.
        inner_cv: Number of INNER folds used to tune inside each outer
            fold's training rows; must be at least 2. ``None`` (the default)
            runs plain cross-validation with no tuning.
        param_grid: The search space, in the key syntax
            :func:`search_hyperparameters` documents.
        strategy: ``"grid"`` or ``"random"``; see
            :func:`search_hyperparameters`.
        n_iter: Candidate budget for ``strategy="random"``.
        objective: Registered metric name the inner search maximises (or
            minimises, honouring the metric's own direction); see
            :func:`search_hyperparameters`.

    Returns:
        Per-fold panels, their across-fold summary, the per-fold fitted
        pipelines, and -- under nested CV -- each fold's winning parameters.

    Raises:
        HABITAPIError: If the table declares no outcome, ``n_splits`` or
            ``inner_cv`` is below 2, a stratum is smaller than the fold
            count, or exactly one of ``inner_cv`` / ``param_grid`` is given.

    Examples:
        >>> from habit import MLSpec, Spec, make_synthetic_feature_table
        >>> import habit.recipes as recipes
        >>> table = make_synthetic_feature_table(n_rows=60, n_features=8, rng=42)
        >>> spec = MLSpec(
        ...     name="demo",
        ...     steps=(Spec("zscore"),),
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
    _require_paired_nesting(inner_cv, param_grid)
    effective = _effective_spec(spec, seed)
    started_at = _now()
    metrics = build_ml_metrics(effective, default_names=_DEFAULT_METRIC_NAMES)
    fold_metrics: List[Mapping[str, float]] = []
    pipelines: List[TablePipeline] = []
    fold_best_params: List[Mapping[str, Any]] = []
    for train_index, validation_index in _fold_pairs(
        table,
        n_splits=n_splits,
        seed=effective.random_seed,
        owner="cross_validate",
    ):
        fit_table = _select_rows(table, train_index)
        fold_spec = effective
        if inner_cv is not None:
            # The inner search sees this fold's TRAINING rows only, which is
            # the whole point of nesting: the validation rows below are
            # untouched by both the tuning and the fitting.
            tuned = search_hyperparameters(
                fit_table,
                effective,
                param_grid,  # type: ignore[arg-type]
                n_splits=inner_cv,
                strategy=strategy,
                n_iter=n_iter,
                objective=objective,
                refit=False,
            )
            fold_spec = tuned.spec
            fold_best_params.append(tuned.best_params)
        pipeline = build_table_pipeline(fold_spec)
        pipeline.fit(fit_table)
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
        fold_best_params=tuple(fold_best_params),
    )


def search_hyperparameters(
    table: FeatureTable,
    spec: MLSpec,
    param_grid: Union[
        Mapping[str, Sequence[Any]], Sequence[Mapping[str, Sequence[Any]]]
    ],
    *,
    n_splits: int = 5,
    seed: Optional[int] = None,
    strategy: str = "grid",
    n_iter: int = 10,
    objective: Optional[str] = None,
    refit: bool = True,
) -> SearchResult:
    """
    Tune hyperparameters by K-fold search and write the winners into the spec.

    The search itself is scikit-learn's (``GridSearchCV`` /
    ``RandomizedSearchCV``) driving a :class:`TablePipeline`, which IS an
    ``sklearn.pipeline.Pipeline``; nothing about the search is reimplemented
    here. What this recipe adds is the two things a study needs and sklearn
    does not provide:

    * **The folds are HABIT's own.** They come from
      :func:`habit.domain.split.kfold_indices` with the spec's seed, so a
      search over K folds partitions the rows exactly as
      :func:`cross_validate` would with the same ``n_splits`` and seed. Each
      candidate is therefore fitted on training rows only -- preprocessing
      statistics and feature selection included, since they are steps of the
      pipeline being cloned per fold, never precomputed on all the rows.
    * **The winners land in the ``MLSpec``.** A tuned model is a DEFINITION,
      not just a fitted object: written back into the spec it keeps its
      fingerprint, serialises to YAML, and can be re-run by someone else.
      A search that returned only ``best_estimator_`` would end the
      provenance chain at the point the parameters were chosen.

    **Grid key syntax.** Keys address one HABIT component parameter through
    the pipeline's step names: ``"<step>__component__<parameter>"``. The
    terminal model's step is called ``"model"`` and every transformation step
    is named after its registered spec name, so
    ``{"model__component__C": [0.1, 1, 10],
    "variance__component__threshold": [0.0, 0.01]}`` tunes the classifier's
    regularisation and the variance filter's threshold. Keys of any other
    shape are rejected up front rather than searched: this recipe can only
    write a value back into the spec if it knows which component and which
    parameter it belongs to, and a search whose result cannot be recorded is
    worse than no search.

    **The objective.** ``objective`` names a registered HABIT metric (not an
    sklearn scorer string), and the metric's own ``greater_is_better`` decides
    the direction, so a "lower is better" metric is minimised without the
    caller negating anything. Omitted, the objective is the FIRST metric of
    ``spec.metrics``, and for a spec with no metric panel it is ``auc``.
    Scoring goes through :meth:`TablePipeline.evaluate`, so the search
    optimises exactly the quantity the final report prints, in the same
    vocabulary.

    Args:
        table: Feature table with a declared outcome; the rows the search
            partitions into folds.
        spec: The modelling definition to tune. Everything not named in the
            grid is left exactly as declared.
        param_grid: One mapping of key to candidate values, or a sequence of
            such mappings (sklearn's disjoint-grids form). For
            ``strategy="random"`` a value may also be a scipy distribution.
        n_splits: Number of search folds; must be at least 2.
        seed: Optional seed override, folded into the spec before anything
            runs, and therefore driving the fold shuffling, the component
            seeding and the random search's own sampling.
        strategy: ``"grid"`` for an exhaustive search over the product of
            the candidate lists, ``"random"`` for ``n_iter`` sampled
            candidates.
        n_iter: Candidate budget for ``strategy="random"``; ignored by
            ``"grid"``, where the budget is the grid.
        objective: Registered metric name to optimise, or ``None`` for the
            fallback chain described above.
        refit: Fit the tuned spec on the whole table and return the result
            in :attr:`SearchResult.model`. ``False`` returns the tuned spec
            only, which is what nested cross-validation needs (it refits on
            its own outer training rows).

    Returns:
        The tuned spec, the winning parameters and score, the per-candidate
        trial table, and -- unless ``refit=False`` -- the final model.

    Raises:
        HABITAPIError: If the table declares no outcome, ``n_splits`` is
            below 2, the strategy is unknown, the grid is empty, or a grid
            key does not address a tunable component parameter of this
            pipeline.

    Examples:
        >>> from habit import MLSpec, Spec, make_synthetic_feature_table
        >>> import habit.recipes as recipes
        >>> table = make_synthetic_feature_table(n_rows=60, n_features=8, rng=42)
        >>> spec = MLSpec(
        ...     name="demo",
        ...     steps=(Spec("zscore"),),
        ...     classifier=Spec("LogisticRegression", {"max_iter": 500}),
        ...     metrics=(Spec("auc"),),
        ... )
        >>> tuned = recipes.search_hyperparameters(
        ...     table, spec, {"model__component__C": [0.01, 1.0]},
        ...     n_splits=3, seed=42,
        ... )
        >>> tuned.spec.classifier.params["C"] in (0.01, 1.0)
        True
        >>> tuned.objective
        'auc'
    """
    _require_outcome(table, owner="search_hyperparameters")
    if strategy not in _SEARCH_STRATEGIES:
        raise HABITAPIError(
            f"search_hyperparameters does not know strategy {strategy!r}; "
            f"supported strategies are {list(_SEARCH_STRATEGIES)}."
        )
    grids = _param_grid_mappings(param_grid)
    effective = _effective_spec(spec, seed)
    started_at = _now()
    pipeline = _searchable_pipeline(effective, table)
    targets = _resolve_param_targets(grids, pipeline)
    metric = _objective_metric(effective, objective)
    greater_is_better = bool(metric.greater_is_better)
    folds = _fold_pairs(
        table,
        n_splits=n_splits,
        seed=effective.random_seed,
        owner="search_hyperparameters",
    )
    search = _search_driver(
        pipeline,
        param_grid,
        strategy=strategy,
        n_iter=n_iter,
        scorer=_objective_scorer(metric, greater_is_better),
        folds=folds,
        seed=effective.random_seed,
    )
    # ``y=None``: the outcome rides inside the FeatureTable that the
    # pipeline's FrameToTable head rebuilds from each row slice, and the
    # scorer reads it from there. Handing sklearn a separate label vector
    # would create a second source of truth for the same column.
    search.fit(table.frame, None)
    tuned = _spec_with_best_params(effective, targets, search.best_params_)
    best_score = float(search.best_score_)
    return SearchResult(
        spec=tuned,
        best_params=dict(search.best_params_),
        best_score=best_score if greater_is_better else -best_score,
        objective=str(metric.spec.name),
        trials=_search_trials(search.cv_results_, greater_is_better),
        manifest=_manifest("search_hyperparameters", tuned, table, started_at),
        model=train_model(table, tuned) if refit else None,
    )


def _require_paired_nesting(
    inner_cv: Optional[int],
    param_grid: Optional[Any],
) -> None:
    """
    Reject a half-declared nested cross-validation.

    Args:
        inner_cv: The inner fold count as given.
        param_grid: The search space as given.

    Raises:
        HABITAPIError: If exactly one of the two is given. Each half alone
            is a silent mistake rather than a harmless one: a grid without
            ``inner_cv`` has nowhere to tune except the outer validation
            rows (leakage, reported as a good score), and ``inner_cv``
            without a grid describes a search over nothing.
    """
    if (inner_cv is None) == (param_grid is None):
        return
    if param_grid is None:
        raise HABITAPIError(
            f"cross_validate received inner_cv={inner_cv} but no param_grid, "
            "so the inner folds would search over nothing. Pass the grid to "
            "tune, or drop inner_cv for plain cross-validation."
        )
    raise HABITAPIError(
        "cross_validate received a param_grid but no inner_cv. Tuning "
        "without an inner split would select hyperparameters on the same "
        "rows the outer fold scores, which reports an optimistically biased "
        "number. Pass inner_cv=<folds> for nested cross-validation, or tune "
        "separately with search_hyperparameters."
    )


def _fold_pairs(
    table: FeatureTable,
    *,
    n_splits: int,
    seed: Optional[int],
    owner: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Materialise the (train, validation) fold indices of one K-fold split.

    The single fold-generating call site of this module, so a search and the
    cross-validation it is nested in partition the rows identically for the
    same ``n_splits`` and seed. Folds are stratified on the outcome whenever
    the endpoint family has strata.

    Args:
        table: The table whose rows are partitioned.
        n_splits: Number of folds.
        seed: Seed driving the shuffle.
        owner: Recipe name used in the error message.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: One ``(train, validation)``
        index pair per fold, in fold order. Materialised (rather than left
        as the generator) because sklearn's search drivers iterate the fold
        list once per candidate.

    Raises:
        HABITAPIError: If ``n_splits`` is below 2, or a stratum is smaller
            than ``n_splits``.
    """
    if n_splits < 2:
        raise HABITAPIError(
            f"{owner} needs at least 2 folds; got n_splits={n_splits}."
        )
    labels = stratify_labels(table.outcome, table.frame)
    return list(
        kfold_indices(
            len(table.frame),
            n_splits=n_splits,
            labels=labels,
            seed=seed,
        )
    )


def _searchable_pipeline(spec: MLSpec, table: FeatureTable) -> TablePipeline:
    """
    Build the spec's pipeline with a frame schema an sklearn driver can slice.

    A search driver slices ``X`` by row, and a ``FeatureTable`` is a frozen
    dataclass with column semantics rather than a row-indexable container, so
    ``X`` is the plain frame and the pipeline's
    :class:`~habit.domain.sklearn_interop.FrameToTable` head rebuilds the
    table for each slice. The schema is read off ``table``, so it cannot
    disagree with the data being searched.

    Args:
        spec: The effective modelling definition.
        table: The table being searched; only its column schema is read.

    Returns:
        TablePipeline: The unfitted pipeline handed to the search driver,
        which clones it once per candidate and fold.
    """
    pipeline = build_table_pipeline(spec)
    head_name = pipeline.steps[0][0]
    pipeline.set_params(**{head_name: FrameToTable.from_table(table)})
    return pipeline


def _param_grid_mappings(
    param_grid: Any,
) -> Tuple[Mapping[str, Any], ...]:
    """
    Normalise a grid argument into a tuple of mappings.

    Args:
        param_grid: One mapping of key to candidates, or a sequence of such
            mappings (sklearn's disjoint-grids form).

    Returns:
        Tuple[Mapping[str, Any], ...]: The grids, for key validation. The
        ORIGINAL argument is what reaches sklearn, so nothing here can
        change the search space.

    Raises:
        HABITAPIError: If the grid is empty or malformed.
    """
    grids: Tuple[Any, ...]
    if isinstance(param_grid, Mapping):
        grids = (param_grid,)
    elif isinstance(param_grid, Sequence) and not isinstance(param_grid, str):
        grids = tuple(param_grid)
    else:
        raise HABITAPIError(
            "param_grid must be a mapping of parameter key to candidate "
            "values, or a sequence of such mappings; got "
            f"{type(param_grid).__name__}."
        )
    if not grids:
        raise HABITAPIError(
            "param_grid is empty, so there is nothing to search. Drop the "
            "search, or name at least one parameter."
        )
    for grid in grids:
        if not isinstance(grid, Mapping) or not grid:
            raise HABITAPIError(
                "Every param_grid entry must be a non-empty mapping of "
                f"parameter key to candidate values; got {grid!r}."
            )
    return grids


def _resolve_param_targets(
    grids: Sequence[Mapping[str, Any]],
    pipeline: TablePipeline,
) -> Dict[str, Tuple[str, int, str]]:
    """
    Map every grid key onto the spec location its winning value belongs in.

    Resolved BEFORE the search rather than after it: a key this recipe
    cannot write back into the spec would otherwise be discovered only once
    the (possibly long) search had finished, and the result would be a set of
    winning parameters with nowhere to record them.

    Args:
        grids: The normalised grids.
        pipeline: The pipeline the search will drive, which is what defines
            the step names a key may address.

    Returns:
        Dict[str, Tuple[str, int, str]]: Key to
        ``(kind, index, parameter)``, where ``kind`` is ``"classifier"``
        (``index`` unused) or ``"step"`` (``index`` positions the step in
        ``spec.steps``).

    Raises:
        HABITAPIError: On a key that is not
            ``"<step>__component__<parameter>"``, or names a step this
            pipeline does not have.
    """
    model_step = pipeline.steps[-1][0]
    # Step i of ``spec.steps`` is step i of this slice by construction: the
    # pipeline is built as [FrameToTable, *components, model].
    step_index = {
        name: index for index, (name, _) in enumerate(pipeline.steps[1:-1])
    }
    tunable = [model_step, *step_index]
    return {
        str(key): _param_target(str(key), model_step, step_index, tunable)
        for grid in grids
        for key in grid
    }


def _param_target(
    key: str,
    model_step: str,
    step_index: Mapping[str, int],
    tunable: Sequence[str],
) -> Tuple[str, int, str]:
    """
    Resolve one grid key into a spec location.

    Args:
        key: The grid key, e.g. ``"model__component__C"``.
        model_step: Step name of the terminal outcome model.
        step_index: Transformation step name to ``spec.steps`` position.
        tunable: Step names to list in error messages.

    Returns:
        Tuple[str, int, str]: ``(kind, index, parameter)``.

    Raises:
        HABITAPIError: On an unaddressable key.
    """
    parts = key.split("__")
    if len(parts) != 3 or parts[1] != "component" or not parts[2]:
        raise HABITAPIError(
            f"search_hyperparameters cannot write the searched parameter "
            f"{key!r} back into the MLSpec. A grid key must read "
            "'<step>__component__<parameter>', e.g. 'model__component__C' or "
            "'variance__component__threshold' -- the 'component' segment is "
            "the HABIT component inside the pipeline's sklearn adapter. "
            f"Tunable steps of this pipeline: {list(tunable)}."
        )
    step, _, parameter = parts
    if step == model_step:
        return ("classifier", -1, parameter)
    if step in step_index:
        return ("step", int(step_index[step]), parameter)
    raise HABITAPIError(
        f"Grid key {key!r} addresses step {step!r}, which this pipeline does "
        f"not have. Tunable steps: {list(tunable)}. Step names are the "
        "components' registered spec names, in declaration order."
    )


def _objective_metric(spec: MLSpec, objective: Optional[str]) -> Any:
    """
    Build the single metric a search optimises.

    Resolved against the classification metric registry, which is the family
    an ``MLSpec`` can declare: its terminal model is a ``classifier`` spec,
    assembled through
    :class:`~habit.domain.classification.ClassifierRegistry`. Scoring itself
    goes through :meth:`TablePipeline.evaluate` and so stays family-agnostic;
    when ``MLSpec`` grows regression and survival terminals, this resolution
    is the one place that has to follow.

    Args:
        spec: The effective spec, whose metric panel supplies the default.
        objective: Registered metric name, or ``None`` for the fallback
            chain: first declared metric, else :data:`_DEFAULT_OBJECTIVE`.

    Returns:
        The metric instance. Its ``greater_is_better`` flag is what decides
        the optimisation direction.

    Raises:
        ComponentNotFoundError: If the name is not a registered metric.
        ConfigurationError: If the metric's parameters fail validation.
    """
    # Lazy import: an L4 recipe may read L3 registries, but importing the
    # evaluation package at module level would pull sklearn and the metric
    # stack into every ``import habit.recipes``.
    from habit.domain.evaluation import MetricRegistry

    if objective is None and spec.metrics:
        # The panel a paper reports is the panel it tuned on, parameters
        # included; taking the whole first entry keeps them together.
        entry = spec.metrics[0]
        return MetricRegistry.create(entry.name, **entry.params)
    return MetricRegistry.create(
        objective if objective is not None else _DEFAULT_OBJECTIVE
    )


def _objective_scorer(metric: Any, greater_is_better: bool) -> Scorer:
    """
    Wrap one HABIT metric as a scikit-learn scorer.

    Scoring goes through :meth:`TablePipeline.evaluate`, which already
    dispatches on the endpoint family (probabilities for a classification
    metric, risk or survival curves for a survival metric), so the search
    optimises exactly the number the final report prints. sklearn always
    MAXIMISES a scorer, hence the sign flip for metrics whose own direction
    is "lower is better".

    Args:
        metric: The metric instance to score with.
        greater_is_better: The metric's own direction.

    Returns:
        Scorer: A callable ``scorer(estimator, X, y=None) -> float``.
    """
    name = str(metric.spec.name)

    def score(estimator: TablePipeline, X: Any, y: Any = None) -> float:
        """Score one fitted candidate on one validation slice."""
        # The head step holds the column schema, so the row slice sklearn
        # handed us becomes the same FeatureTable the pipeline was fitted on
        # the training half of -- outcome column included.
        table = estimator.frame_schema.transform(X)
        value = float(estimator.evaluate(table, (metric,))[name])
        return value if greater_is_better else -value

    return score


def _search_driver(
    pipeline: TablePipeline,
    param_grid: Any,
    *,
    strategy: str,
    n_iter: int,
    scorer: Scorer,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    seed: Optional[int],
) -> Any:
    """
    Configure the scikit-learn search driver for one strategy.

    Args:
        pipeline: The unfitted pipeline to clone per candidate.
        param_grid: The search space, exactly as the caller gave it.
        strategy: ``"grid"`` or ``"random"``.
        n_iter: Candidate budget for ``"random"``.
        scorer: The objective, already sign-corrected for maximisation.
        folds: Explicit ``(train, validation)`` index pairs, so the search
            uses HABIT's fold layout rather than sklearn's own splitter.
        seed: Seed for the random strategy's sampling.

    Returns:
        The configured (unfitted) search object.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    shared: Dict[str, Any] = {
        "scoring": scorer,
        "cv": list(folds),
        # The tuned parameters are refitted through the SPEC (see
        # ``search_hyperparameters``), not through sklearn's own
        # ``best_estimator_``, so that the final model is built by the same
        # assembly path as any other HABIT model.
        "refit": False,
        # A candidate that raises is a bug or an invalid grid value, not a
        # data point: scoring it NaN would let the search quietly return the
        # best of the candidates that happened to work.
        "error_score": "raise",
    }
    if strategy == "grid":
        return GridSearchCV(pipeline, param_grid, **shared)
    return RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=n_iter,
        random_state=seed,
        **shared,
    )


def _search_trials(
    cv_results: Mapping[str, Any],
    greater_is_better: bool,
) -> Tuple[Mapping[str, Any], ...]:
    """
    Turn sklearn's ``cv_results_`` into the per-candidate trial table.

    Args:
        cv_results: The driver's ``cv_results_`` mapping.
        greater_is_better: The objective's own direction, used to undo the
            sign flip the scorer applied, so a reported ``mean_score`` is
            always the metric's own value.

    Returns:
        Tuple[Mapping[str, Any], ...]: One record per candidate, in the
        driver's own candidate order.
    """
    sign = 1.0 if greater_is_better else -1.0
    return tuple(
        {
            "params": dict(params),
            "mean_score": sign * float(mean),
            "std_score": float(std),
            "rank": int(rank),
        }
        for params, mean, std, rank in zip(
            cv_results["params"],
            cv_results["mean_test_score"],
            cv_results["std_test_score"],
            cv_results["rank_test_score"],
        )
    )


def _spec_with_best_params(
    spec: MLSpec,
    targets: Mapping[str, Tuple[str, int, str]],
    best_params: Mapping[str, Any],
) -> MLSpec:
    """
    Return ``spec`` with every searched parameter set to its winning value.

    Only the searched parameters move; every other declaration -- component
    names, step order, untouched parameters, the metric panel, the seed -- is
    carried over verbatim, so the tuned spec differs from the declared one in
    exactly the places the caller asked to tune.

    Args:
        spec: The effective (untuned) spec.
        targets: Grid key to spec location, from
            :func:`_resolve_param_targets`.
        best_params: The driver's winning parameters.

    Returns:
        MLSpec: The tuned definition, in the same field layout as ``spec``.
    """
    step_params = [dict(entry.params) for entry in spec.steps]
    classifier_params = dict(spec.classifier.params)
    for key, value in best_params.items():
        kind, index, parameter = targets[str(key)]
        plain = _plain_param_value(value)
        if kind == "classifier":
            classifier_params[parameter] = plain
        else:
            step_params[index][parameter] = plain
    steps = tuple(
        Spec(name=entry.name, params=params, version=entry.version)
        for entry, params in zip(spec.steps, step_params)
    )
    classifier = Spec(
        name=spec.classifier.name,
        params=classifier_params,
        version=spec.classifier.version,
    )
    return _spec_with_layout(spec, steps, classifier)


def _spec_with_layout(
    spec: MLSpec,
    steps: Tuple[Spec, ...],
    classifier: Spec,
) -> MLSpec:
    """
    Rebuild a spec's table steps in the FIELD LAYOUT it was declared in.

    ``MLSpec`` serialises the deprecated three-chain layout and the single
    ordered ``steps`` layout differently on purpose -- that asymmetry is what
    keeps the fingerprint of every already-published analysis stable. A tuned
    spec must therefore keep its predecessor's layout, or the same analysis
    would fingerprint differently before and after tuning for a reason that
    has nothing to do with the tuning. ``dataclasses.replace`` cannot be used
    for this: it re-supplies the derived ``steps`` alongside the chains, and
    changing one without the other is exactly what ``MLSpec`` rejects.

    Args:
        spec: The spec whose layout is being preserved.
        steps: The full ordered step list to install.
        classifier: The (possibly tuned) terminal classifier spec.

    Returns:
        MLSpec: The rebuilt spec.
    """
    common: Dict[str, Any] = {
        "name": spec.name,
        "classifier": classifier,
        "metrics": spec.metrics,
        "random_seed": spec.random_seed,
        "version": spec.version,
    }
    if not spec.declares_deprecated_chains:
        return MLSpec(steps=steps, **common)
    n_pre = len(spec.pre_preprocessing_feature_selectors)
    n_preprocessors = len(spec.table_preprocessors)
    return MLSpec(
        pre_preprocessing_feature_selectors=steps[:n_pre],
        table_preprocessors=steps[n_pre : n_pre + n_preprocessors],
        feature_selectors=steps[n_pre + n_preprocessors :],
        **common,
    )


def _plain_param_value(value: Any) -> Any:
    """
    Coerce a searched value into something a ``Spec`` can serialise.

    A randomised search samples from scipy distributions, which yield numpy
    scalars; left as they are, they would reach YAML as opaque objects and
    the tuned spec would not round-trip. Everything else is passed through
    untouched, so no component ever sees a value the caller did not offer.

    Args:
        value: One winning parameter value.

    Returns:
        The value as a plain Python object.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain_param_value(item) for item in value.tolist()]
    return value


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
