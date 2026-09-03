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
"""Tests for TablePipeline: the train/predict-consistent tabular flow."""

from __future__ import annotations

import json
import pickle
import zipfile
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline

from habit.api.exceptions import CompatibilityError, HABITAPIError
from habit.contracts import BinaryOutcome, FeatureTable
from habit.pipeline.assembly import build_table_pipeline
from habit.classification import LogisticRegressionClassifier, RandomForestClassifier
from habit.evaluation import AccuracyMetric, AucMetric, HosmerLemeshowPValueMetric
from habit.feature_selection import FeatureSelectorRegistry, IccSelector, VarianceSelector
from habit.pipeline import TablePipeline
from habit.pipeline.sklearn_interop import FrameToTable, as_transformer
from habit.table_preprocessing import TablePreprocessorRegistry, ZScorePreprocessor
from habit.exceptions import ComponentNotFoundError
from habit.spec import MLSpec, Spec

from .conftest import make_feature_table


def _pipeline() -> TablePipeline:
    """A variance-then-zscore chain ending in logistic regression."""
    return TablePipeline(
        steps=[VarianceSelector(threshold=0.01), ZScorePreprocessor()],
        classifier=LogisticRegressionClassifier(max_iter=500),
    )


@pytest.mark.unit
def test_fit_transform_predict_roundtrip() -> None:
    """The fitted chain selects, scales and predicts new subjects."""
    train = make_feature_table(seed=1)
    pipeline = _pipeline().fit(train)
    new = make_feature_table(tuple(f"N{i}" for i in range(6)), seed=2)
    labels = pipeline.predict(new)
    assert list(labels.index) == list(new.frame["subject"])
    probabilities = pipeline.predict_proba(new)
    assert probabilities.shape == (6, 2)
    # transform exposes the classifier-ready table directly.
    transformed = pipeline.transform(new)
    assert transformed.feature_columns == pipeline.transform(train).feature_columns


@pytest.mark.unit
def test_prediction_uses_training_statistics() -> None:
    """New tables are scaled with TRAINING mean/std (no leakage)."""
    train = make_feature_table(seed=3)
    pipeline = TablePipeline(
        steps=[ZScorePreprocessor()], classifier=LogisticRegressionClassifier()
    ).fit(train)
    new = make_feature_table(["X1", "X2"], seed=4)
    transformed = pipeline.transform(new)
    train_block = train.frame[list(train.feature_columns)]
    expected = (new.frame[list(train.feature_columns)] - train_block.mean()) / train_block.std()
    np.testing.assert_allclose(
        transformed.frame[list(transformed.feature_columns)].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
    )


@pytest.mark.unit
def test_unfitted_pipeline_and_missing_classifier_raise() -> None:
    """transform/predict before fit, and a missing classifier, are errors."""
    with pytest.raises(HABITAPIError):
        TablePipeline(steps=[], classifier=None)  # type: ignore[arg-type]
    pipeline = _pipeline()
    with pytest.raises(HABITAPIError):
        pipeline.transform(make_feature_table())
    with pytest.raises(HABITAPIError):
        pipeline.predict(make_feature_table())


@pytest.mark.unit
def test_evaluate_scores_label_and_probability_metrics() -> None:
    """evaluate returns one value per metric, keyed by the spec name."""
    train = make_feature_table(seed=5)
    pipeline = _pipeline().fit(train)
    results = pipeline.evaluate(
        train, [AccuracyMetric(), AucMetric(), HosmerLemeshowPValueMetric(n_groups=4)]
    )
    assert set(results) == {"accuracy", "auc", "hosmer_lemeshow_p_value"}
    assert results["accuracy"] >= 0.9
    assert 0.0 <= results["auc"] <= 1.0
    assert 0.0 <= results["hosmer_lemeshow_p_value"] <= 1.0


@pytest.mark.unit
def test_evaluate_requires_metrics_and_outcome() -> None:
    """An empty metric list or an outcome-less table is a clear error."""
    pipeline = _pipeline().fit(make_feature_table(seed=6))
    with pytest.raises(HABITAPIError):
        pipeline.evaluate(make_feature_table(seed=7), [])
    with pytest.raises(HABITAPIError):
        pipeline.evaluate(make_feature_table(seed=7, outcome=False), [AccuracyMetric()])


@pytest.mark.unit
def test_set_random_state_propagates_to_seedable_components() -> None:
    """One seeding call reaches every Seedable component of the pipeline."""
    train = make_feature_table(seed=8)
    first = TablePipeline(steps=[], classifier=RandomForestClassifier())
    second = TablePipeline(steps=[], classifier=RandomForestClassifier())
    first.set_random_state(17)
    second.set_random_state(17)
    np.testing.assert_array_equal(
        first.fit(train).predict(train).to_numpy(),
        second.fit(train).predict(train).to_numpy(),
    )


@pytest.mark.unit
def test_spec_describes_every_stage() -> None:
    """The composed spec lists the step specs and the model spec."""
    pipeline = _pipeline()
    spec = pipeline.spec
    assert spec.name == "table_pipeline"
    assert [s["name"] for s in spec.params["steps"]] == ["variance", "zscore"]
    assert spec.params["model"]["name"] == "LogisticRegression"


@pytest.mark.unit
def test_save_load_roundtrip_preserves_predictions(tmp_path) -> None:
    """A saved pipeline reloads with identical fitted behaviour."""
    train = make_feature_table(seed=9)
    pipeline = _pipeline().fit(train)
    destination = pipeline.save(tmp_path / "model.habitpipeline")
    with zipfile.ZipFile(destination, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert manifest["format"] == "habit.tablepipeline"
    assert manifest["format_version"] == 3
    assert manifest["is_fitted"] is True
    assert [record["spec"]["name"] for record in manifest["steps"]] == [
        "variance",
        "zscore",
    ]
    new = make_feature_table(tuple(f"N{i}" for i in range(5)), seed=10)
    loaded = TablePipeline.load(destination)
    np.testing.assert_allclose(
        loaded.predict_proba(new).to_numpy(), pipeline.predict_proba(new).to_numpy()
    )


@pytest.mark.unit
def test_load_rejects_foreign_and_newer_files(tmp_path) -> None:
    """Non-pipeline files and newer format versions raise CompatibilityError."""
    foreign = tmp_path / "foreign.zip"
    with zipfile.ZipFile(foreign, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"format": "other"}))
    with pytest.raises(CompatibilityError):
        TablePipeline.load(foreign)

    train = make_feature_table(seed=11)
    pipeline = _pipeline().fit(train)
    destination = pipeline.save(tmp_path / "newer.habitpipeline")
    with zipfile.ZipFile(destination, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        payload = archive.read("payload.pkl")
    manifest["format_version"] = 999
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("payload.pkl", payload)
    with pytest.raises(CompatibilityError):
        TablePipeline.load(destination)


@pytest.mark.unit
def test_load_rejects_internally_inconsistent_archives(tmp_path) -> None:
    """A manifest disagreeing with its payload signals archive corruption."""
    train = make_feature_table(seed=12)
    pipeline = _pipeline().fit(train)
    destination = pipeline.save(tmp_path / "tampered.habitpipeline")
    with zipfile.ZipFile(destination, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        payload = archive.read("payload.pkl")
    manifest["steps"] = [dict(record, spec=Spec(name="fake").to_dict()) for record in manifest["steps"]]
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("payload.pkl", payload)
    with pytest.raises(CompatibilityError):
        TablePipeline.load(destination)


@pytest.mark.unit
def test_repeat_tables_reach_icc_step_only() -> None:
    """Steps accepting repeat_tables receive them; the rest see the primary."""
    ids = tuple(f"S{i}" for i in range(16))
    primary = make_feature_table(ids, n_noise=1, seed=40)
    rng = np.random.RandomState(41)
    repeat = make_feature_table(ids, n_noise=1, seed=42)
    repeat.frame["signal"] = primary.frame["signal"] + rng.normal(scale=0.01, size=16)
    repeat.frame["noise0"] = rng.normal(size=16)
    pipeline = TablePipeline(
        steps=[IccSelector(threshold=0.75)],
        classifier=LogisticRegressionClassifier(),
    ).fit(primary, repeat_tables=[repeat])
    # The unstable noise column was filtered at fit time, so it is dropped
    # from prediction tables too.
    transformed = pipeline.transform(primary)
    assert "signal" in transformed.feature_columns
    assert "noise0" not in transformed.feature_columns


# ---------------------------------------------------------------------------
# Pre-preprocessing selection stage (v0.1's ``before_z_score`` stage)
# ---------------------------------------------------------------------------


def _staged_variance_table(
    subject_ids: Sequence[str],
    *,
    high_var_scale: float = 4.0,
) -> FeatureTable:
    """
    Build a deterministic table with known per-column raw variances.

    ``low_a``/``low_b`` alternate between two nearby levels (variance well
    below 0.5); ``high_var`` is the same alternation scaled by
    ``high_var_scale``, giving it a raw variance of ~4.0 at the default --
    the only column above a 0.5 variance threshold. After z-scoring every
    column's variance is ~1.0, which is what makes the stage assignment
    observable: the same threshold keeps only ``high_var`` pre-preprocessing
    and keeps everything post-preprocessing.
    """
    n = len(subject_ids)
    pattern = (np.arange(n) % 2).astype(float)
    frame = pd.DataFrame(
        {
            "subject": list(subject_ids),
            "low_a": pattern,
            "low_b": (1.0 - pattern) * 0.5,
            "high_var": pattern * high_var_scale,
            # Classes repeat in pairs so no feature separates them perfectly.
            "y": (np.arange(n) % 4) // 2,
        }
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("low_a", "low_b", "high_var"),
        outcome=BinaryOutcome("y"),
    )


def _staged_ml_spec(*, pre_stage: bool) -> MLSpec:
    """An MLSpec whose variance selector sits in the requested stage."""
    selector = Spec(name="variance", params={"threshold": 0.5})
    return MLSpec(
        name="staged_demo",
        classifier=Spec(name="LogisticRegression", params={"max_iter": 500}),
        pre_preprocessing_feature_selectors=(selector,) if pre_stage else (),
        table_preprocessors=(Spec(name="zscore"),),
        feature_selectors=() if pre_stage else (selector,),
    )


@pytest.mark.unit
def test_build_table_pipeline_orders_stages_as_declared() -> None:
    """The pipeline runs pre-selectors, preprocessors, then post-selectors."""
    spec = MLSpec(
        name="order_demo",
        classifier=Spec(name="LogisticRegression"),
        pre_preprocessing_feature_selectors=(
            Spec(name="variance", params={"threshold": 0.0}),
        ),
        table_preprocessors=(Spec(name="zscore"),),
        feature_selectors=(Spec(name="correlation"),),
    )
    pipeline = build_table_pipeline(spec)
    assert [component.spec.name for component in pipeline.components] == [
        "variance",
        "zscore",
        "correlation",
    ]
    # ``.steps`` now carries scikit-learn's meaning: the FrameToTable head,
    # one adapter per component, and the terminal model adapter.
    assert [name for name, _ in pipeline.steps] == [
        "frame_to_table",
        "variance",
        "zscore",
        "correlation",
        "model",
    ]


@pytest.mark.unit
def test_build_table_pipeline_reads_the_single_ordered_step_list() -> None:
    """
    An ``MLSpec.steps`` chain assembles in list order, whatever the kinds.

    This is what the deprecated three-slot layout could not express: two
    preprocessors with a selector between them. Names resolve across both
    registries, which is why one list can hold either kind.
    """
    spec = MLSpec(
        name="interleaved",
        classifier=Spec(name="LogisticRegression"),
        steps=(
            Spec(name="zscore"),
            Spec(name="variance", params={"threshold": 0.0}),
            Spec(name="minmax"),
            Spec(name="correlation"),
        ),
    )
    pipeline = build_table_pipeline(spec)
    assert [component.spec.name for component in pipeline.components] == [
        "zscore",
        "variance",
        "minmax",
        "correlation",
    ]


@pytest.mark.unit
def test_build_table_pipeline_rejects_an_unknown_step_name() -> None:
    """
    A misspelled step is a loud failure, never a skipped step.

    The message has to name both vocabularies: a user reading it does not
    necessarily know whether they meant a preprocessor or a selector.
    """
    spec = MLSpec(
        name="typo",
        classifier=Spec(name="LogisticRegression"),
        steps=(Spec(name="z_score"),),
    )
    with pytest.raises(ComponentNotFoundError) as excinfo:
        build_table_pipeline(spec)
    assert "table preprocessor" in str(excinfo.value)
    assert "feature selector" in str(excinfo.value)


@pytest.mark.unit
def test_the_two_step_registries_share_no_names() -> None:
    """
    Name-only resolution is sound because the vocabularies are disjoint.

    ``build_table_step`` refuses to guess on an overlap, so this test is the
    early warning: a future registration that collides breaks the ordered
    step list, and it should break here rather than in someone's run.
    """
    assert not (
        set(TablePreprocessorRegistry.available())
        & set(FeatureSelectorRegistry.available())
    )


@pytest.mark.unit
def test_the_deprecated_layout_assembles_through_the_same_path() -> None:
    """
    Both layouts produce the same pipeline for the same order.

    ``MLSpec`` folds the chains into ``steps`` and ``build_table_pipeline``
    reads only ``steps``, so there is one assembly path and the two layouts
    cannot drift apart.
    """
    ordered = build_table_pipeline(
        MLSpec(
            name="ordered",
            classifier=Spec(name="LogisticRegression"),
            steps=(
                Spec(name="variance", params={"threshold": 0.0}),
                Spec(name="zscore"),
                Spec(name="correlation"),
            ),
        )
    )
    legacy = build_table_pipeline(
        MLSpec(
            name="legacy",
            classifier=Spec(name="LogisticRegression"),
            pre_preprocessing_feature_selectors=(
                Spec(name="variance", params={"threshold": 0.0}),
            ),
            table_preprocessors=(Spec(name="zscore"),),
            feature_selectors=(Spec(name="correlation"),),
        )
    )
    assert [component.spec.to_dict() for component in ordered.components] == [
        component.spec.to_dict() for component in legacy.components
    ]


@pytest.mark.unit
def test_pre_stage_selector_reads_raw_variances() -> None:
    """Pre-preprocessing selection keeps the raw high-variance column."""
    ids = tuple(f"S{i:02d}" for i in range(20))
    table = _staged_variance_table(ids)
    pipeline = build_table_pipeline(_staged_ml_spec(pre_stage=True)).fit(table)
    # Raw variances: high_var ~4.0 vs low_* below 0.5 -- only high_var
    # survives the 0.5 threshold BEFORE any z-scoring happens.
    assert pipeline.transform(table).feature_columns == ("high_var",)


@pytest.mark.unit
def test_post_stage_selector_reads_scaled_variances() -> None:
    """The same selector after z-scoring sees every variance as ~1.0."""
    ids = tuple(f"S{i:02d}" for i in range(20))
    table = _staged_variance_table(ids)
    pipeline = build_table_pipeline(_staged_ml_spec(pre_stage=False)).fit(table)
    # Post-z-score all variances are ~1.0 > 0.5, so nothing is pruned: the
    # stage assignment, not the selector, decides the semantics.
    assert pipeline.transform(table).feature_columns == ("low_a", "low_b", "high_var")


# ---------------------------------------------------------------------------
# scikit-learn Pipeline inheritance (v1.1)
# ---------------------------------------------------------------------------


def _sklearn_ready_pipeline(table: FeatureTable) -> TablePipeline:
    """A pipeline whose head declares ``table``'s column schema."""
    return TablePipeline(
        steps=[
            FrameToTable.from_table(table),
            VarianceSelector(threshold=0.0),
            ZScorePreprocessor(),
        ],
        model=LogisticRegressionClassifier(max_iter=500),
    )


@pytest.mark.unit
def test_pipeline_is_an_sklearn_pipeline() -> None:
    """The class IS an ``sklearn.pipeline.Pipeline``, not a look-alike."""
    assert issubclass(TablePipeline, SkPipeline)
    pipeline = _pipeline()
    # sklearn's own structural expectations of ``steps``.
    assert isinstance(pipeline.steps, list)
    assert all(
        isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str)
        for step in pipeline.steps
    )
    assert set(pipeline.named_steps) == {
        "frame_to_table",
        "variance",
        "zscore",
        "model",
    }
    assert is_classifier(pipeline)


@pytest.mark.unit
def test_components_expose_the_unwrapped_habit_objects() -> None:
    """``.components`` is the HABIT view; ``.model`` is the terminal one."""
    selector = VarianceSelector(threshold=0.01)
    scaler = ZScorePreprocessor()
    model = LogisticRegressionClassifier(max_iter=500)
    pipeline = TablePipeline(steps=[selector, scaler], model=model)
    assert pipeline.components == (selector, scaler)
    assert pipeline.model is model
    assert pipeline.classifier is model


@pytest.mark.unit
def test_pipeline_fits_its_own_component_objects_in_place() -> None:
    """Fitted state lands on the objects the caller passed, not on copies."""
    selector = VarianceSelector(threshold=0.01)
    pipeline = TablePipeline(
        steps=[selector], model=LogisticRegressionClassifier(max_iter=500)
    )
    pipeline.fit(make_feature_table(seed=21))
    assert pipeline.components[0] is selector
    # The reporting / save paths read the selection off this very object.
    assert selector.transform(make_feature_table(seed=22)).feature_columns


@pytest.mark.unit
def test_get_params_reports_only_sklearn_pipeline_parameters() -> None:
    """``model``/``classifier`` are construction sugar, never parameters."""
    pipeline = _pipeline()
    shallow = pipeline.get_params(deep=False)
    assert set(shallow) == set(SkPipeline._get_param_names())
    assert shallow["steps"] is pipeline.steps
    deep = pipeline.get_params(deep=True)
    # Nested addressing is what a param_grid needs.
    assert deep["model"] is pipeline.steps[-1][1]
    assert deep["model__component"] is pipeline.model
    assert deep["model__component__max_iter"] == 500


@pytest.mark.unit
def test_clone_returns_an_unfitted_equivalent_pipeline() -> None:
    """``clone`` rebuilds the whole pipeline, spec included, unfitted."""
    original = _pipeline().fit(make_feature_table(seed=23))
    copy = clone(original)
    assert isinstance(copy, TablePipeline)
    assert [name for name, _ in copy.steps] == [
        name for name, _ in original.steps
    ]
    assert copy.spec.fingerprint() == original.spec.fingerprint()
    assert copy.components[0] is not original.components[0]
    with pytest.raises(HABITAPIError):
        copy.transform(make_feature_table(seed=24))


@pytest.mark.unit
def test_clone_carries_the_seed_into_every_fold() -> None:
    """A seeded pipeline clones seeded, so CV folds stay reproducible."""
    pipeline = TablePipeline(
        steps=[], model=RandomForestClassifier()
    )
    pipeline.set_random_state(31)
    assert clone(pipeline).model._seed == 31


@pytest.mark.unit
def test_set_params_replaces_a_step_through_sklearn() -> None:
    """A step is addressable by name, the sklearn way."""
    pipeline = _pipeline()
    replacement = ZScorePreprocessor(across_features=True)
    pipeline.set_params(zscore=as_transformer(replacement, copy_on_fit=False))
    assert pipeline.components[1] is replacement
    assert pipeline.spec.params["steps"][1]["params"]["across_features"] is True


@pytest.mark.unit
def test_frame_schema_head_passes_feature_tables_through_unchanged() -> None:
    """A FeatureTable never round-trips through frame construction."""
    table = make_feature_table(seed=25)
    head = TablePipeline(steps=[], model=LogisticRegressionClassifier()).frame_schema
    assert head.transform(table) is table


@pytest.mark.unit
def test_frame_to_table_must_sit_at_the_head() -> None:
    """A misplaced FrameToTable would silently discard upstream selection."""
    with pytest.raises(HABITAPIError, match="HEAD"):
        TablePipeline(
            steps=[ZScorePreprocessor(), FrameToTable()],
            model=LogisticRegressionClassifier(),
        )


@pytest.mark.unit
def test_a_bare_frame_without_a_declared_schema_fails_loudly() -> None:
    """Modelling on an identifier column must never happen silently."""
    table = make_feature_table(seed=26)
    pipeline = TablePipeline(
        steps=[ZScorePreprocessor()], model=LogisticRegressionClassifier()
    )
    with pytest.raises(HABITAPIError, match="declares no column schema"):
        pipeline.fit(table.frame)


@pytest.mark.unit
def test_cross_val_score_runs_on_the_raw_frame() -> None:
    """sklearn's CV driver slices the frame; the head rebuilds the table."""
    table = make_feature_table(tuple(f"S{i:02d}" for i in range(40)), seed=27)
    scores = cross_val_score(
        _sklearn_ready_pipeline(table),
        table.frame,
        table.frame["y"].to_numpy(),
        cv=3,
        scoring="roc_auc",
    )
    assert scores.shape == (3,)
    assert np.all(scores > 0.5)


@pytest.mark.unit
def test_grid_search_addresses_nested_component_parameters() -> None:
    """``model__component__C`` reaches the HABIT classifier's own parameter."""
    table = make_feature_table(tuple(f"S{i:02d}" for i in range(40)), seed=28)
    search = GridSearchCV(
        _sklearn_ready_pipeline(table),
        {"model__component__C": [0.001, 1.0, 100.0]},
        cv=3,
    )
    search.fit(table.frame, table.frame["y"].to_numpy())
    assert search.best_params_["model__component__C"] in (0.001, 1.0, 100.0)
    # The winning value really is the one the refitted pipeline carries.
    best = search.best_estimator_
    assert best.model.spec.params["C"] == search.best_params_["model__component__C"]


@pytest.mark.unit
def test_predict_returns_arrays_for_frames_and_series_for_tables() -> None:
    """HABIT type in, HABIT type out; sklearn type in, sklearn type out."""
    table = make_feature_table(tuple(f"S{i:02d}" for i in range(30)), seed=29)
    pipeline = _sklearn_ready_pipeline(table).fit(table)
    assert isinstance(pipeline.predict(table), pd.Series)
    assert isinstance(pipeline.predict_proba(table), pd.DataFrame)
    assert isinstance(pipeline.predict(table.frame), np.ndarray)
    probabilities = pipeline.predict_proba(table.frame)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (30, 2)
    # ``classes_`` carries the endpoint's own dtype, so label-aware scorers
    # can match it against the y they were handed.
    assert list(pipeline.classes_) == [0, 1]


@pytest.mark.unit
def test_selector_report_is_logged_from_the_adapter(caplog) -> None:
    """The per-selector feature-count report survived the move into adapters."""
    table = make_feature_table(seed=30, n_noise=2, constant_column=True)
    with caplog.at_level("INFO", logger="habit.pipeline.sklearn_interop"):
        TablePipeline(
            steps=[VarianceSelector(threshold=0.0), ZScorePreprocessor()],
            model=LogisticRegressionClassifier(max_iter=500),
        ).fit(table)
    messages = [record.getMessage() for record in caplog.records]
    assert "Step 1: Applying 'variance' feature selection" in messages
    assert "  Features before this step: 4" in messages
    assert "  Features after this step: 3" in messages
    assert "  Number of features removed: 1" in messages
    # Preprocessors were never part of this report and still are not.
    assert not any("'zscore' feature selection" in m for m in messages)


# ---------------------------------------------------------------------------
# Hard constraints: spec fingerprint shape and .habitpipeline compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_spec_payload_shape_is_locked() -> None:
    """
    Lock the composed ``Spec`` payload SHAPE, byte for byte.

    Every provenance record ever written by HABIT hashes this payload, so a
    new key, a renamed key or an extra entry (the ``FrameToTable`` head or one
    of the sklearn adapters leaking in) would move every recorded fingerprint
    and silently invalidate the golden baselines. The literal below is the
    contract; changing it is a breaking change to the artefact format, not a
    refactor.
    """
    pipeline = TablePipeline(
        steps=[
            FrameToTable(id_columns=("subject",), outcome=BinaryOutcome("y")),
            VarianceSelector(threshold=0.01),
            ZScorePreprocessor(),
        ],
        model=LogisticRegressionClassifier(max_iter=500),
    )
    payload = pipeline.spec.to_dict()
    assert payload["name"] == "table_pipeline"
    assert set(payload["params"]) == {"steps", "model"}
    assert payload["params"]["steps"] == [
        {
            "name": "variance",
            "params": {"threshold": 0.01, "top_k": None, "top_percent": None},
            "version": "1.0",
        },
        {"name": "zscore", "params": {"across_features": False}, "version": "1.0"},
    ]
    assert payload["params"]["model"] == {
        "name": "LogisticRegression",
        "params": {
            "C": 1.0,
            "class_weight": None,
            "max_iter": 500,
            "penalty": "l2",
            "solver": "liblinear",
        },
        "version": "1.0",
    }
    # A pipeline WITHOUT an explicit head must fingerprint identically: the
    # head is interop plumbing, never scientific definition.
    bare = TablePipeline(
        steps=[VarianceSelector(threshold=0.01), ZScorePreprocessor()],
        model=LogisticRegressionClassifier(max_iter=500),
    )
    assert bare.spec.fingerprint() == pipeline.spec.fingerprint()


def _write_format_version_1_file(path, pipeline: TablePipeline):
    """
    Write a ``.habitpipeline`` exactly as HABIT v1.0 wrote them.

    Reproducing the old writer here (rather than shipping a binary fixture)
    keeps the compatibility test readable and pins the v1 layout in one
    place: a JSON manifest at ``format_version`` 1 plus a payload pickling
    the HABIT components, with no frame schema.
    """
    def _record(component):
        cls = type(component)
        return {
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "spec": component.spec.to_dict(),
        }

    components = list(pipeline.components)
    manifest = {
        "format": "habit.tablepipeline",
        "format_version": 1,
        "habit_version": "1.0.4",
        "steps": [_record(component) for component in components],
        "model": _record(pipeline.model),
        "is_fitted": True,
        "fit_output_columns": list(pipeline.transform(_LOAD_PROBE).feature_columns),
    }
    payload = {
        "steps": components,
        "model": pipeline.model,
        "is_fitted": True,
        "fit_output_columns": tuple(
            pipeline.transform(_LOAD_PROBE).feature_columns
        ),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr("payload.pkl", pickle.dumps(payload))
    return path


#: Fixed table the format-version-1 fixture is fitted and probed with.
_LOAD_PROBE = make_feature_table(tuple(f"S{i:02d}" for i in range(20)), seed=33)


@pytest.mark.unit
def test_load_rejects_format_version_1_files_with_migration_guidance(tmp_path) -> None:
    """
    A ``.habitpipeline`` written before the sklearn refactor still loads.

    V2 removes the module paths embedded in v1 payload pickles. The archive
    must be rejected before unpickling, with actionable migration guidance.
    """
    pipeline = _pipeline().fit(_LOAD_PROBE)
    expected = pipeline.predict_proba(_LOAD_PROBE).to_numpy()
    legacy = _write_format_version_1_file(
        tmp_path / "v1.habitpipeline", pipeline
    )
    with pytest.raises(CompatibilityError, match="HABIT v1 .habitpipeline"):
        TablePipeline.load(legacy)


@pytest.mark.unit
def test_saved_pipeline_declares_format_version_3(tmp_path) -> None:
    """New files announce v2 format version 3 and carry the frame schema."""
    table = make_feature_table(seed=34)
    pipeline = _sklearn_ready_pipeline(table).fit(table)
    destination = pipeline.save(tmp_path / "v2.habitpipeline")
    with zipfile.ZipFile(destination, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert manifest["format_version"] == 3
    assert manifest["declares_frame_schema"] is True
    assert manifest["step_names"] == [
        "frame_to_table",
        "variance",
        "zscore",
        "model",
    ]
    loaded = TablePipeline.load(destination)
    assert loaded.frame_schema.declares_schema
    np.testing.assert_allclose(
        loaded.predict_proba(table).to_numpy(),
        pipeline.predict_proba(table).to_numpy(),
    )


@pytest.mark.unit
def test_pre_stage_selection_is_fitted_on_training_rows_only() -> None:
    """Evaluation tables reuse the TRAIN selection; they never re-select."""
    train_ids = tuple(f"T{i:02d}" for i in range(20))
    eval_ids = tuple(f"E{i:02d}" for i in range(6))
    train = _staged_variance_table(train_ids)
    # On the evaluation table the variance ranking is inverted: high_var is
    # constant there while low_a spans a wide range. A leakage-prone
    # implementation would re-select low_a here.
    evaluation = _staged_variance_table(eval_ids, high_var_scale=0.0)
    evaluation.frame["low_a"] = evaluation.frame["low_a"] * 8.0
    pipeline = build_table_pipeline(_staged_ml_spec(pre_stage=True)).fit(train)
    transformed = pipeline.transform(evaluation)
    assert transformed.feature_columns == ("high_var",)
