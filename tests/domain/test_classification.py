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
"""Tests for the fourteen built-in classifiers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError, OptionalDependencyError
from habit.domain.classification import (
    AdaboostClassifier,
    AutogluonTabularClassifier,
    BernoulliNbClassifier,
    ClassifierRegistry,
    DecisionTreeClassifier,
    GaussianNbClassifier,
    GradientBoostingClassifier,
    KnnClassifier,
    LogisticRegressionClassifier,
    MlpClassifier,
    MultinomialNbClassifier,
    RandomForestClassifier,
    SvcClassifier,
    SvmClassifier,
    XgboostClassifier,
)
from habit.domain.protocols import Seedable
from habit.domain.table_protocols import Classifier

from .conftest import make_feature_table

#: The thirteen estimator-backed classifiers, with whether they need
#: non-negative inputs (the naive-Bayes variants).
_SKLEARN_FAMILY = (
    (DecisionTreeClassifier, False),
    (KnnClassifier, False),
    (SvmClassifier, False),
    (SvcClassifier, False),
    (MlpClassifier, False),
    (LogisticRegressionClassifier, False),
    (RandomForestClassifier, False),
    (GradientBoostingClassifier, False),
    (XgboostClassifier, False),
    (AdaboostClassifier, False),
    (GaussianNbClassifier, False),
    (MultinomialNbClassifier, True),
    (BernoulliNbClassifier, True),
)


@pytest.mark.unit
def test_registry_lists_all_fourteen_classifiers() -> None:
    """The registry constructs every built-in classifier by its v0.1 name."""
    assert set(ClassifierRegistry.available()) == {
        "DecisionTree",
        "KNN",
        "SVM",
        "SVC",
        "MLP",
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "AdaBoost",
        "GaussianNB",
        "MultinomialNB",
        "BernoulliNB",
        "AutoGluonTabular",
    }
    for name in ClassifierRegistry.available():
        instance = ClassifierRegistry.create(name)
        assert isinstance(instance, Classifier)
        assert instance.spec.name == name
        assert ClassifierRegistry.get_params_model(name) is not None


@pytest.mark.unit
@pytest.mark.parametrize("cls,non_negative", _SKLEARN_FAMILY, ids=lambda v: getattr(v, "__name__", str(v)))
def test_classifier_fit_predict_roundtrip(cls, non_negative: bool) -> None:
    """Fit on separable synthetic data, then predict labels and probabilities."""
    table = make_feature_table(
        tuple(f"S{i}" for i in range(30)), n_noise=2, seed=1, non_negative=non_negative
    )
    if cls is BernoulliNbClassifier:
        # BernoulliNB binarises internally at 0.5; give it genuinely binary
        # features so the signal survives the binarisation.
        for column in table.feature_columns:
            table.frame[column] = (table.frame[column] > table.frame[column].median()).astype(float)
    classifier = cls()
    classifier.set_random_state(0)  # Seedable: deterministic across instances.
    classifier.fit(table)

    labels = classifier.predict(table)
    assert list(labels.index) == list(table.frame["subject"])
    # Every estimator learns the signal well above chance (~0.5); the
    # threshold stays loose because kNN/NB smooth the train-set boundary.
    assert (labels.to_numpy() == table.frame["y"].to_numpy()).mean() >= 0.8

    probabilities = classifier.predict_proba(table)
    assert probabilities.shape == (30, 2)
    assert set(probabilities.columns) == {"0", "1"}
    assert ((probabilities.to_numpy() >= 0) & (probabilities.to_numpy() <= 1)).all()


@pytest.mark.unit
def test_predict_before_fit_and_schema_drift_raise() -> None:
    """Unfitted predict and missing fit columns are loud errors."""
    table = make_feature_table()
    classifier = LogisticRegressionClassifier()
    with pytest.raises(HABITAPIError):
        classifier.predict(table)
    fitted = classifier.fit(table)
    drifted = make_feature_table(n_noise=0)
    drifted = type(table)(
        frame=drifted.frame.drop(columns=["signal"]),
        id_columns=drifted.id_columns,
        feature_columns=tuple(c for c in drifted.feature_columns if c != "signal"),
        outcome=drifted.outcome,
        provenance=drifted.provenance,
    )
    with pytest.raises(HABITAPIError):
        fitted.predict(drifted)


@pytest.mark.unit
def test_seed_makes_stochastic_classifier_deterministic() -> None:
    """Two same-seeded RandomForests predict identically; Seedable holds."""
    table = make_feature_table(seed=2)
    assert isinstance(RandomForestClassifier(), Seedable)
    first, second = RandomForestClassifier(), RandomForestClassifier()
    first.set_random_state(13)
    second.set_random_state(13)
    np.testing.assert_array_equal(
        first.fit(table).predict(table).to_numpy(),
        second.fit(table).predict(table).to_numpy(),
    )


@pytest.mark.unit
def test_spec_records_constructor_params() -> None:
    """The spec captures the algorithm name and constructor parameters."""
    classifier = LogisticRegressionClassifier(C=0.5, max_iter=500)
    assert classifier.spec.name == "LogisticRegression"
    assert classifier.spec.params["C"] == 0.5
    assert classifier.spec.params["max_iter"] == 500


@pytest.mark.unit
def test_autogluon_classifier_optional_dependency() -> None:
    """Without AutoGluon installed, fit fails with a precise optional error."""
    if importlib.util.find_spec("autogluon") is not None:
        pytest.skip("AutoGluon is installed in this environment")
    classifier = AutogluonTabularClassifier()
    assert isinstance(classifier, Classifier)
    assert isinstance(classifier, Seedable)
    with pytest.raises(OptionalDependencyError):
        classifier.fit(make_feature_table())
