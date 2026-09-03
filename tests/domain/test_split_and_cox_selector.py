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
"""Tests for endpoint-aware splitting and the univariate Cox selector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import HABITAPIError
from habit.contracts import (
    BinaryOutcome,
    ContinuousOutcome,
    FeatureTable,
    MulticlassOutcome,
    SurvivalOutcome,
)
from habit.feature_selection import FeatureSelectorRegistry
from habit.evaluation.split import kfold_indices, stratify_labels, train_test_indices

pytestmark = pytest.mark.unit


def _survival_table(n: int = 80, seed: int = 0) -> FeatureTable:
    """Survival table where f1 drives the hazard, f2/f3 are noise."""
    rng = np.random.RandomState(seed)
    f1 = rng.normal(size=n)
    time = np.clip(10.0 * np.exp(-0.9 * f1) * rng.uniform(0.5, 1.5, n), 0.5, None)
    event = (rng.rand(n) < 0.7).astype(int)
    frame = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(n)],
            "f1": f1,
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            "t": time,
            "e": event,
        }
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f1", "f2", "f3"),
        outcome=SurvivalOutcome(time_column="t", event_column="e"),
    )


# ---------------------------------------------------------------------------
# stratify_labels
# ---------------------------------------------------------------------------


def test_stratify_labels_uses_the_event_indicator_for_survival() -> None:
    """Survival stratifies on observed-vs-censored, never on time."""
    table = _survival_table()
    labels = stratify_labels(table.outcome, table.frame)
    assert labels is not None
    assert set(np.unique(labels)) <= {0, 1}
    assert labels.tolist() == table.frame["e"].tolist()


def test_stratify_labels_uses_the_label_for_classification() -> None:
    """Binary and multiclass endpoints stratify on their class label."""
    frame = pd.DataFrame(
        {"subject": ["a", "b", "c"], "f": [1.0, 2.0, 3.0], "y": [0, 1, 0]}
    )
    binary = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f",),
        outcome=BinaryOutcome("y"),
    )
    assert stratify_labels(binary.outcome, binary.frame).tolist() == [0, 1, 0]
    multi = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f",),
        outcome=MulticlassOutcome("y"),
    )
    assert stratify_labels(multi.outcome, multi.frame).tolist() == [0, 1, 0]


def test_stratify_labels_is_none_for_continuous_and_missing() -> None:
    """A continuous endpoint has no strata, and neither does no endpoint."""
    frame = pd.DataFrame(
        {"subject": ["a", "b"], "f": [1.0, 2.0], "y": [1.5, 2.5]}
    )
    continuous = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f",),
        outcome=ContinuousOutcome("y"),
    )
    assert stratify_labels(continuous.outcome, continuous.frame) is None
    assert stratify_labels(None, continuous.frame) is None


# ---------------------------------------------------------------------------
# train_test_indices / kfold_indices
# ---------------------------------------------------------------------------


def test_train_test_split_stratified_on_event() -> None:
    """A stratified survival split keeps the event rate on both sides."""
    table = _survival_table()
    labels = stratify_labels(table.outcome, table.frame)
    train_index, test_index = train_test_indices(
        len(table.frame), test_size=0.3, labels=labels, seed=0
    )
    train_rate = labels[train_index].mean()
    test_rate = labels[test_index].mean()
    assert abs(train_rate - test_rate) < 0.15
    assert len(set(train_index) & set(test_index)) == 0


def test_train_test_split_is_reproducible() -> None:
    """The same seed yields the identical split."""
    table = _survival_table()
    labels = stratify_labels(table.outcome, table.frame)
    a = train_test_indices(len(table.frame), labels=labels, seed=7)
    b = train_test_indices(len(table.frame), labels=labels, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_kfold_keeps_event_rate_in_every_fold() -> None:
    """Each fold of a stratified survival K-fold holds both events and censored."""
    table = _survival_table(n=100)
    labels = stratify_labels(table.outcome, table.frame)
    folds = list(kfold_indices(len(table.frame), n_splits=5, labels=labels, seed=0))
    assert len(folds) == 5
    for _, validation_index in folds:
        # Every validation fold observes at least one event and one censored row.
        assert set(np.unique(labels[validation_index])) == {0, 1}


def test_kfold_rejects_a_stratum_smaller_than_the_folds() -> None:
    """Fewer events than folds makes stratification impossible, loudly."""
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # a single event
    with pytest.raises(HABITAPIError, match="smallest stratum"):
        list(kfold_indices(10, n_splits=5, labels=labels))


def test_unstratified_kfold_for_continuous() -> None:
    """A continuous endpoint yields plain (unstratified) folds that tile the rows."""
    n = 30
    folds = list(kfold_indices(n, n_splits=3, labels=None, seed=0))
    covered = np.concatenate([validation for _, validation in folds])
    assert sorted(covered.tolist()) == list(range(n))


# ---------------------------------------------------------------------------
# univariate_cox selector
# ---------------------------------------------------------------------------


def test_univariate_cox_ranks_the_signal_first() -> None:
    """The hazard-driving feature beats the noise columns by p-value."""
    table = _survival_table(n=120)
    selector = FeatureSelectorRegistry.create(
        "univariate_cox", n_features_to_select=1
    ).fit(table)
    assert selector.selected_columns_ == ("f1",)


def test_univariate_cox_threshold_path_selects_by_pvalue() -> None:
    """Without an explicit count, the p_threshold keeps the strong signal."""
    table = _survival_table(n=120)
    selector = FeatureSelectorRegistry.create("univariate_cox", p_threshold=0.01).fit(
        table
    )
    assert "f1" in selector.selected_columns_


def test_univariate_cox_transform_restricts_columns() -> None:
    """transform applies the fit-time selection to any later table."""
    table = _survival_table(n=120)
    selector = FeatureSelectorRegistry.create(
        "univariate_cox", n_features_to_select=1
    ).fit(table)
    reduced = selector.transform(table)
    assert reduced.feature_columns == ("f1",)
    # The endpoint survives the restriction.
    assert reduced.outcome is table.outcome


def test_univariate_cox_rejects_a_non_survival_endpoint() -> None:
    """A Cox selector on a binary table fails with a typed message."""
    frame = pd.DataFrame(
        {"subject": ["a", "b", "c"], "f": [1.0, 2.0, 3.0], "y": [0, 1, 0]}
    )
    table = FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f",),
        outcome=BinaryOutcome("y"),
    )
    with pytest.raises(HABITAPIError, match="'binary'"):
        FeatureSelectorRegistry.create("univariate_cox").fit(table)
