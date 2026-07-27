# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.
#
"""
Contract tests for the feature-selection entry point.

Every selector implicitly assumes that the feature-name list it receives lines
up with the columns of the matrix it receives. These tests pin that assumption
down at the only place it is enforced (``run_selector``) plus the one selector
that historically fitted on the unsliced matrix (``lasso``), so a future edit
that reintroduces misalignment fails loudly here instead of silently mislabeling
coefficients.

Covers:
- Candidate normalization and validation inside ``run_selector``
- Output restriction to the candidate set
- ``lasso_selector`` name/coefficient alignment under subset and reordered input
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import pytest

from habit.core.machine_learning.feature_selectors.lasso_selector import lasso_selector
from habit.core.machine_learning.feature_selectors.selector_registry import (
    SelectorRegistry,
    run_selector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n: int = 80, p: int = 6, seed: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a small reproducible matrix whose columns carry distinguishable signal.

    Each column is scaled by a different factor so that a mislabeled coefficient
    vector produces a different selection than a correctly labeled one.

    Args:
        n: Number of samples.
        p: Number of features.
        seed: Seed for the random generator.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target.
    """
    rng = np.random.RandomState(seed)
    raw: np.ndarray = rng.randn(n, p)
    # Give column i an amplitude of (i + 1) so columns are not interchangeable.
    raw = raw * np.arange(1, p + 1, dtype=float)
    X = pd.DataFrame(raw, columns=[f"feature_{i}" for i in range(p)])
    # Target depends on the first two columns, keeping the problem learnable.
    logit: np.ndarray = raw[:, 0] * 0.8 + raw[:, 1] * 0.5
    y = pd.Series((logit > np.median(logit)).astype(int), name="label")
    return X, y


class _Probe:
    """
    Record what a selector actually receives, then echo the input back.

    Registering a probe is the only way to observe the frame/name pair handed to
    a selector, which is exactly the invariant under test.
    """

    def __init__(self) -> None:
        self.seen_columns: List[str] = []
        self.seen_names: List[str] = []
        self.echo: List[str] = []

    def __call__(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]) -> List[str]:
        self.seen_columns = list(X.columns)
        self.seen_names = list(selected_features)
        return list(self.echo) if self.echo else list(selected_features)


@pytest.fixture
def probe_selector() -> Iterator[_Probe]:
    """
    Register a probe selector under a unique name and unregister afterwards.

    The registry is process-global class state, so the entry must be removed to
    keep other tests unaffected.
    """
    probe = _Probe()
    name = "__probe_selector_for_tests__"
    SelectorRegistry.register(name)(probe)
    try:
        yield probe
    finally:
        SelectorRegistry._registry.pop(name, None)
        SelectorRegistry._metadata.pop(name, None)


PROBE_NAME = "__probe_selector_for_tests__"


# ---------------------------------------------------------------------------
# run_selector: input contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.ml
class TestRunSelectorInputContract:
    """The frame and the name list handed to a selector must always agree."""

    def test_subset_is_sliced_before_reaching_selector(self, probe_selector: _Probe) -> None:
        """A selector must only ever see the candidate columns, never the full matrix."""
        X, y = _make_data()
        candidates: List[str] = ["feature_3", "feature_1"]

        run_selector(PROBE_NAME, X, y, candidates)

        assert probe_selector.seen_columns == candidates
        assert probe_selector.seen_names == candidates

    def test_reordered_input_keeps_columns_and_names_aligned(
        self, probe_selector: _Probe
    ) -> None:
        """A reordered candidate list must not desynchronize columns from names."""
        X, y = _make_data()
        reordered: List[str] = list(reversed(X.columns.tolist()))

        run_selector(PROBE_NAME, X, y, reordered)

        assert probe_selector.seen_columns == reordered
        assert probe_selector.seen_names == probe_selector.seen_columns

    def test_none_candidates_defaults_to_all_columns(self, probe_selector: _Probe) -> None:
        """Omitting the candidate list means "use every column", in column order."""
        X, y = _make_data()

        run_selector(PROBE_NAME, X, y, None)

        assert probe_selector.seen_columns == X.columns.tolist()
        assert probe_selector.seen_names == X.columns.tolist()

    def test_duplicate_candidates_are_collapsed(self, probe_selector: _Probe) -> None:
        """Repeated candidate names must not produce repeated columns."""
        X, y = _make_data()

        run_selector(PROBE_NAME, X, y, ["feature_2", "feature_0", "feature_2"])

        assert probe_selector.seen_columns == ["feature_2", "feature_0"]
        assert probe_selector.seen_names == probe_selector.seen_columns

    def test_unknown_candidate_raises_with_actionable_message(
        self, probe_selector: _Probe
    ) -> None:
        """An unknown feature name must fail fast and name the offender."""
        X, y = _make_data()

        with pytest.raises(ValueError, match="absent from"):
            run_selector(PROBE_NAME, X, y, ["feature_0", "not_a_feature"])

    def test_duplicated_column_labels_are_rejected(self, probe_selector: _Probe) -> None:
        """Duplicated labels in the matrix make name-based selection ambiguous."""
        X, y = _make_data()
        X_dup = pd.concat([X, X[["feature_0"]]], axis=1)

        with pytest.raises(ValueError, match="duplicated column labels"):
            run_selector(PROBE_NAME, X_dup, y, None)


# ---------------------------------------------------------------------------
# run_selector: output contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.ml
class TestRunSelectorOutputContract:
    """The result must always be usable to index the caller's DataFrame."""

    def test_names_outside_candidate_set_are_dropped(self, probe_selector: _Probe) -> None:
        """
        Selectors consulting external tables (e.g. ICC) may return foreign names.

        Those must be filtered so downstream indexing cannot raise KeyError.
        """
        X, y = _make_data()
        probe_selector.echo = ["feature_0", "feature_from_another_table"]

        result: List[str] = run_selector(PROBE_NAME, X, y, ["feature_0", "feature_1"])

        assert result == ["feature_0"]

    def test_duplicate_output_is_collapsed(self, probe_selector: _Probe) -> None:
        """A selector returning the same name twice must not duplicate columns."""
        X, y = _make_data()
        probe_selector.echo = ["feature_1", "feature_1", "feature_0"]

        result: List[str] = run_selector(PROBE_NAME, X, y, ["feature_0", "feature_1"])

        assert result == ["feature_1", "feature_0"]

    def test_result_can_always_index_the_input_frame(self, probe_selector: _Probe) -> None:
        """The contract's practical payoff: the result is always a valid indexer."""
        X, y = _make_data()
        probe_selector.echo = ["feature_2", "ghost_feature"]

        result: List[str] = run_selector(PROBE_NAME, X, y, None)

        # Would raise KeyError if a foreign name leaked through.
        assert X[result].shape == (len(X), 1)


# ---------------------------------------------------------------------------
# lasso: internal self-consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.ml
class TestLassoAlignment:
    """
    ``lasso_selector`` must label coefficients with the columns it fitted on.

    It used to fit on the unsliced ``X`` while labeling with ``selected_features``,
    which silently mismatched whenever the two disagreed.
    """

    def test_subset_input_selects_only_from_the_subset(self) -> None:
        """With a narrower candidate list, no outside feature may be returned."""
        X, y = _make_data()
        candidates: List[str] = ["feature_0", "feature_2", "feature_4"]

        selected, _, _, coefs_path = lasso_selector(
            X=X, y=y, cv=3, random_state=0, selected_features=candidates
        )

        assert set(selected).issubset(set(candidates))
        # The coefficient path must describe the candidate subset, not all of X.
        assert coefs_path.shape[0] == len(candidates)

    def test_subset_input_matches_pre_sliced_input(self) -> None:
        """
        Passing a subset must equal pre-slicing the frame yourself.

        This is the property that the pipeline previously relied on by accident,
        because it always pre-sliced before calling.
        """
        X, y = _make_data()
        candidates: List[str] = ["feature_0", "feature_2", "feature_4"]

        via_argument, alpha_argument, _, _ = lasso_selector(
            X=X, y=y, cv=3, random_state=0, selected_features=candidates
        )
        via_slicing, alpha_slicing, _, _ = lasso_selector(
            X=X[candidates], y=y, cv=3, random_state=0
        )

        assert via_argument == via_slicing
        assert alpha_argument == pytest.approx(alpha_slicing)

    def test_reordered_input_yields_the_same_feature_set(self) -> None:
        """
        Reordering the candidate list must not change which features are chosen.

        Under the old implementation the coefficient vector stayed in column
        order while the labels followed the reordered list, so the selected set
        changed with the ordering.
        """
        X, y = _make_data()
        ordered: List[str] = X.columns.tolist()
        reordered: List[str] = list(reversed(ordered))

        selected_ordered, _, _, _ = lasso_selector(
            X=X, y=y, cv=3, random_state=0, selected_features=ordered
        )
        selected_reordered, _, _, _ = lasso_selector(
            X=X, y=y, cv=3, random_state=0, selected_features=reordered
        )

        assert set(selected_ordered) == set(selected_reordered)

    def test_coefficients_belong_to_their_named_feature(self) -> None:
        """
        A constant column must be reported with a zero coefficient.

        Constant input carries no signal, so Lasso cannot assign it weight. If
        names and coefficients were offset, the zero would land on a different
        feature and the constant one would appear selected.
        """
        X, y = _make_data()
        X = X.copy()
        X["feature_constant"] = 1.0
        candidates: List[str] = ["feature_constant", "feature_0", "feature_1"]

        selected, _, _, _ = lasso_selector(
            X=X, y=y, cv=3, random_state=0, selected_features=candidates
        )

        assert "feature_constant" not in selected
