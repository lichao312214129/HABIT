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
"""
Contract tests for the converged variance / correlation selection algorithms.

``variance`` (feature selector) and ``variance_filter`` (table preprocessor)
are the same algorithm registered under two names, and so are ``correlation``
and ``correlation_filter``. After the convergence all four delegate to one
kernel implementation, which creates two risks these tests exist to close:

1. The two names had DIFFERENT behaviour in the degenerate case -- the
   preprocessor kept the highest-variance column when nothing cleared the
   threshold, the selector returned nothing. Collapsing them onto one
   implementation must preserve both behaviours, not pick a winner.
2. Adding the ``keep_at_least_one`` parameter must not move any existing
   ``Spec`` fingerprint, because every provenance record and golden baseline
   HABIT has ever written hashes those payloads.

The reference implementations below are transcriptions of the pre-convergence
code (v1.0.4, commit ``ced2e583``). They are the oracle: the tests compare the
converged kernel against the numbers the old code produced, rather than
against the kernel's own current output.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from habit.feature_selection import CorrelationSelector, VarianceSelector
from habit.feature_selection.registry import FeatureSelectorRegistry
from habit.table_preprocessing.methods import (
    CorrelationFilterPreprocessor,
    VarianceFilterPreprocessor,
)
from habit.table_preprocessing.registry import TablePreprocessorRegistry
from habit.kernels import feature_transforms as kernel

from .conftest import make_feature_table


# ---------------------------------------------------------------------------
# Reference implementations (pre-convergence behaviour, verbatim logic)
# ---------------------------------------------------------------------------


def _v1_selector_variance(
    block: pd.DataFrame,
    threshold: float,
    top_k: Optional[int] = None,
    top_percent: Optional[float] = None,
) -> List[str]:
    """
    Reproduce the pre-convergence ``VarianceSelector.fit`` selection.

    Args:
        block: Unit-by-feature matrix.
        threshold: Variance cut-off (strictly greater than survives).
        top_k: Keep this many highest-variance columns, checked first.
        top_percent: Keep this percent (0-100) of columns, checked second.

    Returns:
        List[str]: Surviving column names, with NO empty-result fallback --
        the selector was allowed to select nothing.
    """
    variances = block.var().sort_values(ascending=False)
    if top_k is not None and top_k > 0:
        k = min(top_k, len(variances))
        return list(variances.index[:k])
    if top_percent is not None and 0 < top_percent <= 100:
        k = int(np.ceil(len(variances) * top_percent / 100))
        return list(variances.index[:k])
    return list(block.var()[block.var() > threshold].index)


def _v1_kernel_variance(block: pd.DataFrame, threshold: float) -> List[str]:
    """
    Reproduce the pre-convergence ``select_variance_columns`` kernel.

    Args:
        block: Unit-by-feature matrix.
        threshold: Variance cut-off (strictly greater than survives).

    Returns:
        List[str]: Surviving column names, falling back to the single
        highest-variance column so the chain never empties the matrix.
    """
    variances = block.var()
    selected = variances[variances > threshold].index.tolist()
    if not selected:
        selected = [variances.sort_values(ascending=False).index[0]]
    return [str(column) for column in selected]


def _v1_selector_correlation(
    block: pd.DataFrame, threshold: float, method: str
) -> List[str]:
    """
    Reproduce the pre-convergence ``CorrelationSelector.fit`` selection.

    Args:
        block: Unit-by-feature matrix.
        threshold: Absolute-correlation cut-off.
        method: Correlation method passed to ``DataFrame.corr``.

    Returns:
        List[str]: Surviving column names from the left-to-right greedy walk.
    """
    full_corr = block.corr(method=method)
    features = list(block.columns)
    i = 0
    while i < len(features):
        current = features[i]
        to_remove = [
            features[j]
            for j in range(i + 1, len(features))
            if abs(full_corr.loc[current, features[j]]) > threshold
        ]
        features = [f for f in features if f not in to_remove]
        i += 1
    return features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spread_table():
    """
    Build a table with a wide variance spread plus one constant column.

    Returns:
        FeatureTable: ``signal`` (moderate variance), three shrunk ``noise``
        columns (tiny variance), ``spread`` (large variance) and ``constant``
        (zero variance), so every variance mode has a distinguishable answer.
    """
    table = make_feature_table(constant_column=True)
    for noise_column in ("noise0", "noise1", "noise2"):
        table.frame[noise_column] = table.frame[noise_column] * 0.01
    table.frame["spread"] = np.linspace(-10.0, 10.0, table.frame.shape[0])
    return type(table)(
        frame=table.frame,
        id_columns=table.id_columns,
        feature_columns=(*table.feature_columns, "spread"),
        outcome=table.outcome,
        provenance=table.provenance,
    )


def _all_constant_table():
    """
    Build a table in which EVERY feature column has zero variance.

    This is the degenerate case where the two registered names historically
    disagreed: the preprocessor kept one column, the selector kept none.

    Returns:
        FeatureTable: Four constant feature columns with distinct values.
    """
    table = make_feature_table(n_noise=2)
    frame = table.frame.copy()
    frame["signal"] = 1.0
    frame["noise0"] = 2.0
    frame["noise1"] = 3.0
    frame["flat"] = 4.0
    return type(table)(
        frame=frame,
        id_columns=table.id_columns,
        feature_columns=("signal", "noise0", "noise1", "flat"),
        outcome=table.outcome,
        provenance=table.provenance,
    )


def _in_table_order(table, names: List[str]) -> List[str]:
    """
    Project a selection onto the fit table's declared column order.

    ``FittedSelectorBase._remember_selection`` normalises every selector's
    output to the table's own feature-column order, because downstream steps
    need a stable schema. The ranked variance modes therefore cannot be
    compared to the kernel's descending-variance output directly; both sides
    have to go through the same normalisation.

    Args:
        table: The table the selector was fitted on.
        names: Column names in whatever order the reference produced.

    Returns:
        List[str]: ``names`` re-ordered to follow ``table.feature_columns``.
    """
    wanted = set(names)
    return [column for column in table.feature_columns if column in wanted]


def _correlated_table():
    """
    Build a table holding one exact duplicate and one moderately correlated
    column, so a 0.8 cut-off and a 0.95 cut-off give DIFFERENT answers.

    Returns:
        FeatureTable: ``signal``, its exact rescaled copy ``signal_copy``
        (correlation 1.0) and ``signal_mixed`` (correlation between 0.8 and
        0.95), plus the noise columns.
    """
    table = make_feature_table(seed=3)
    frame = table.frame.copy()
    frame["signal_copy"] = frame["signal"] * 3.0
    frame["signal_mixed"] = frame["signal"] + 1.2 * frame["noise0"]
    return type(table)(
        frame=frame,
        id_columns=table.id_columns,
        feature_columns=(
            "signal",
            "signal_copy",
            "signal_mixed",
            "noise0",
            "noise1",
            "noise2",
        ),
        outcome=table.outcome,
        provenance=table.provenance,
    )


# ---------------------------------------------------------------------------
# Both names still resolve
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_four_registered_names_remain_available() -> None:
    """
    The convergence removed an implementation, never a name.

    Red line 3 (old YAML keeps running) means a config naming any of the four
    must still resolve, in the registry it always resolved from.
    """
    assert "variance" in FeatureSelectorRegistry.available()
    assert "correlation" in FeatureSelectorRegistry.available()
    assert "variance_filter" in TablePreprocessorRegistry.available()
    assert "correlation_filter" in TablePreprocessorRegistry.available()
    assert isinstance(
        FeatureSelectorRegistry.create("variance"), VarianceSelector
    )
    assert isinstance(
        TablePreprocessorRegistry.create("variance_filter"),
        VarianceFilterPreprocessor,
    )


# ---------------------------------------------------------------------------
# Numerical equivalence with the pre-convergence code
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0.0, 0.01, 0.5, 1.0])
def test_variance_selector_matches_its_pre_convergence_numbers(
    threshold: float,
) -> None:
    """The threshold mode selects exactly what the old selector selected."""
    table = _spread_table()
    block = table.frame[list(table.feature_columns)]
    selected = VarianceSelector(threshold=threshold).fit(table).selected_columns_
    assert list(selected) == _in_table_order(
        table, _v1_selector_variance(block, threshold)
    )


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [1, 2, 3, 99])
def test_variance_selector_top_k_matches_pre_convergence_numbers(
    top_k: int,
) -> None:
    """
    ``top_k`` keeps the same columns as the old ranked mode.

    Both sides go through :func:`_in_table_order`, which is where the base
    class puts the selection anyway; what is being locked here is the SET the
    variance ranking produces, including the ``min(top_k, n_columns)`` clamp
    that the ``99`` case exercises.
    """
    table = _spread_table()
    block = table.frame[list(table.feature_columns)]
    selected = VarianceSelector(top_k=top_k).fit(table).selected_columns_
    assert list(selected) == _in_table_order(
        table, _v1_selector_variance(block, 0.0, top_k=top_k)
    )


@pytest.mark.unit
@pytest.mark.parametrize("top_percent", [10.0, 33.0, 50.0, 100.0])
def test_variance_selector_top_percent_matches_pre_convergence_numbers(
    top_percent: float,
) -> None:
    """``top_percent`` rounds up to the same column count as before."""
    table = _spread_table()
    block = table.frame[list(table.feature_columns)]
    selected = VarianceSelector(top_percent=top_percent).fit(table).selected_columns_
    assert list(selected) == _in_table_order(
        table, _v1_selector_variance(block, 0.0, top_percent=top_percent)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "top_k,top_percent", [(2, None), (3, None), (None, 33.0), (None, 50.0)]
)
def test_kernel_ranked_modes_keep_the_descending_variance_order(
    top_k: Optional[int], top_percent: Optional[float]
) -> None:
    """
    At the kernel boundary the ranked modes still rank.

    The selector's base class re-orders the result, which would mask a broken
    ranking inside the kernel; the kernel is therefore checked against the
    old selector's RAW output, order included, so a future caller that does
    not re-order still gets ranked names.
    """
    table = _spread_table()
    block = table.frame[list(table.feature_columns)]
    assert kernel.select_variance_columns(
        block, 0.0, top_k=top_k, top_percent=top_percent
    ) == _v1_selector_variance(block, 0.0, top_k=top_k, top_percent=top_percent)


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0.0, 0.01, 0.5, 1.0])
def test_variance_filter_matches_its_pre_convergence_numbers(
    threshold: float,
) -> None:
    """The preprocessor keeps exactly the columns the old kernel kept."""
    table = _spread_table()
    block = table.frame[list(table.feature_columns)]
    fitted = VarianceFilterPreprocessor(variance_threshold=threshold).fit(table)
    kept = fitted.transform(table).feature_columns
    assert list(kept) == _v1_kernel_variance(block, threshold)


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0.5, 0.8, 0.95, 0.99])
def test_correlation_selector_matches_its_pre_convergence_numbers(
    threshold: float,
) -> None:
    """
    The greedy walk survives the move into the kernel unchanged.

    The pre-convergence selector and kernel were two transcriptions of the
    same walk; comparing against the selector's version proves the kernel is
    a faithful replacement of BOTH.
    """
    table = _correlated_table()
    block = table.frame[list(table.feature_columns)]
    selected = (
        CorrelationSelector(threshold=threshold, method="spearman")
        .fit(table)
        .selected_columns_
    )
    assert list(selected) == _v1_selector_correlation(block, threshold, "spearman")


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0.5, 0.8, 0.95, 0.99])
def test_correlation_filter_matches_its_pre_convergence_numbers(
    threshold: float,
) -> None:
    """The preprocessor's walk agrees with the same reference walk."""
    table = _correlated_table()
    block = table.frame[list(table.feature_columns)]
    fitted = CorrelationFilterPreprocessor(corr_threshold=threshold).fit(table)
    kept = fitted.transform(table).feature_columns
    assert list(kept) == _v1_selector_correlation(block, threshold, "spearman")


# ---------------------------------------------------------------------------
# The two names agree once their parameters are made equivalent
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0.0, 0.01, 0.5])
def test_the_two_variance_names_agree_at_equivalent_parameters(
    threshold: float,
) -> None:
    """
    One implementation means one answer once the parameters are aligned.

    This is the point of the convergence: the remaining differences are the
    two documented ones (parameter spelling, fallback default), and nothing
    else.
    """
    table = _spread_table()
    selector = VarianceSelector(threshold=threshold, keep_at_least_one=True)
    preprocessor = VarianceFilterPreprocessor(
        variance_threshold=threshold, keep_at_least_one=True
    )
    assert list(selector.fit(table).selected_columns_) == list(
        preprocessor.fit(table).transform(table).feature_columns
    )


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0.5, 0.8, 0.95])
def test_the_two_correlation_names_agree_at_equivalent_parameters(
    threshold: float,
) -> None:
    """Aligning ``threshold`` with ``corr_threshold`` aligns the answers."""
    table = _correlated_table()
    selector = CorrelationSelector(threshold=threshold, method="spearman")
    preprocessor = CorrelationFilterPreprocessor(
        corr_threshold=threshold, corr_method="spearman"
    )
    assert list(selector.fit(table).selected_columns_) == list(
        preprocessor.fit(table).transform(table).feature_columns
    )


# ---------------------------------------------------------------------------
# The degenerate case: the fallback difference is preserved on both sides
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_variance_filter_keeps_the_highest_variance_column_when_none_survive() -> None:
    """
    The v0.1 fallback is intact: a preprocessing chain never empties.

    Without this, ``variance_filter`` on an all-constant block would hand the
    next step a zero-column frame and the run would die somewhere unrelated
    (``check_array``: "at least one array or dtype is required"), which is
    exactly the failure mode the v0.1 rule was written to prevent.
    """
    table = _all_constant_table()
    kept = VarianceFilterPreprocessor().fit(table).transform(table).feature_columns
    assert len(kept) == 1
    block = table.frame[list(table.feature_columns)]
    assert list(kept) == _v1_kernel_variance(block, 0.0)


@pytest.mark.unit
def test_variance_selector_still_selects_nothing_when_none_survive() -> None:
    """
    The selector's opposite behaviour is equally intact.

    "No feature clears this threshold" is a legitimate finding for a
    selector, and silently keeping one column would fabricate a feature the
    user's threshold rejected.
    """
    table = _all_constant_table()
    selected = VarianceSelector(threshold=0.0).fit(table).selected_columns_
    assert list(selected) == []


@pytest.mark.unit
def test_the_fallback_is_reachable_from_both_names() -> None:
    """
    ``keep_at_least_one`` is a parameter, not a hard-wired per-name rule.

    A user who wants the preprocessor's guarantee from the selector (or the
    selector's honesty from the preprocessor) can ask for it, which is what
    makes one implementation able to serve both names.
    """
    table = _all_constant_table()
    forced = VarianceSelector(threshold=0.0, keep_at_least_one=True)
    assert len(forced.fit(table).selected_columns_) == 1
    honest = VarianceFilterPreprocessor(keep_at_least_one=False)
    assert honest.fit(table).transform(table).feature_columns == ()


@pytest.mark.unit
def test_kernel_returns_empty_for_an_empty_block() -> None:
    """
    An empty matrix cannot yield a fallback column.

    The fallback indexes position 0 of the variance series; guarding the
    empty case keeps that from becoming an ``IndexError`` on a table whose
    feature columns were all removed upstream.
    """
    empty = pd.DataFrame(index=range(5))
    assert kernel.select_variance_columns(empty, 0.0) == []
    assert kernel.select_variance_columns(empty, 0.0, keep_at_least_one=False) == []


# ---------------------------------------------------------------------------
# Fingerprint stability: the new parameter is invisible at its default
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_specs_do_not_mention_the_new_parameter() -> None:
    """
    Hard constraint: adding a parameter must not move any fingerprint.

    A ``Spec`` payload is hashed into every provenance record and golden
    baseline, so the new key is recorded only when it DEVIATES from the
    per-name default. These payloads are the literal pre-convergence ones.
    """
    assert VarianceSelector().spec.params == {
        "threshold": 0.0,
        "top_k": None,
        "top_percent": None,
    }
    assert VarianceFilterPreprocessor().spec.params == {"variance_threshold": 0.0}
    assert CorrelationSelector().spec.params == {
        "threshold": 0.8,
        "method": "spearman",
    }
    assert CorrelationFilterPreprocessor().spec.params == {
        "corr_threshold": 0.95,
        "corr_method": "spearman",
    }


@pytest.mark.unit
def test_explicit_defaults_also_leave_the_fingerprint_untouched() -> None:
    """
    Passing the default explicitly is indistinguishable from omitting it.

    Otherwise a YAML that spelled out ``keep_at_least_one: false`` would
    fingerprint differently from one that left it out, while running the
    identical computation.
    """
    assert (
        VarianceSelector(keep_at_least_one=False).spec.fingerprint()
        == VarianceSelector().spec.fingerprint()
    )
    assert (
        VarianceFilterPreprocessor(keep_at_least_one=True).spec.fingerprint()
        == VarianceFilterPreprocessor().spec.fingerprint()
    )


@pytest.mark.unit
def test_a_deviating_fallback_is_recorded_and_changes_the_fingerprint() -> None:
    """
    The asymmetry hides the default, never a non-default choice.

    A run that flipped the fallback computed different numbers, so its
    provenance record must say so.
    """
    forced = VarianceSelector(keep_at_least_one=True)
    assert forced.spec.params["keep_at_least_one"] is True
    assert forced.spec.fingerprint() != VarianceSelector().spec.fingerprint()

    honest = VarianceFilterPreprocessor(keep_at_least_one=False)
    assert honest.spec.params["keep_at_least_one"] is False
    assert (
        honest.spec.fingerprint() != VarianceFilterPreprocessor().spec.fingerprint()
    )


@pytest.mark.unit
def test_the_new_parameter_round_trips_through_the_registries() -> None:
    """
    A YAML naming the parameter reaches the component, and back again.

    The registry checks the constructor signature, so an undeclared parameter
    is rejected at the same public boundary that creates the component.
    """
    selector = FeatureSelectorRegistry.create("variance", keep_at_least_one=True)
    assert selector.spec.params["keep_at_least_one"] is True
    assert "keep_at_least_one" in FeatureSelectorRegistry.constructor_signature(
        "variance"
    ).parameters

    preprocessor = TablePreprocessorRegistry.create(
        "variance_filter", keep_at_least_one=False
    )
    assert preprocessor.spec.params["keep_at_least_one"] is False
    assert "keep_at_least_one" in TablePreprocessorRegistry.constructor_signature(
        "variance_filter"
    ).parameters
