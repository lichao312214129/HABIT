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
"""Feature-preprocessing tests: v0.1 numerical parity and chain semantics.

The parity tests are the load-bearing ones. A refactor that quietly changes a
scaling formula would still pass every structural test while making every
published habitat definition irreproducible, so each of the eight methods is
compared against the v0.1 implementation on the same input, at full float
precision.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import HABITAPIError
from habit.compat.engines.habitat_analysis.feature_preprocessing.pipeline import (
    apply_preprocessing_pipeline,
)
from habit.domain.feature_preprocessing import (
    CohortPreprocessingChain,
    FeaturePreprocessingMethodRegistry,
    FeatureWhitelist,
    Impute,
    SubjectPreprocessingChain,
    build_methods,
)
from habit.spec.specs import Spec


def _block(rows: int = 40, columns: int = 4, seed: int = 7) -> pd.DataFrame:
    """
    Build a reproducible unit-by-feature matrix with awkward but finite values.

    Args:
        rows: Number of clustering units.
        columns: Number of features.
        seed: Generator seed.

    Returns:
        A matrix mixing scales, a heavy tail and a constant column, so that
        scaling, filtering and discretising all have something to act on.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=5.0, scale=2.0, size=(rows, columns))
    data[:, 0] *= 100.0
    data[0, 1] = data[:, 1].max() * 25.0
    frame = pd.DataFrame(data, columns=[f"f{index}" for index in range(columns)])
    frame["const"] = 3.0
    return frame


#: v0.1 spellings of the eight methods with parameters worth pinning. The
#: ``global_normalize`` variants matter most: that flag is renamed in v1, so a
#: translation slip would show up here as a numerical difference rather than as
#: an import error.
_PARITY_CASES: List[Dict[str, Any]] = [
    {"method": "minmax"},
    {"method": "minmax", "global_normalize": True},
    {"method": "zscore"},
    {"method": "zscore", "global_normalize": True},
    {"method": "robust"},
    {"method": "robust", "global_normalize": True},
    {"method": "log"},
    {"method": "log", "global_normalize": True},
    {"method": "winsorize", "winsor_limits": [0.05, 0.05]},
    {"method": "winsorize", "winsor_limits": [0.1, 0.2], "global_normalize": True},
    {"method": "binning", "n_bins": 5, "bin_strategy": "uniform"},
    {"method": "binning", "n_bins": 4, "bin_strategy": "quantile"},
    {"method": "variance_filter", "variance_threshold": 0.0},
    {"method": "variance_filter", "variance_threshold": 100.0},
    {"method": "correlation_filter", "corr_threshold": 0.9, "corr_method": "spearman"},
    {"method": "correlation_filter", "corr_threshold": 0.5, "corr_method": "pearson"},
]


def _v1_method(v0_case: Dict[str, Any]) -> Any:
    """
    Instantiate the v1 method matching a v0.1 method configuration.

    Args:
        v0_case: v0.1 method entry, using v0.1 parameter spellings.

    Returns:
        The corresponding v1 method instance.
    """
    params = dict(v0_case)
    name = params.pop("method")
    if "global_normalize" in params:
        params["across_features"] = params.pop("global_normalize")
    return FeaturePreprocessingMethodRegistry.create(name, **params)


@pytest.mark.unit
@pytest.mark.parametrize("v0_case", _PARITY_CASES, ids=lambda case: str(case))
def test_each_method_reproduces_v01_numerically(v0_case: Dict[str, Any]) -> None:
    """Every method matches v0.1 bit for bit on identical finite input."""
    block = _block()
    v0_output, _, _ = apply_preprocessing_pipeline(
        block.copy(), [dict(v0_case)], fit=True
    )
    v1_output = SubjectPreprocessingChain([_v1_method(v0_case)])(block)
    # Column-filtering methods legitimately return a subset; compare on the
    # v0.1 result's columns so a parity failure is about VALUES, and let the
    # column sets be asserted separately.
    assert list(v1_output.columns) == list(v0_output.columns)
    pd.testing.assert_frame_equal(
        v1_output, v0_output[list(v1_output.columns)], check_dtype=False
    )


@pytest.mark.unit
def test_full_v01_subject_chain_reproduces_v01() -> None:
    """A realistic multi-step subject chain matches v0.1 end to end."""
    block = _block(seed=11)
    v0_chain = [
        {"method": "winsorize", "winsor_limits": [0.05, 0.05]},
        {"method": "minmax"},
    ]
    v0_output, _, _ = apply_preprocessing_pipeline(block.copy(), v0_chain, fit=True)
    chain = SubjectPreprocessingChain([_v1_method(case) for case in v0_chain])
    pd.testing.assert_frame_equal(chain(block), v0_output, check_dtype=False)


@pytest.mark.unit
def test_full_v01_cohort_chain_reproduces_v01_fit_and_transform() -> None:
    """A cohort chain matches v0.1 both when fitting and when replaying."""
    train = _block(rows=60, seed=3)
    test = _block(rows=25, seed=99)
    v0_chain = [
        {"method": "winsorize", "winsor_limits": [0.05, 0.05]},
        {"method": "variance_filter", "variance_threshold": 0.0},
        {"method": "zscore"},
    ]
    v0_train, baseline, states = apply_preprocessing_pipeline(
        train.copy(), v0_chain, fit=True
    )
    v0_test, _, _ = apply_preprocessing_pipeline(
        test.copy(), v0_chain, step_states=states, baseline=baseline, fit=False
    )

    chain = CohortPreprocessingChain([_v1_method(case) for case in v0_chain])
    v1_train = chain.fit_transform(train)
    v1_test = chain.transform(test)
    pd.testing.assert_frame_equal(v1_train, v0_train, check_dtype=False)
    pd.testing.assert_frame_equal(v1_test, v0_test, check_dtype=False)


def _block_float32(rows: int = 80, columns: int = 6, seed: int = 11) -> pd.DataFrame:
    """
    Build a float32 unit-by-feature matrix like voxel radiomics tables.

    Args:
        rows: Number of clustering units.
        columns: Number of features.
        seed: Generator seed.

    Returns:
        A float32 matrix. v0.1 writes radiomics features as float32
        (``output_float32=True``); promoting them before cohort z-score is
        exactly the ck3 drift mode that rtol=1e-6 parity rejects.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=5.0, scale=2.0, size=(rows, columns)).astype(np.float32)
    data[:, 0] *= 100.0
    data[0, 1] = float(data[:, 1].max() * 25.0)
    return pd.DataFrame(data, columns=[f"f{index}" for index in range(columns)])


@pytest.mark.unit
def test_float32_cohort_zscore_matches_v01_within_rtol_1e_6() -> None:
    """
    Cohort z-score on float32 input matches v0.1 within rtol=1e-6.

    This is the voxel-radiomics ck3 failure mode: raw float32 features enter
    a group-level ``variance_filter`` + ``zscore`` chain (no prior winsorize
    that would have promoted both paths to float64). v1 used to promote via
    ``apply_impute`` and float64 statistic Series, shifting means/stds enough
    that the matrix entering k-means exceeded rtol=1e-6 while labels still
    agreed.
    """
    train = _block_float32()
    v0_chain = [
        {"method": "variance_filter", "variance_threshold": 0.01},
        {"method": "zscore"},
    ]
    v0_train, _, _ = apply_preprocessing_pipeline(train.copy(), v0_chain, fit=True)
    v1_train = CohortPreprocessingChain(
        [_v1_method(case) for case in v0_chain]
    ).fit_transform(train)

    assert list(v1_train.columns) == list(v0_train.columns)
    assert v1_train.dtypes.equals(v0_train.dtypes)
    expected = v0_train.to_numpy(dtype=np.float64)
    actual = v1_train.to_numpy(dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)
    pd.testing.assert_frame_equal(v1_train, v0_train, check_dtype=True)


@pytest.mark.unit
def test_float32_subject_minmax_then_cohort_zscore_matches_v01() -> None:
    """
    Subject minmax (float32-preserving) then cohort z-score matches v0.1.

    Winsorize is deliberately omitted: it promotes both stacks to float64 via
    quantile bounds and hides the dtype bug. minmax keeps float32 in v0.1, so
    the cohort z-score must see the same dtype and reduction order in v1.
    """
    voxels = _block_float32(rows=200, columns=5, seed=3)
    subject_chain = [{"method": "minmax"}]
    cohort_chain = [
        {"method": "variance_filter", "variance_threshold": 0.01},
        {"method": "zscore"},
    ]

    v0_subject, _, _ = apply_preprocessing_pipeline(
        voxels.copy(), subject_chain, fit=True
    )
    v1_subject = SubjectPreprocessingChain(
        [_v1_method(case) for case in subject_chain]
    )(voxels)
    pd.testing.assert_frame_equal(v1_subject, v0_subject, check_dtype=True)

    v0_cohort, _, _ = apply_preprocessing_pipeline(
        v0_subject.copy(), cohort_chain, fit=True
    )
    v1_cohort = CohortPreprocessingChain(
        [_v1_method(case) for case in cohort_chain]
    ).fit_transform(v1_subject)
    np.testing.assert_allclose(
        v1_cohort.to_numpy(dtype=np.float64),
        v0_cohort.to_numpy(dtype=np.float64),
        rtol=1e-6,
        atol=0.0,
    )
    pd.testing.assert_frame_equal(v1_cohort, v0_cohort, check_dtype=True)


@pytest.mark.unit
def test_subject_chain_is_stateless_across_calls() -> None:
    """Each call recomputes statistics, so call order cannot leak."""
    chain = SubjectPreprocessingChain(build_methods([{"name": "minmax"}]))
    first = _block(seed=1)
    second = _block(seed=2) * 1000.0
    baseline_first = chain(first)
    chain(second)
    pd.testing.assert_frame_equal(chain(first), baseline_first)
    # Statelessness is visible in the result: each matrix maps onto [0, 1]
    # using its OWN range, which is what removes between-subject variation.
    assert chain(second).to_numpy().max() == pytest.approx(1.0)


@pytest.mark.unit
def test_cohort_chain_replays_training_statistics() -> None:
    """A fitted cohort chain does NOT rescale a new matrix by its own range."""
    chain = CohortPreprocessingChain(build_methods([{"name": "minmax"}]))
    chain.fit(_block(seed=1))
    doubled = _block(seed=1) * 2.0
    output = chain.transform(doubled)
    # Replaying training bounds must push a uniformly larger matrix ABOVE 1;
    # a value of exactly 1.0 would mean the chain had refitted itself.
    assert output.to_numpy().max() > 1.0


@pytest.mark.unit
def test_cohort_chain_requires_fitting_before_transform() -> None:
    """Transforming an unfitted cohort chain is an error, not a silent refit."""
    chain = CohortPreprocessingChain(build_methods([{"name": "zscore"}]))
    with pytest.raises(HABITAPIError, match="must be fitted"):
        chain.transform(_block())


@pytest.mark.unit
def test_cohort_chain_state_roundtrip_reproduces_transform() -> None:
    """A restored chain transforms identically, so a model stays portable."""
    chain = CohortPreprocessingChain(
        build_methods(
            [
                {"name": "winsorize", "params": {"winsor_limits": [0.05, 0.05]}},
                {"name": "binning", "params": {"n_bins": 4}},
            ]
        )
    )
    train = _block(seed=5)
    chain.fit(train)
    expected = chain.transform(train)
    restored = CohortPreprocessingChain.from_state(chain.state)
    pd.testing.assert_frame_equal(restored.transform(train), expected)
    assert restored.output_columns == chain.output_columns


#: Every method that learns state, with parameters that exercise the
#: per-column branch (``across_features=False``) where the state is a mapping.
_STATEFUL_CASES: List[Dict[str, Any]] = [
    {"name": "impute", "params": {"strategy": "median"}},
    {"name": "minmax"},
    {"name": "minmax", "params": {"across_features": True}},
    {"name": "zscore"},
    {"name": "robust"},
    {"name": "log"},
    {"name": "winsorize", "params": {"winsor_limits": [0.05, 0.05]}},
    {"name": "binning", "params": {"n_bins": 4, "bin_strategy": "uniform"}},
    {"name": "binning", "params": {"n_bins": 3, "bin_strategy": "quantile"}},
    {"name": "binning", "params": {"n_bins": 3, "bin_strategy": "kmeans"}},
    {"name": "variance_filter"},
    {"name": "correlation_filter"},
]


@pytest.mark.unit
@pytest.mark.parametrize("case", _STATEFUL_CASES, ids=lambda case: str(case))
def test_fitted_state_is_plain_json(case: Dict[str, Any]) -> None:
    """Every method's state survives ``json.dumps`` unaided.

    A cohort chain's state travels inside ``HabitatModel``, and that artefact
    is deliberately not a bare pickle. A method holding a ``Series`` or a
    fitted scikit-learn object would make every model that uses it unsavable,
    which is precisely the models worth sharing.
    """
    chain = CohortPreprocessingChain(build_methods([case]))
    chain.fit(_block())
    encoded = json.dumps(chain.state)
    # Round-tripping through JSON must not change what the chain computes.
    restored = CohortPreprocessingChain.from_state(json.loads(encoded))
    pd.testing.assert_frame_equal(
        restored.transform(_block()), chain.transform(_block())
    )


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["uniform", "quantile", "kmeans"])
def test_binning_matches_kbinsdiscretizer_from_edges_alone(strategy: str) -> None:
    """Binning reproduces scikit-learn while storing only bin edges.

    The state keeps edges rather than the fitted ``KBinsDiscretizer`` so a
    shared model never depends on unpickling another version's estimator. That
    only holds up if the edge-based reimplementation is exact.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    train = _block(seed=3)
    reference = KBinsDiscretizer(
        n_bins=5, encode="ordinal", strategy=strategy, random_state=0
    ).fit(train.to_numpy())

    method = FeaturePreprocessingMethodRegistry.create(
        "binning", n_bins=5, bin_strategy=strategy
    )
    if hasattr(method, "set_random_state"):
        method.set_random_state(0)
    chain = CohortPreprocessingChain([method])
    chain.fit(train)

    np.testing.assert_array_equal(
        chain.transform(train).to_numpy(), reference.transform(train.to_numpy())
    )
    # Values outside the fitted range must land in the extreme bins rather
    # than in invented ones -- prediction cohorts routinely exceed it.
    extreme = train.copy()
    extreme.iloc[0] = -1e6
    extreme.iloc[1] = 1e6
    np.testing.assert_array_equal(
        chain.transform(extreme).to_numpy(),
        reference.transform(extreme.to_numpy()),
    )


@pytest.mark.unit
def test_cohort_chain_rejects_matrix_missing_fitted_columns() -> None:
    """A missing feature column is named, not silently dropped."""
    chain = CohortPreprocessingChain(build_methods([{"name": "zscore"}]))
    train = _block()
    chain.fit(train)
    with pytest.raises(HABITAPIError, match="missing"):
        chain.transform(train.drop(columns=["f0"]))


@pytest.mark.unit
def test_both_chains_prepend_impute_and_record_it() -> None:
    """Imputation is inserted when unnamed, and appears in the spec."""
    for chain in (
        SubjectPreprocessingChain(build_methods([{"name": "minmax"}])),
        CohortPreprocessingChain(build_methods([{"name": "minmax"}])),
    ):
        assert [method.spec.name for method in chain.methods] == ["impute", "minmax"]
        steps = chain.spec.to_dict()["params"]["steps"]
        assert [step["name"] for step in steps] == ["impute", "minmax"]


@pytest.mark.unit
def test_explicit_impute_keeps_its_configured_position_and_strategy() -> None:
    """Naming impute explicitly overrides both the insertion and the default."""
    chain = SubjectPreprocessingChain(
        build_methods(
            [
                {"name": "winsorize"},
                {"name": "impute", "params": {"strategy": "median"}},
            ]
        )
    )
    names = [method.spec.name for method in chain.methods]
    assert names == ["winsorize", "impute"]
    assert chain.methods[1].spec.params["strategy"] == "median"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("mean", 3.0), ("median", 3.0), ("zero", 0.0)],
)
def test_impute_replaces_non_finite_with_the_requested_statistic(
    strategy: str, expected: float
) -> None:
    """Each strategy fills from the column's finite values as documented."""
    block = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.inf, 5.0]})
    method = Impute(strategy=strategy)
    filled = method.transform(block, method.fit(block))
    assert filled["a"].tolist() == [1.0, expected, 3.0, expected, 5.0]


@pytest.mark.unit
def test_impute_zeroes_a_column_with_no_finite_value() -> None:
    """A wholly degenerate column collapses to 0.0 instead of spreading NaN."""
    block = pd.DataFrame({"good": [1.0, 2.0], "dead": [np.nan, np.inf]})
    method = Impute()
    filled = method.transform(block, method.fit(block))
    assert filled["dead"].tolist() == [0.0, 0.0]
    assert filled["good"].tolist() == [1.0, 2.0]


@pytest.mark.unit
def test_cohort_impute_uses_training_statistics_not_the_new_matrix() -> None:
    """Non-finite values in new data are filled from the TRAINING statistics.

    This is the one intentional departure from v0.1, which replaced infinities
    using the mean of whichever block was being transformed -- letting test
    data influence its own transformation inside a stateful chain.
    """
    chain = CohortPreprocessingChain([Impute()])
    chain.fit(pd.DataFrame({"a": [0.0, 10.0]}))
    output = chain.transform(pd.DataFrame({"a": [100.0, np.nan, 300.0]}))
    assert output["a"].tolist() == [100.0, 5.0, 300.0]


class _InfinityMaker:
    """A method that manufactures an infinity, standing in for a bad config."""

    changes_columns: bool = False

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name="infinity_maker")

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """Learn nothing."""
        return {}

    def transform(
        self, block: pd.DataFrame, state: Dict[str, Any]
    ) -> pd.DataFrame:
        """Return the block with one poisoned entry."""
        poisoned = block.copy()
        poisoned.iloc[0, 1] = np.inf
        return poisoned


@pytest.mark.unit
def test_chain_rejects_non_finite_output_naming_the_column() -> None:
    """A method that manufactures infinity fails loudly, naming the column.

    The chain's input is imputed, so a non-finite output can only come from a
    method. v0.1 let such values reach scikit-learn and fail there, far from
    the configuration that caused them.
    """
    block = _block()
    with pytest.raises(HABITAPIError, match=r"non-finite.*'f1'"):
        SubjectPreprocessingChain([_InfinityMaker()])(block)
    chain = CohortPreprocessingChain([_InfinityMaker()])
    with pytest.raises(HABITAPIError, match="non-finite"):
        chain.fit(block)


@pytest.mark.unit
def test_empty_chain_is_rejected_rather_than_silently_doing_nothing() -> None:
    """An empty chain is a configuration mistake, not a no-op."""
    for chain_type in (SubjectPreprocessingChain, CohortPreprocessingChain):
        with pytest.raises(HABITAPIError, match="at least one method"):
            chain_type([])


@pytest.mark.unit
def test_chain_rejects_an_object_that_is_not_a_method() -> None:
    """A wrong plug-in type is reported with the attributes it lacks."""
    with pytest.raises(HABITAPIError, match="lacks"):
        SubjectPreprocessingChain([object()])


@pytest.mark.unit
def test_variance_filter_keeps_one_column_when_all_would_be_dropped() -> None:
    """The v0.1 rule: a chain never hands an empty matrix downstream."""
    block = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
    chain = SubjectPreprocessingChain(
        build_methods([{"name": "variance_filter", "params": {"variance_threshold": 1.0}}])
    )
    assert chain(block).shape[1] == 1


@pytest.mark.unit
def test_registry_exposes_every_v01_method_name() -> None:
    """A v0.1 configuration can name any method it always could."""
    assert set(FeaturePreprocessingMethodRegistry.available()) >= {
        "minmax",
        "zscore",
        "robust",
        "maxabs",
        "quantile",
        "l2",
        "log",
        "winsorize",
        "binning",
        "variance_filter",
        "correlation_filter",
    }


@pytest.mark.unit
def test_maxabs_quantile_l2_are_finite() -> None:
    """New scalers run on a mixed-scale block without NaN."""
    block = _block()
    maxabs = FeaturePreprocessingMethodRegistry.create("maxabs")
    quantile = FeaturePreprocessingMethodRegistry.create(
        "quantile", n_quantiles=32,
    )
    l2 = FeaturePreprocessingMethodRegistry.create("l2")
    maxabs_out = maxabs.transform(block, maxabs.fit(block))
    quantile_out = quantile.transform(block, quantile.fit(block))
    l2_out = l2.transform(block, l2.fit(block))
    assert np.isfinite(maxabs_out.to_numpy()).all()
    assert np.isfinite(quantile_out.to_numpy()).all()
    assert np.isfinite(l2_out.to_numpy()).all()
    np.testing.assert_allclose(
        maxabs_out.abs().max().to_numpy(),
        np.ones(maxabs_out.shape[1]),
    )
    norms = np.linalg.norm(l2_out.to_numpy(dtype=np.float64), axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-12)


@pytest.mark.unit
def test_method_specs_roundtrip_through_build_methods() -> None:
    """Chains rebuild from their own spec, which is what provenance requires."""
    original = SubjectPreprocessingChain(
        build_methods(
            [
                {"name": "winsorize", "params": {"winsor_limits": [0.02, 0.03]}},
                {"name": "robust", "params": {"across_features": True}},
            ]
        )
    )
    steps = original.spec.to_dict()["params"]["steps"]
    rebuilt = SubjectPreprocessingChain(
        build_methods([Spec.from_dict(step) for step in steps])
    )
    block = _block(seed=21)
    pd.testing.assert_frame_equal(rebuilt(block), original(block))
    assert rebuilt.spec.fingerprint() == original.spec.fingerprint()


@pytest.mark.unit
def test_feature_whitelist_keeps_exactly_the_named_columns_in_order() -> None:
    """The whitelist selects by name, not by position or data."""
    block = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}
    )
    method = FeatureWhitelist(["c", "a"])
    state = method.fit(block)
    out = method.transform(block, state)
    assert list(out.columns) == ["c", "a"]
    pd.testing.assert_series_equal(out["c"], block["c"])


@pytest.mark.unit
def test_feature_whitelist_rejects_an_empty_list() -> None:
    """A whitelist that keeps nothing is a configuration mistake."""
    with pytest.raises(HABITAPIError, match="must not be empty"):
        FeatureWhitelist([])


@pytest.mark.unit
def test_feature_whitelist_rejects_a_missing_feature() -> None:
    """A named feature absent from the matrix is an error, never a silent drop.

    The whitelist exists to guarantee that a habitat run clusters exactly the
    precise set; dropping a missing name would break that contract invisibly.
    """
    block = pd.DataFrame({"a": [1.0], "b": [2.0]})
    method = FeatureWhitelist(["a", "ghost"])
    with pytest.raises(HABITAPIError, match="ghost"):
        method.fit(block)
    with pytest.raises(HABITAPIError, match="ghost"):
        method.transform(block, method.fit(pd.DataFrame({"a": [1.0], "ghost": [0.0]})))


@pytest.mark.unit
def test_feature_whitelist_builds_through_the_registry() -> None:
    """The method is reachable from a spec/YAML payload like any other."""
    methods = build_methods(
        [{"name": "feature_whitelist", "params": {"features": ["b"]}}]
    )
    block = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = SubjectPreprocessingChain(methods)(block)
    assert list(out.columns) == ["b"]
    assert methods[0].spec.to_dict()["params"]["features"] == ["b"]


@pytest.mark.unit
def test_precise_correlation_filter_matches_prior_keep_last_rule() -> None:
    """Signed r>0.7 and p<0.05 drops the earlier column, not the later one."""
    rng = np.random.default_rng(1)
    n = 60
    early = rng.normal(size=n)
    later = early + rng.normal(scale=0.02, size=n)
    opposite = -early
    noise = rng.normal(size=n)
    block = pd.DataFrame(
        {"early": early, "later": later, "opposite": opposite, "noise": noise}
    )
    methods = build_methods(
        [{
            "name": "precise_correlation_filter",
            "params": {"corr_threshold": 0.7, "p_threshold": 0.05},
        }]
    )
    out = SubjectPreprocessingChain(methods)(block)
    assert "early" not in out.columns
    assert list(out.columns) == ["later", "opposite", "noise"]
