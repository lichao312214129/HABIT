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
"""Contract tests for the third-party parameter policy.

Covers the three rules the policy pins down (developer/api_upgrade
06/08 docs):

1. Every registered domain ``*Params`` model forbids unknown keys, so a typo
   in YAML or a ``Spec`` fails at validation time instead of being silently
   dropped.
2. Thin-wrapper components accept vendor keyword arguments through the
   reserved ``estimator_params`` key; the mapping is folded into
   ``spec.params`` (hence the fingerprint) whenever non-empty, and keys
   colliding with declared parameters, wrapper-fixed call arguments, or the
   HABIT-injected ``random_state`` are rejected at construction time.
3. Passthrough keys are validated against the vendor callable's signature at
   build/call time: a key recorded in the fingerprint must reach the vendor,
   never be dropped silently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

import pytest
from pydantic import BaseModel, ValidationError

from habit.exceptions import HABITAPIError
from habit.registry.core import ComponentRegistry
from habit.utils.estimator_utils import ESTIMATOR_PARAMS_KEY

from .conftest import make_feature_table, make_field


# ---------------------------------------------------------------------------
# Rule 1: every registered domain *Params model forbids unknown keys
# ---------------------------------------------------------------------------


def _domain_registries() -> List[Type[ComponentRegistry]]:
    """Import every L3 domain registry (importing registers the builtins)."""
    from habit.domain.assignment.registry import HabitatAssignerRegistry
    from habit.domain.classification.registry import ClassifierRegistry
    from habit.domain.evaluation.registry import MetricRegistry
    from habit.domain.evaluation.regression_registry import RegressionMetricRegistry
    from habit.domain.evaluation.survival_registry import SurvivalMetricRegistry
    from habit.domain.feature_preprocessing.registry import (
        FeaturePreprocessingMethodRegistry,
    )
    from habit.domain.feature_selection.registry import FeatureSelectorRegistry
    from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
    from habit.domain.habitat_model.registry import HabitatModelFitterRegistry
    from habit.domain.regression.registry import RegressorRegistry
    from habit.domain.supervoxel.registry import SupervoxelizerRegistry
    from habit.domain.supervoxel_features.registry import (
        SupervoxelFeatureExtractorRegistry,
    )
    from habit.domain.survival.registry import SurvivalModelRegistry
    from habit.domain.table_preprocessing.registry import TablePreprocessorRegistry
    from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry

    return [
        HabitatAssignerRegistry,
        ClassifierRegistry,
        MetricRegistry,
        RegressionMetricRegistry,
        SurvivalMetricRegistry,
        FeaturePreprocessingMethodRegistry,
        FeatureSelectorRegistry,
        HabitatFeatureExtractorRegistry,
        HabitatModelFitterRegistry,
        RegressorRegistry,
        SupervoxelizerRegistry,
        SupervoxelFeatureExtractorRegistry,
        SurvivalModelRegistry,
        TablePreprocessorRegistry,
        VoxelFeatureExtractorRegistry,
    ]


@pytest.mark.unit
def test_every_registered_params_model_forbids_unknown_keys() -> None:
    """All domain params schemas use ``extra='forbid'`` (no silent drops)."""
    missing: List[str] = []
    not_forbidding: List[str] = []
    n_checked = 0
    for registry in _domain_registries():
        for name in registry.available():
            model = registry.get_params_model(name)
            label = f"{registry.__name__}:{name}"
            if model is None:
                missing.append(label)
                continue
            assert issubclass(model, BaseModel), label
            if model.model_config.get("extra") != "forbid":
                not_forbidding.append(label)
            n_checked += 1
    assert not missing, f"components without a params model: {missing}"
    assert not not_forbidding, f"params models not forbidding extras: {not_forbidding}"
    assert n_checked > 0


# ---------------------------------------------------------------------------
# Rule 1 on the pilot schemas: unknown keys raise at validation time
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pilot_params_models_reject_unknown_keys() -> None:
    """A typo in a pilot component's params fails as a ValidationError."""
    from habit.domain.classification.models import LogisticRegressionClassifierParams
    from habit.domain.feature_selection.selectors import LassoSelectorParams
    from habit.domain.supervoxel.slic import SlicSupervoxelizerParams

    for model in (
        SlicSupervoxelizerParams,
        LogisticRegressionClassifierParams,
        LassoSelectorParams,
    ):
        with pytest.raises(ValidationError):
            model(**{"n_supervoxel": 10})  # typo of a plausible key


# ---------------------------------------------------------------------------
# Rule 2: passthrough lands in the fingerprint; conflicts fail at construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_passthrough_enters_spec_params_and_fingerprint() -> None:
    """Non-empty estimator_params change the fingerprint; empty ones do not."""
    from habit.domain.supervoxel.slic import SlicSupervoxelizer

    plain = SlicSupervoxelizer(n_supervoxels=8)
    empty = SlicSupervoxelizer(n_supervoxels=8, estimator_params={})
    tuned = SlicSupervoxelizer(n_supervoxels=8, estimator_params={"sigma": 1.0})

    assert ESTIMATOR_PARAMS_KEY not in plain.spec.params
    assert plain.spec.fingerprint() == empty.spec.fingerprint()
    assert tuned.spec.params[ESTIMATOR_PARAMS_KEY] == {"sigma": 1.0}
    assert tuned.spec.fingerprint() != plain.spec.fingerprint()


@pytest.mark.unit
def test_passthrough_conflicts_fail_at_construction() -> None:
    """Declared / fixed / HABIT-injected keys are rejected with ownership."""
    from habit.domain.supervoxel.slic import SlicSupervoxelizer

    with pytest.raises(HABITAPIError, match="compactness"):
        SlicSupervoxelizer(estimator_params={"compactness": 5.0})
    with pytest.raises(HABITAPIError, match="start_label"):
        SlicSupervoxelizer(estimator_params={"start_label": 0})
    with pytest.raises(HABITAPIError, match="random_state"):
        SlicSupervoxelizer(estimator_params={"random_state": 0})


# ---------------------------------------------------------------------------
# Rule 3: keys unknown to the vendor callable fail at build/call time
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_slic_rejects_vendor_unknown_key_at_call_time() -> None:
    """A passthrough typo reaches neither skimage nor the fingerprint silently."""
    from habit.domain.supervoxel.slic import SlicSupervoxelizer

    field = make_field()
    # ``n_segment`` is not the HABIT-fixed ``n_segments``: construction passes,
    # the vendor signature check must fail the call with a did-you-mean hint.
    slicer = SlicSupervoxelizer(n_supervoxels=4, estimator_params={"n_segment": 3})
    with pytest.raises(HABITAPIError, match="n_segment"):
        slicer(field)


@pytest.mark.unit
def test_slic_end_to_end_with_passthrough() -> None:
    """A real vendor kwarg (``max_num_iter``) runs and is fingerprinted."""
    from habit.domain.supervoxel.slic import SlicSupervoxelizer

    field = make_field()
    slicer = SlicSupervoxelizer(n_supervoxels=4, estimator_params={"max_num_iter": 5})
    result = slicer(field)
    assert result.label_array.shape == tuple(field.geometry.shape)
    assert slicer.spec.params[ESTIMATOR_PARAMS_KEY] == {"max_num_iter": 5}
    # The provenance carries the same fingerprint the spec reports.
    assert result.provenance.spec_fingerprint == slicer.spec.fingerprint()


# ---------------------------------------------------------------------------
# sklearn classifier pilot
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classifier_passthrough_reaches_estimator_and_fingerprint() -> None:
    """``fit_intercept=False`` must reach sklearn and appear in the spec."""
    from habit.domain.classification.models import LogisticRegressionClassifier

    table = make_feature_table(tuple(f"S{i}" for i in range(20)), seed=7)
    clf = LogisticRegressionClassifier(estimator_params={"fit_intercept": False})
    # Seed injection stays outside the constructor (v1.0 naming decisions).
    clf.set_random_state(42)
    clf.fit(table)
    assert clf._estimator.fit_intercept is False
    assert clf.spec.params[ESTIMATOR_PARAMS_KEY] == {"fit_intercept": False}
    assert clf.spec.fingerprint() != LogisticRegressionClassifier().spec.fingerprint()
    predictions = clf.predict(table)
    assert len(predictions) == len(table.frame)


@pytest.mark.unit
def test_classifier_passthrough_conflicts_fail() -> None:
    """Declared keys and the injected seed cannot hide in the passthrough."""
    from habit.domain.classification.models import LogisticRegressionClassifier

    with pytest.raises(HABITAPIError, match="'C'"):
        LogisticRegressionClassifier(estimator_params={"C": 2.0})
    with pytest.raises(HABITAPIError, match="random_state"):
        LogisticRegressionClassifier(estimator_params={"random_state": 0})


@pytest.mark.unit
def test_classifier_vendor_unknown_key_fails_at_fit() -> None:
    """sklearn rejects nothing silently: a bogus key fails at fit time."""
    from habit.domain.classification.models import LogisticRegressionClassifier

    table = make_feature_table(tuple(f"S{i}" for i in range(20)), seed=7)
    clf = LogisticRegressionClassifier(estimator_params={"not_a_param": 1})
    with pytest.raises(HABITAPIError, match="not_a_param"):
        clf.fit(table)


# ---------------------------------------------------------------------------
# Feature-selector pilot
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lasso_passthrough_reaches_lassocv_and_fingerprint() -> None:
    """A vendor kwarg (``eps``) runs, is fingerprinted, and keeps the signal."""
    from habit.domain.feature_selection.selectors import LassoSelector

    table = make_feature_table(tuple(f"S{i}" for i in range(30)), n_noise=2, seed=12)
    selector = LassoSelector(cv=3, n_jobs=1, estimator_params={"eps": 1e-3})
    selector.set_random_state(3)
    kept = selector.fit(table).transform(table).feature_columns
    assert "signal" in kept
    assert selector.spec.params[ESTIMATOR_PARAMS_KEY] == {"eps": 1e-3}
    assert selector.spec.fingerprint() != LassoSelector().spec.fingerprint()


@pytest.mark.unit
def test_lasso_passthrough_conflicts_fail() -> None:
    """Declared keys and the injected seed cannot hide in the passthrough."""
    from habit.domain.feature_selection.selectors import LassoSelector

    with pytest.raises(HABITAPIError, match="'cv'"):
        LassoSelector(estimator_params={"cv": 3})
    with pytest.raises(HABITAPIError, match="random_state"):
        LassoSelector(estimator_params={"random_state": 0})


@pytest.mark.unit
def test_lasso_vendor_unknown_key_fails_at_fit() -> None:
    """A bogus passthrough key fails at fit time with ownership context."""
    from habit.domain.feature_selection.selectors import LassoSelector

    table = make_feature_table(tuple(f"S{i}" for i in range(30)), n_noise=2, seed=12)
    selector = LassoSelector(cv=3, n_jobs=1, estimator_params={"not_a_param": 1})
    with pytest.raises(HABITAPIError, match="not_a_param"):
        selector.fit(table)
