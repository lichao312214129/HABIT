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
The scikit-learn parameter protocol on HABIT's tabular domain components.

Three properties are locked down here, because breaking any of them fails
silently rather than loudly:

1. ``get_params()`` keys match the constructor's parameter names, so
   ``sklearn.base.clone`` can rebuild the component.
2. ``get_params()`` never drifts from ``spec.params``. A drift would make a
   hyperparameter search report an optimum the fingerprint says was never
   used.
3. ``set_params()`` changes the spec fingerprint AND the fitted result, so a
   searched value is really the value that ran.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from sklearn.base import clone

from habit.exceptions import HABITAPIError
from habit.domain.classification import ClassifierRegistry
from habit.domain.feature_selection import FeatureSelectorRegistry
from habit.domain.regression import RegressorRegistry
from habit.domain.survival import SurvivalModelRegistry
from habit.domain.table_preprocessing import TablePreprocessorRegistry
from habit.utils.estimator_utils import ESTIMATOR_PARAMS_KEY

from .conftest import make_feature_table

#: Constructor arguments for the few components with required parameters.
_REQUIRED_ARGS: Dict[str, Dict[str, Any]] = {
    "icc_precomputed": {"icc_results_path": "icc.json", "groups": ["group_a"]},
}


def _all_components() -> List[Tuple[str, Any]]:
    """
    Build one instance of every registered tabular component.

    Returns:
        List[Tuple[str, Any]]: ``(label, component)`` pairs where the label is
        ``"<domain>.<name>"``, covering classifiers, feature selectors, table
        preprocessors, regressors and survival models. Construction only --
        no optional heavy dependency is imported, because those are pulled in
        lazily at fit time.
    """
    registries = (
        ClassifierRegistry,
        FeatureSelectorRegistry,
        TablePreprocessorRegistry,
        RegressorRegistry,
        SurvivalModelRegistry,
    )
    components: List[Tuple[str, Any]] = []
    for registry in registries:
        for name in registry.available():
            arguments = _REQUIRED_ARGS.get(name, {})
            components.append(
                (f"{registry.domain}.{name}", registry.create(name, **arguments))
            )
    return components


_COMPONENTS = _all_components()
_IDS = [label for label, _ in _COMPONENTS]


@pytest.mark.unit
@pytest.mark.parametrize("label,component", _COMPONENTS, ids=_IDS)
def test_get_params_keys_are_constructor_parameters(label: str, component: Any) -> None:
    """Every ``get_params`` key names a constructor parameter, and vice versa."""
    signature = inspect.signature(type(component).__init__)
    declared = {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    assert set(component.get_params(deep=False)) == declared, label


@pytest.mark.unit
@pytest.mark.parametrize("label,component", _COMPONENTS, ids=_IDS)
def test_clone_round_trips_through_the_constructor(label: str, component: Any) -> None:
    """``clone`` rebuilds an equivalent component (same spec, same fingerprint)."""
    copy_ = clone(component)
    assert copy_ is not component, label
    assert type(copy_) is type(component), label
    assert copy_.spec.to_dict() == component.spec.to_dict(), label
    assert copy_.spec.fingerprint() == component.spec.fingerprint(), label


@pytest.mark.unit
@pytest.mark.parametrize("label,component", _COMPONENTS, ids=_IDS)
def test_get_params_covers_every_spec_parameter(label: str, component: Any) -> None:
    """
    ``spec.params`` is a subset of ``get_params`` with identical values.

    The two must agree value-by-value, since ``spec.params`` is what the
    fingerprint hashes and ``get_params`` is what a search reads and writes.
    Sequence parameters are compared after normalising tuple/list, which is
    the only representational freedom the specs take (``MLP`` serialises
    ``hidden_layer_sizes`` as a list for YAML round-tripping).
    """
    params = component.get_params(deep=False)
    for key, spec_value in component.spec.params.items():
        assert key in params, f"{label}: spec.params[{key!r}] is not a get_params key"
        actual = params[key]
        if isinstance(spec_value, (list, tuple)) and isinstance(actual, (list, tuple)):
            assert list(spec_value) == list(actual), f"{label}.{key}"
        else:
            assert spec_value == actual, f"{label}.{key}"


@pytest.mark.unit
@pytest.mark.parametrize("label,component", _COMPONENTS, ids=_IDS)
def test_clone_preserves_the_random_seed(label: str, component: Any) -> None:
    """
    A seeded component clones seeded.

    HABIT keeps the seed out of the constructor (v1.0 naming decisions), so
    scikit-learn's default ``clone`` would drop it and every
    cross-validation fold of a ``GridSearchCV`` would run unseeded -- wrong
    numbers, no error.
    """
    setter = getattr(component, "set_random_state", None)
    if not callable(setter):
        pytest.skip(f"{label} is deterministic")
    setter(1234)
    assert getattr(clone(component), "_seed", None) == 1234, label


@pytest.mark.unit
def test_set_params_rejects_unknown_names() -> None:
    """A misspelled parameter is an error, never a silently ignored keyword."""
    selector = FeatureSelectorRegistry.create("variance")
    with pytest.raises(HABITAPIError):
        selector.set_params(threshhold=0.5)


@pytest.mark.unit
def test_set_params_changes_the_spec_fingerprint() -> None:
    """A searched value reaches the fingerprint, so provenance stays honest."""
    classifier = ClassifierRegistry.create("LogisticRegression")
    before = classifier.spec.fingerprint()
    classifier.set_params(C=0.01)
    assert classifier.spec.params["C"] == 0.01
    assert classifier.spec.fingerprint() != before


@pytest.mark.unit
def test_set_params_changes_what_fit_actually_does() -> None:
    """
    ``set_params`` affects the next ``fit``, not just the reported spec.

    A near-zero ``C`` shrinks the logistic coefficients towards zero, so the
    fitted estimator is observably different from the default one.
    """
    table = make_feature_table(seed=101)
    default = ClassifierRegistry.create("LogisticRegression")
    shrunk = ClassifierRegistry.create("LogisticRegression")
    shrunk.set_params(C=1e-4)
    default.fit(table)
    shrunk.fit(table)
    default_norm = float(np.abs(default._estimator.coef_).sum())
    shrunk_norm = float(np.abs(shrunk._estimator.coef_).sum())
    assert shrunk_norm < default_norm


@pytest.mark.unit
def test_set_params_re_runs_constructor_validation() -> None:
    """Constructor guards apply to searched values exactly as to configured ones."""
    selector = FeatureSelectorRegistry.create("stepwise")
    with pytest.raises(HABITAPIError):
        selector.set_params(direction="sideways")


@pytest.mark.unit
def test_set_params_preserves_the_seed() -> None:
    """Reconfiguring a component does not silently unseed it."""
    selector = FeatureSelectorRegistry.create("lasso")
    selector.set_random_state(7)
    selector.set_params(cv=3)
    assert selector._seed == 7
    assert selector.spec.params["cv"] == 3


@pytest.mark.unit
def test_estimator_params_stays_out_of_the_spec_when_empty() -> None:
    """
    The passthrough key is visible to ``get_params`` but not to the spec.

    ``estimator_params`` is folded into ``spec.params`` only when non-empty,
    so components nobody customised keep their historical fingerprint. Making
    it a ``get_params`` key must not change that asymmetry.
    """
    default = ClassifierRegistry.create("LogisticRegression")
    assert default.get_params()[ESTIMATOR_PARAMS_KEY] == {}
    assert ESTIMATOR_PARAMS_KEY not in default.spec.params

    customised = ClassifierRegistry.create(
        "LogisticRegression", estimator_params={"fit_intercept": False}
    )
    assert customised.spec.params[ESTIMATOR_PARAMS_KEY] == {"fit_intercept": False}
    assert clone(customised).spec.to_dict() == customised.spec.to_dict()


@pytest.mark.unit
def test_get_params_returns_defensive_copies() -> None:
    """Mutating the returned mapping cannot change a component's fingerprint."""
    classifier = ClassifierRegistry.create(
        "LogisticRegression", estimator_params={"fit_intercept": False}
    )
    before = classifier.spec.fingerprint()
    classifier.get_params()[ESTIMATOR_PARAMS_KEY]["fit_intercept"] = True
    assert classifier.spec.fingerprint() == before
