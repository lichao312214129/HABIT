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
Architecture contract self-checks (sklearn-style ``check_*``).

These tests enforce the two cross-domain conventions that keep HABIT easy to
learn:

1. Every registry subclasses the shared
   :class:`~habit.core.common.registry._BaseRegistry` and therefore exposes the
   uniform ``register`` / ``get`` / ``available`` / ``register_params_model`` /
   ``get_params_model`` surface. Class-based factories additionally subclass
   :class:`~habit.core.common.registry.ClassRegistry` (adding ``create``), while
   callable registries subclass
   :class:`~habit.core.common.registry.CallableRegistry` (adding ``get_entry`` /
   ``entries``).
2. Every top-level orchestrator exposes its declared terminal method(s)
   (``run`` or ``fit`` + ``predict``) as listed in
   :data:`~habit.core.common.orchestrator.ORCHESTRATOR_CONTRACT`.

Registries / orchestrators that depend on optional third-party packages
(``ants``, ``radiomics``, ...) are skipped when those packages are absent, so
this file runs cleanly in any environment.
"""

from __future__ import annotations

import importlib
from typing import Tuple

import pytest

from habit.core.common.registry import (
    CallableRegistry,
    ClassRegistry,
    _BaseRegistry,
)
from habit.core.common.orchestrator import (
    ORCHESTRATOR_CONTRACT,
    check_orchestrator_class,
)

# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

#: Class-based factories (payload is a class; expose ``create``).
#: {registry_id: (import_path, attribute_name)}
CLASS_REGISTRIES = {
    "preprocessor": (
        "habit.core.preprocessing.preprocessor_factory",
        "PreprocessorFactory",
    ),
    "model": ("habit.core.machine_learning.models.factory", "ModelFactory"),
    "clustering": (
        "habit.core.habitat_analysis.clustering.base_clustering",
        "ClusteringAlgorithmFactory",
    ),
    "feature_extractor": (
        "habit.core.habitat_analysis.clustering_features.base_extractor",
        "FeatureExtractorRegistry",
    ),
    "feature_preprocessing": (
        "habit.core.habitat_analysis.feature_preprocessing.base_preprocessing",
        "PreprocessingMethodFactory",
    ),
    "habitat_feature": (
        "habit.core.habitat_analysis.feature_registry",
        "HabitatFeatureFactory",
    ),
}

#: Callable registries (payload is a function; expose ``get_entry`` / ``entries``).
#: {registry_id: (import_path, attribute_name)}
CALLABLE_REGISTRIES = {
    "feature_selector": (
        "habit.core.machine_learning.feature_selectors.selector_registry",
        "SelectorRegistry",
    ),
    "metric": (
        "habit.core.machine_learning.evaluation.metrics",
        "MetricRegistry",
    ),
}

#: Every registry, regardless of payload kind.
ALL_REGISTRIES = {**CLASS_REGISTRIES, **CALLABLE_REGISTRIES}

#: Contract shared by every registry (class-based and callable).
BASE_CONTRACT_METHODS = (
    "register",
    "get",
    "available",
    "register_params_model",
    "get_params_model",
)


def _import_attr(import_path: str, attr: str):
    """Import ``attr`` from ``import_path``, skipping on missing optional deps."""
    try:
        module = importlib.import_module(import_path)
    except ImportError as exc:  # optional third-party dependency absent
        pytest.skip(f"Optional dependency missing for {import_path}: {exc}")
    return getattr(module, attr)


@pytest.mark.unit
@pytest.mark.parametrize("registry_id", sorted(ALL_REGISTRIES))
def test_registry_subclasses_base_registry(registry_id: str) -> None:
    """Every registry must subclass the shared ``_BaseRegistry`` core."""
    import_path, attr = ALL_REGISTRIES[registry_id]
    registry = _import_attr(import_path, attr)
    assert issubclass(registry, _BaseRegistry)


@pytest.mark.unit
@pytest.mark.parametrize("registry_id", sorted(CLASS_REGISTRIES))
def test_class_registry_subclasses_class_registry(registry_id: str) -> None:
    """Each class-based factory must subclass ``ClassRegistry`` and add ``create``."""
    import_path, attr = CLASS_REGISTRIES[registry_id]
    registry = _import_attr(import_path, attr)
    assert issubclass(registry, ClassRegistry)
    assert callable(getattr(registry, "create", None))


@pytest.mark.unit
@pytest.mark.parametrize("registry_id", sorted(CALLABLE_REGISTRIES))
def test_callable_registry_subclasses_callable_registry(registry_id: str) -> None:
    """Each callable registry must subclass ``CallableRegistry`` and add ``entries``."""
    import_path, attr = CALLABLE_REGISTRIES[registry_id]
    registry = _import_attr(import_path, attr)
    assert issubclass(registry, CallableRegistry)
    assert callable(getattr(registry, "get_entry", None))
    assert callable(getattr(registry, "entries", None))


@pytest.mark.unit
@pytest.mark.parametrize("registry_id", sorted(ALL_REGISTRIES))
def test_registry_exposes_uniform_contract(registry_id: str) -> None:
    """Each registry must expose the full uniform registry contract."""
    import_path, attr = ALL_REGISTRIES[registry_id]
    registry = _import_attr(import_path, attr)
    for method_name in BASE_CONTRACT_METHODS:
        assert callable(getattr(registry, method_name, None)), (
            f"{attr!r} is missing uniform registry method '{method_name}()'."
        )


@pytest.mark.unit
@pytest.mark.parametrize("registry_id", sorted(ALL_REGISTRIES))
def test_registry_available_returns_list(registry_id: str) -> None:
    """``available()`` must return a list of registered names."""
    import_path, attr = ALL_REGISTRIES[registry_id]
    registry = _import_attr(import_path, attr)
    names = registry.available()
    assert isinstance(names, list)


@pytest.mark.unit
def test_registries_do_not_share_storage() -> None:
    """Distinct registries must own independent ``_registry`` mappings."""
    loaded = {}
    for registry_id, (import_path, attr) in ALL_REGISTRIES.items():
        try:
            module = importlib.import_module(import_path)
        except ImportError:
            continue
        loaded[registry_id] = getattr(module, attr)
    # No two loaded registries may reference the same dict object.
    ids = [id(reg._registry) for reg in loaded.values()]
    assert len(ids) == len(set(ids)), "Two registries share the same _registry dict."


@pytest.mark.unit
def test_habitat_feature_factory_creates_registered_handler() -> None:
    """Habitat feature handlers use the same named factory lookup as preprocessors."""
    from typing import Any, Dict

    from habit.core.habitat_analysis.feature_registry import (
        BaseHabitatFeature,
        BatchExportContext,
        HabitatFeatureFactory,
        SubjectExtractionContext,
    )

    class ContractFeature(BaseHabitatFeature):
        """Minimal handler used to verify the factory contract."""

        subject_data_key = "contract"
        output_csv_name = "contract.csv"
        progress_desc = "Contract Feature"

        def extract_subject(self, ctx: SubjectExtractionContext) -> Dict[str, Any]:
            """Return a minimal per-subject feature result."""
            return {"subject": ctx.subj}

        def export_batch(
            self,
            data: Dict[str, Dict[str, Any]],
            ctx: BatchExportContext,
        ) -> None:
            """Implement the required batch-export contract for this test."""
            return None

    HabitatFeatureFactory.register("contract_feature")(ContractFeature)
    handler = HabitatFeatureFactory.get_handler("contract_feature")

    assert isinstance(handler, ContractFeature)
    assert handler.feature_name() == "contract_feature"
    assert "contract_feature" in HabitatFeatureFactory.registered_feature_names()


# ---------------------------------------------------------------------------
# Orchestrator contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("domain_key", sorted(ORCHESTRATOR_CONTRACT))
def test_orchestrator_exposes_terminal_methods(domain_key: str) -> None:
    """Each orchestrator must expose its declared terminal method(s)."""
    import_path, class_name, terminal_methods = ORCHESTRATOR_CONTRACT[domain_key]
    orchestrator_cls = _import_attr(import_path, class_name)
    check_orchestrator_class(orchestrator_cls, terminal_methods)
