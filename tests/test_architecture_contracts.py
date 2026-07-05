# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.
"""
Architecture contract self-checks (sklearn-style ``check_*``).

These tests enforce the two cross-domain conventions that keep HABIT easy to
learn:

1. Every class-based factory subclasses the shared
   :class:`~habit.core.common.registry.ClassRegistry` and therefore exposes the
   uniform ``register`` / ``create`` / ``get`` / ``available`` /
   ``register_params_model`` / ``get_params_model`` surface.
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

from habit.core.common.registry import ClassRegistry
from habit.core.common.orchestrator import (
    ORCHESTRATOR_CONTRACT,
    check_orchestrator_class,
)

# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

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
    "habitat_feature_plugin": (
        "habit.core.habitat_analysis.feature_registry",
        "HabitatFeatureRegistry",
    ),
}

REGISTRY_CONTRACT_METHODS = (
    "register",
    "create",
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
@pytest.mark.parametrize("registry_id", sorted(CLASS_REGISTRIES))
def test_registry_subclasses_class_registry(registry_id: str) -> None:
    """Each factory must subclass the shared ``ClassRegistry`` base."""
    import_path, attr = CLASS_REGISTRIES[registry_id]
    registry = _import_attr(import_path, attr)
    assert issubclass(registry, ClassRegistry)


@pytest.mark.unit
@pytest.mark.parametrize("registry_id", sorted(CLASS_REGISTRIES))
def test_registry_exposes_uniform_contract(registry_id: str) -> None:
    """Each factory must expose the full uniform registry contract."""
    import_path, attr = CLASS_REGISTRIES[registry_id]
    registry = _import_attr(import_path, attr)
    for method_name in REGISTRY_CONTRACT_METHODS:
        assert callable(getattr(registry, method_name, None)), (
            f"{attr!r} is missing uniform registry method '{method_name}()'."
        )


@pytest.mark.unit
@pytest.mark.parametrize("registry_id", sorted(CLASS_REGISTRIES))
def test_registry_available_returns_list(registry_id: str) -> None:
    """``available()`` must return a list of registered names."""
    import_path, attr = CLASS_REGISTRIES[registry_id]
    registry = _import_attr(import_path, attr)
    names = registry.available()
    assert isinstance(names, list)


@pytest.mark.unit
def test_registries_do_not_share_storage() -> None:
    """Distinct factories must own independent ``_registry`` mappings."""
    loaded = {}
    for registry_id, (import_path, attr) in CLASS_REGISTRIES.items():
        try:
            module = importlib.import_module(import_path)
        except ImportError:
            continue
        loaded[registry_id] = getattr(module, attr)
    # No two loaded registries may reference the same dict object.
    ids = [id(reg._registry) for reg in loaded.values()]
    assert len(ids) == len(set(ids)), "Two factories share the same _registry dict."


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
