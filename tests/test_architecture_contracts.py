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


# ---------------------------------------------------------------------------
# Layered dependency assertions (v1.0 architecture, L0-L5)
# ---------------------------------------------------------------------------
#
# The v1.0 layered packages under ``habit/`` obey a strict downward
# dependency rule (developer/api_upgrade/06 §3): a layer may only import the
# layers below it, and L0-L3 code must never reference configuration
# concepts (``yaml`` / ``out_dir`` / ``data_dir`` / ``run_mode`` /
# ``config_file``). These assertions scan MODULE-LEVEL imports via AST;
# function-level imports inside method bodies are the sanctioned escape
# hatch for lazy cross-layer lookups (mirroring the existing lazy-export
# pattern) and are therefore excluded from the scan.

import ast
from pathlib import Path

_HABIT_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "habit"

#: Foundations any layer may import: version, shared exceptions/registry
#: base, the public exception facade, and shared utils.
_FOUNDATION_PREFIXES = (
    "habit._version",
    "habit.core.common",
    "habit.api.exceptions",
    "habit.api.image",  # reused by habit.contracts per architecture mapping
    "habit.utils",
)

#: layer package -> habit-module prefixes it must NOT import at module level.
#: Packages that do not exist yet in a given phase are simply not scanned.
_LAYER_FORBIDDEN_IMPORTS = {
    "habit.kernels": ("habit.",),  # L0 imports no habit module at all
    "habit.contracts": (
        "habit.adapters",
        "habit.domain",
        "habit.execution",
        "habit.registry",
        "habit.spec",
        "habit.recipes",
        "habit.cli",
        "habit.commands",
        "habit.compat",
        "habit.core.habitat_analysis",
        "habit.core.machine_learning",
        "habit.core.preprocessing",
    ),
    "habit.adapters": (
        "habit.domain",
        "habit.execution",
        "habit.registry",
        "habit.spec",
        "habit.recipes",
        "habit.cli",
        "habit.commands",
        "habit.compat",
        "habit.core.habitat_analysis",
        "habit.core.machine_learning",
        "habit.core.preprocessing",
    ),
    "habit.execution": (
        "habit.adapters",
        "habit.domain",
        "habit.registry",
        "habit.spec",
        "habit.recipes",
        "habit.cli",
        "habit.commands",
        "habit.compat",
        "habit.core.habitat_analysis",
        "habit.core.machine_learning",
        "habit.core.preprocessing",
        "habit.api.habitat",
        "habit.api.clinical",
    ),
    "habit.domain": (
        "habit.recipes",
        "habit.spec",
        "habit.cli",
        "habit.commands",
        "habit.compat",
    ),
    "habit.registry": (
        "habit.adapters",
        "habit.domain",
        "habit.execution",
        "habit.spec",
        "habit.recipes",
        "habit.cli",
        "habit.commands",
        "habit.compat",
    ),
    "habit.spec": (
        "habit.recipes",
        "habit.cli",
        "habit.commands",
        "habit.compat",
    ),
    "habit.recipes": (
        "habit.cli",
        "habit.commands",
    ),
    "habit.compat": (
        "habit.cli",
        "habit.commands",
        "habit.recipes",
    ),
}

#: L0-L3 packages that must stay free of configuration concepts.
_CONFIG_FREE_PACKAGES = (
    "habit.kernels",
    "habit.contracts",
    "habit.adapters",
    "habit.execution",
    "habit.domain",
    "habit.registry",
)

#: Identifiers that signal configuration-layer concepts leaking into L0-L3.
_CONFIG_CONCEPT_IDENTIFIERS = ("yaml", "out_dir", "data_dir", "run_mode", "config_file")

#: Documented exemptions: (module file, identifier) pairs where the name is a
#: user-facing convenience rather than a configuration dependency. The
#: prototype itself defines ``StudyResult.save(out_dir)`` as the single
#: explicit write act of the contracts layer.
_CONFIG_CONCEPT_EXEMPTIONS = {
    ("manifest.py", "out_dir"),
}


def _iter_layer_python_files(package: str) -> list[Path]:
    """Yield every .py file of an existing layer package."""
    package_dir = _HABIT_PACKAGE_ROOT / package.removeprefix("habit.").replace(".", "/")
    if not package_dir.is_dir():
        return []
    return sorted(package_dir.rglob("*.py"))


def _module_level_imports(path: Path) -> list[str]:
    """
    Return module-level imported module names, resolving relative imports.

    Imports nested under ``if TYPE_CHECKING:`` blocks or inside function and
    class bodies are excluded: the former never execute, the latter are the
    sanctioned lazy-lookup escape hatch.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = path.parent.relative_to(_HABIT_PACKAGE_ROOT.parent).parts

    def _resolve(node: ast.ImportFrom) -> str:
        if node.level == 0:
            return node.module or ""
        base = list(package_parts[: len(package_parts) - node.level + 1])
        if node.module:
            base.extend(node.module.split("."))
        return ".".join(base)

    imports: list[str] = []
    for statement in tree.body:
        nodes: list[ast.AST] = []
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            nodes.append(statement)
        elif isinstance(statement, ast.If):
            # Skip ``if TYPE_CHECKING:`` blocks; collect other top-level
            # conditional imports (e.g. guarded optional dependencies).
            test = statement.test
            is_type_checking = (
                isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            ) or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if not is_type_checking:
                nodes.extend(
                    node
                    for node in ast.walk(statement)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                )
        elif isinstance(statement, ast.Try):
            nodes.extend(
                node
                for node in ast.walk(statement)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
        for node in nodes:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            else:
                imports.append(_resolve(node))
    return imports


def _config_concept_violations(path: Path) -> list[str]:
    """Find configuration-concept identifiers used anywhere in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        identifier = None
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, ast.Attribute):
            identifier = node.attr
        elif isinstance(node, ast.keyword):
            identifier = node.arg
        elif isinstance(node, ast.arg):
            identifier = node.arg
        if identifier in _CONFIG_CONCEPT_IDENTIFIERS and (
            (path.name, identifier) not in _CONFIG_CONCEPT_EXEMPTIONS
        ):
            violations.append(f"{path.name}:{getattr(node, 'lineno', '?')}:{identifier}")
    return violations


@pytest.mark.unit
@pytest.mark.parametrize("package", sorted(_LAYER_FORBIDDEN_IMPORTS))
def test_layer_does_not_import_upwards(package: str) -> None:
    """Module-level imports of each layer must respect the downward rule."""
    files = _iter_layer_python_files(package)
    if not files:
        pytest.skip(f"{package} does not exist yet in this phase.")
    forbidden = _LAYER_FORBIDDEN_IMPORTS[package]
    offenders: list[str] = []
    for path in files:
        for imported in _module_level_imports(path):
            if not imported.startswith("habit"):
                continue
            if imported.startswith(_FOUNDATION_PREFIXES):
                continue
            if any(imported == prefix or imported.startswith(f"{prefix}.") for prefix in forbidden if prefix != "habit.") or (
                forbidden == ("habit.",)
            ):
                offenders.append(f"{path.relative_to(_HABIT_PACKAGE_ROOT)} imports {imported}")
    assert not offenders, f"Layer violations in {package}: {offenders}"


@pytest.mark.unit
@pytest.mark.parametrize("package", _CONFIG_FREE_PACKAGES)
def test_layer_is_free_of_configuration_concepts(package: str) -> None:
    """L0-L3 modules must not reference yaml/out_dir/run_mode/data_dir."""
    files = _iter_layer_python_files(package)
    if not files:
        pytest.skip(f"{package} does not exist yet in this phase.")
    violations: list[str] = []
    for path in files:
        violations.extend(_config_concept_violations(path))
    assert not violations, f"Configuration concepts leaked into {package}: {violations}"


@pytest.mark.unit
def test_contracts_import_stays_lightweight() -> None:
    """``import habit.contracts`` must not pull in SimpleITK.

    Imaging IO must stay lazy so that the contracts layer is usable in
    notebook and service contexts without loading native libraries. (The
    ``yaml`` module is already loaded by the plain ``import habit`` baseline
    through the v0.1 ``habit.core.common`` config chain, so it is asserted
    at the AST level instead of the module level.)
    """
    import subprocess
    import sys

    script = (
        "import sys, habit.contracts\n"
        "blocked = {'SimpleITK', 'sitk', 'antspyx', 'radiomics'} & set(sys.modules)\n"
        "print('blocked_modules', sorted(blocked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout or "") + (completed.stderr or "")
    assert "blocked_modules []" in completed.stdout
