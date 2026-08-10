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
"""Contract tests for the reproducible Windows Python 3.10 installer inputs."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
#: PyPI distribution name, used to recognize self-referencing extras.
DISTRIBUTION_NAME = "habitat-analysis"
CPU_LOCK_FILE = (
    PROJECT_ROOT / "installer" / "requirements-runtime-win-py310.lock"
)
GPU_LOCK_FILE = (
    PROJECT_ROOT / "installer" / "requirements-gpu-torch-win-py310.lock"
)
AUTOML_LOCK_FILE = PROJECT_ROOT / "installer" / "requirements-automl-win-py310.lock"
ANALYSIS_LOCK_FILE = (
    PROJECT_ROOT / "installer" / "requirements-analysis-win-py310.lock"
)
VENDOR_ASSETS_FILE = PROJECT_ROOT / "installer" / "vendor_assets.json"
ENVIRONMENT_FILE = PROJECT_ROOT / "installer" / "environment-cpu.yml"

OFFLINE_RUNTIME_PACKAGES = {"pyradiomics"}
PYRADIOMICS_NETWORK_DEPENDENCIES = {
    "pywavelets": "1.8.0",
    "pykwalify": "1.8.0",
}
#: Transitive pins the offline installer must resolve itself because the
#: package that needs them does not declare a wheel-compatible requirement.
#: ``six`` is imported by mrmr-selection's numba path at run time.
ALLOWED_UNDECLARED_LOCK_PACKAGES = {"six"}
TORCH_LAYER_PACKAGES = {"torch"}
FORBIDDEN_DEFAULT_PACKAGES = {
    "autogluon",
    "gradio",
    "lifelines",
    "opencv-python",
    "plotly",
    "pyvista",
    "shap",
    "torch",
    "torchvision",
    "trimesh",
    "versioneer",
}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_name(name: str) -> str:
    """
    Normalize a Python distribution name using the PEP 503 comparison rule.

    Args:
        name: Distribution name from a dependency declaration.

    Returns:
        str: Lowercase name with punctuation runs normalized to a hyphen.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_exact_requirement(requirement: str) -> Tuple[str, str]:
    """
    Parse one direct dependency that must use a single exact ``==`` pin.

    Args:
        requirement: Requirement line without comments or pip options.

    Returns:
        Tuple[str, str]: Canonical distribution name and exact version.
    """
    match = re.fullmatch(
        r"(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==(?P<version>.+)",
        requirement.strip(),
    )
    assert match is not None, f"Dependency is not exactly pinned: {requirement}"
    name = match.group("name")
    version = match.group("version").strip()
    assert name and version, f"Dependency has an empty name or version: {requirement}"
    assert not any(token in version for token in ("<", ">", "~=", "!=", ";"))
    return _canonical_name(name), version


def _parse_ranged_requirement(requirement: str) -> Tuple[str, str]:
    """
    Parse one library dependency, which must declare a bounded version range.

    HABIT is a library first: an exact pin in the package metadata would make
    it uninstallable next to another scientific-Python stack. A range without
    an upper bound is the opposite failure -- it lets a future major release
    silently break users. Both are rejected here.

    A PEP 508 environment marker (``; python_version < '3.14'``) is stripped
    before parsing: this contract polices the version range, and callers
    de-duplicate marker-conditional entries against the running interpreter.

    Args:
        requirement: Requirement line from the project metadata.

    Returns:
        Tuple[str, str]: Canonical distribution name and its version specifier.
    """
    match = re.fullmatch(
        r"(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?(?P<specifier>[<>=!~,.0-9A-Za-z*]*)",
        requirement.split(";", 1)[0].strip(),
    )
    assert match is not None, f"Unparsable dependency: {requirement}"
    name = match.group("name")
    specifier = match.group("specifier").strip()
    assert "==" not in specifier, (
        f"{name} is exactly pinned in package metadata: {requirement}. "
        "Exact pins belong to the installer lock files."
    )
    assert ">=" in specifier, f"{name} declares no minimum version: {requirement}"
    return _canonical_name(name), specifier


def _read_requirement_lines(path: Path) -> List[str]:
    """
    Read package lines while excluding comments and pip index directives.

    Args:
        path: Requirements or lock file to inspect.

    Returns:
        List[str]: Non-empty package requirement lines in declaration order.
    """
    requirements: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        requirements.append(line)
    return requirements


def _requirements_map(path: Path) -> Dict[str, str]:
    """
    Load an exact requirements file and reject duplicate distributions.

    Args:
        path: Requirements or lock file to inspect.

    Returns:
        Dict[str, str]: Canonical package names mapped to exact versions.
    """
    result: Dict[str, str] = {}
    for requirement in _read_requirement_lines(path):
        name, version = _parse_exact_requirement(requirement)
        assert name not in result, f"Duplicate dependency {name!r} in {path.name}"
        result[name] = version
    return result


def _dependency_array(section: str, key: str) -> List[str]:
    """
    Read one PEP 621 dependency array without requiring a TOML test dependency.

    The dependency arrays intentionally contain only quoted strings, so their
    TOML syntax is also a safe Python list literal. This keeps the contract
    test runnable in a newly bootstrapped Python 3.10 environment before test
    tooling installs a TOML parser (``tomllib`` is 3.11+).

    Args:
        section: Table header to read, for example ``project``.
        key: Array key inside that table, for example ``dependencies``.

    Returns:
        List[str]: Requirement strings in declaration order.
    """
    text = PYPROJECT_FILE.read_text(encoding="utf-8")
    section_match = re.search(
        rf"(?ms)^\[{re.escape(section)}\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    assert section_match is not None, f"pyproject.toml must contain [{section}]."
    array_match = re.search(
        rf"(?ms)^{re.escape(key)}\s*=\s*(\[.*?\])\s*$",
        section_match.group(1),
    )
    assert array_match is not None, f"[{section}] must declare {key}."
    requirements = ast.literal_eval(array_match.group(1))
    assert isinstance(requirements, list)
    assert all(isinstance(item, str) for item in requirements)
    return requirements


def _marker_applies(dependency: str) -> bool:
    """
    Report whether a requirement's environment marker holds here.

    A distribution may be declared twice with disjoint markers (pyarrow is
    split at Python 3.14). Only the entry matching THIS interpreter is
    relevant: the locks under test encode the Windows py310 bundle, which this
    interpreter runs.

    Args:
        dependency: Requirement string, with or without a ``;`` marker.

    Returns:
        bool: ``True`` when the requirement applies to this interpreter.
    """
    _, _, marker = dependency.partition(";")
    if not marker.strip():
        return True
    from packaging.markers import Marker

    return bool(Marker(marker.strip()).evaluate())


def _project_dependency_ranges() -> Dict[str, str]:
    """
    Return the default (non-optional) dependency ranges declared by the package.

    Returns:
        Dict[str, str]: Canonical dependency names mapped to version specifiers.
    """
    result: Dict[str, str] = {}
    for dependency in _dependency_array("project", "dependencies"):
        if not _marker_applies(dependency):
            continue
        name, specifier = _parse_ranged_requirement(dependency)
        assert name not in result, f"Duplicate project dependency: {name}"
        result[name] = specifier
    return result


def _self_referenced_extras(requirement: str) -> Optional[Tuple[str, ...]]:
    """
    Detect a self-referencing extra such as ``habitat-analysis[tables,viz]``.

    An extra may depend on other extras of the same distribution. setuptools
    records it verbatim in the wheel metadata and pip resolves it (verified by
    ``test_meta_extras_are_self_references_that_resolve``), which is how the
    ``ml`` / ``analysis`` / ``all`` / ``full`` groups avoid restating package
    lists that would then drift.

    Args:
        requirement: One entry of an optional-dependency array.

    Returns:
        Optional[Tuple[str, ...]]: The referenced extra names, or ``None``
        when the entry is an ordinary third-party requirement.
    """
    match = re.fullmatch(
        r"(?P<name>[A-Za-z0-9_.-]+)\[(?P<extras>[A-Za-z0-9_,.-]+)\]",
        requirement.strip(),
    )
    if match is None:
        return None
    if _canonical_name(match.group("name")) != _canonical_name(DISTRIBUTION_NAME):
        return None
    return tuple(part.strip() for part in match.group("extras").split(","))


def _optional_dependency_ranges() -> Dict[str, Dict[str, str]]:
    """
    Return the declared ranges of every optional-dependency group.

    Self-referencing extras are expanded transitively, so every group maps to
    the third-party packages a user actually receives. That keeps the contracts
    below (lock coverage, feature scoping) meaningful regardless of whether a
    group lists its packages directly or aggregates other groups.

    Returns:
        Dict[str, Dict[str, str]]: Extra name -> (dependency name -> specifier).
    """
    text = PYPROJECT_FILE.read_text(encoding="utf-8")
    section_match = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    assert section_match is not None, "pyproject.toml must declare extras."
    direct: Dict[str, Dict[str, str]] = {}
    references: Dict[str, Tuple[str, ...]] = {}
    for extra, array in re.findall(
        r"(?ms)^([A-Za-z0-9_-]+)\s*=\s*(\[.*?\])\s*$", section_match.group(1)
    ):
        requirements = ast.literal_eval(array)
        referenced: List[str] = []
        packages: Dict[str, str] = {}
        for requirement in requirements:
            self_extras = _self_referenced_extras(requirement)
            if self_extras is not None:
                referenced.extend(self_extras)
                continue
            if not _marker_applies(requirement):
                continue
            name, specifier = _parse_ranged_requirement(requirement)
            packages[name] = specifier
        direct[extra] = packages
        references[extra] = tuple(referenced)

    for extra in references:
        for referenced_extra in references[extra]:
            assert referenced_extra in direct, (
                f"extra {extra!r} references undeclared extra {referenced_extra!r}"
            )

    def _resolve(extra: str, seen: Tuple[str, ...] = ()) -> Dict[str, str]:
        assert extra not in seen, f"circular self-extra reference: {seen + (extra,)}"
        resolved = dict(direct[extra])
        for referenced_extra in references[extra]:
            resolved.update(_resolve(referenced_extra, seen + (extra,)))
        return resolved

    groups = {extra: _resolve(extra) for extra in direct}
    assert groups, "no optional-dependency groups parsed"
    return groups


def _declared_ranges() -> Dict[str, str]:
    """
    Return every dependency range HABIT declares, default or optional.

    Returns:
        Dict[str, str]: Canonical dependency names mapped to version specifiers.
    """
    declared = dict(_project_dependency_ranges())
    for group in _optional_dependency_ranges().values():
        declared.update(group)
    return declared


def _sha256(path: Path) -> str:
    """
    Calculate a vendor file hash without loading the executable into memory.

    Args:
        path: Existing vendor artifact to hash.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _setup_keyword(keyword_name: str) -> ast.AST:
    """
    Return one keyword value from the top-level setuptools ``setup`` call.

    Args:
        keyword_name: Keyword argument to locate.

    Returns:
        ast.AST: Parsed expression assigned to the requested keyword.
    """
    setup_tree = ast.parse(
        (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8"),
        filename="setup.py",
    )
    setup_calls = [
        node
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    assert len(setup_calls) == 1
    for keyword in setup_calls[0].keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    raise AssertionError(f"setup.py does not pass {keyword_name!r} to setup().")


def test_python_310_contract_is_explicit_and_patch_pinned() -> None:
    """
    PyPI metadata allows 3.10–3.14; the Windows installer still pins 3.10.20.

    The portable ZIP is a reproducible product environment (exact patch). The
    library metadata is wider so third-party projects can ``pip install HABIT``
    into existing 3.11–3.14 stacks (radiomics remains an optional extra).
    """
    pyproject_text = PYPROJECT_FILE.read_text(encoding="utf-8")
    environment_text = ENVIRONMENT_FILE.read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10,<3.15"' in pyproject_text
    assert 'python = ">=3.10,<3.15"' in pyproject_text
    assert re.search(r"(?m)^\s*-\s*python=3\.10\.20\s*$", environment_text)


def test_heavy_features_are_declared_only_as_targeted_optional_extras() -> None:
    """Package metadata must keep heavy feature families out of the default set."""
    extras = _optional_dependency_ranges()
    pyproject_text = PYPROJECT_FILE.read_text(encoding="utf-8")

    # AutoGluon must stay the narrow tabular distribution: the umbrella package
    # pulls in the text/vision stacks, which HABIT never calls.
    assert "autogluon-tabular" in extras["automl"]
    assert "autogluon" not in extras["automl"]
    assert '"autogluon.tabular[lightgbm,catboost]' in pyproject_text
    assert "torch" in extras["torch"]
    assert "torchvision" not in pyproject_text


def test_setup_reads_runtime_metadata_and_keeps_c_extension() -> None:
    """Legacy setuptools builds must consume pyproject and keep the optional C ext."""
    install_requires = _setup_keyword("install_requires")
    python_requires = _setup_keyword("python_requires")
    extension_modules = _setup_keyword("ext_modules")
    cmdclass = _setup_keyword("cmdclass")

    assert isinstance(install_requires, ast.Call)
    assert isinstance(install_requires.func, ast.Name)
    assert install_requires.func.id == "_read_runtime_dependencies"
    assert isinstance(python_requires, ast.Call)
    assert isinstance(python_requires.func, ast.Name)
    assert python_requires.func.id == "_read_python_requirement"
    # ext_modules=list(_extension_modules()) — optional native accel with fallback.
    assert isinstance(extension_modules, ast.Call)
    assert isinstance(extension_modules.func, ast.Name)
    assert extension_modules.func.id == "list"
    assert isinstance(cmdclass, ast.Dict)
    cmdclass_src = ast.unparse(cmdclass) if hasattr(ast, "unparse") else ""
    assert "_OptionalBuildExt" in cmdclass_src or any(
        isinstance(v, ast.Name) and v.id == "_OptionalBuildExt"
        for v in cmdclass.values
    )


def test_project_dependencies_are_range_bounded_and_feature_scoped() -> None:
    """
    Package metadata declares ranges, not pins, and stays feature-scoped.

    ``_parse_ranged_requirement`` already rejects exact pins and missing lower
    bounds for every entry, so this test states the remaining two rules: each
    range is closed at the top, and no heavy optional stack is required by
    default.
    """
    project_dependencies = _project_dependency_ranges()

    unbounded = [
        name
        for name, specifier in project_dependencies.items()
        if "<" not in specifier
    ]
    assert not unbounded, f"Dependencies with no upper bound: {unbounded}"
    assert not FORBIDDEN_DEFAULT_PACKAGES.intersection(project_dependencies)
    assert "autogluon" not in project_dependencies
    # PyRadiomics is intentionally optional: PyPI sdists often fail to build
    # (missing numpy build dep / Python 3.12+ versioneer breakage).
    assert "pyradiomics" not in project_dependencies
    extras = _optional_dependency_ranges()
    # PyRadiomics is installed separately (Windows Release wheel / PyPI /
    # conda-forge). The ``radiomics`` extra is an empty documented alias;
    # ``all`` must not pull pyradiomics either (broken Windows sdist).
    assert "radiomics" in extras
    assert extras["radiomics"] == {}
    assert "pyradiomics" not in extras["radiomics"]
    assert "pyradiomics" not in extras["all"]
    # Comments may mention the separate-install recipe; extras must not list it.
    for group_name, group in extras.items():
        assert "pyradiomics" not in group, (
            f"extra {group_name!r} must not declare pyradiomics"
        )


def test_meta_extras_are_self_references_that_resolve() -> None:
    """
    ``all`` and ``full`` must aggregate other extras instead of restating them.

    A hand-copied package list in a meta-extra is guaranteed to drift the next
    time a group changes -- exactly what the old ``all`` array risked. Writing
    them as self-references makes drift impossible by construction, and pip
    resolves them (setuptools records the requirement verbatim, and
    ``Requires-Dist: habitat-analysis[all]; extra == "full"`` is a normal
    dependency edge for the resolver).

    ``full`` is also the documented migration target for pre-1.1.0 users, so it
    must transitively contain every package that used to be a REQUIRED
    dependency and is now optional.
    """
    text = PYPROJECT_FILE.read_text(encoding="utf-8")
    section_match = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    assert section_match is not None
    arrays = dict(
        re.findall(r"(?ms)^([A-Za-z0-9_-]+)\s*=\s*(\[.*?\])\s*$", section_match.group(1))
    )

    for meta_extra in ("all", "full"):
        entries = ast.literal_eval(arrays[meta_extra])
        assert entries, f"extra {meta_extra!r} must not be empty"
        for entry in entries:
            assert _self_referenced_extras(entry) is not None, (
                f"extra {meta_extra!r} restates the requirement {entry!r} "
                "instead of referencing the extra that owns it."
            )

    extras = _optional_dependency_ranges()
    # Everything demoted from required to optional in 1.1.0. `chardet` is
    # absent on purpose: it was removed outright, not moved to an extra.
    demoted = {
        "matplotlib",
        "seaborn",
        "scikit-image",
        "pydicom",
        "pyarrow",
        "openpyxl",
    }
    missing = sorted(demoted - set(extras["full"]))
    assert not missing, (
        f"extra 'full' does not restore {missing}, so it is not a valid "
        "migration path for a pre-1.1.0 bare install."
    )
    assert set(extras["all"]) <= set(extras["full"])


def test_cpu_network_lock_covers_every_network_direct_dependency() -> None:
    """
    The default lock installs every declared dependency it must resolve online.

    The Windows bundle ships one working environment, so the lock legitimately
    covers optional workflows too. What it must never do is drift: every
    package it pins is either declared by HABIT (default or extra), a
    requirement of the offline PyRadiomics wheel, or a documented transitive
    exception.
    """
    locked = _requirements_map(CPU_LOCK_FILE)
    declared = _declared_ranges()

    missing = {
        name
        for name in _project_dependency_ranges()
        if name not in OFFLINE_RUNTIME_PACKAGES and name not in locked
    }
    assert not missing, f"Default dependencies absent from the lock: {sorted(missing)}"

    for name, version in PYRADIOMICS_NETWORK_DEPENDENCIES.items():
        assert locked.get(name) == version, (
            f"PyRadiomics is installed with --no-deps, so {name} must be pinned "
            "in the default lock."
        )

    undeclared = (
        set(locked)
        - set(declared)
        - set(PYRADIOMICS_NETWORK_DEPENDENCIES)
        - ALLOWED_UNDECLARED_LOCK_PACKAGES
    )
    assert not undeclared, (
        f"Lock pins packages HABIT never declares: {sorted(undeclared)}. "
        "Add them to pyproject.toml or to ALLOWED_UNDECLARED_LOCK_PACKAGES "
        "with a reason."
    )
    assert "pyradiomics" not in locked
    assert "habit" not in locked


def test_gpu_lock_defines_only_the_compatible_torch_replacement_layer() -> None:
    """The GPU lock must contain only HABIT's CUDA compute dependency."""
    expected_gpu_layer = {
        "torch": "2.4.0+cu121",
    }

    assert _requirements_map(GPU_LOCK_FILE) == expected_gpu_layer
    assert set(expected_gpu_layer) == TORCH_LAYER_PACKAGES


def test_optional_profiles_install_only_feature_scoped_packages() -> None:
    """Optional locks stay inside their own extra and skip umbrella packages."""
    extras = _optional_dependency_ranges()
    automl_locked = _requirements_map(AUTOML_LOCK_FILE)
    analysis_locked = _requirements_map(ANALYSIS_LOCK_FILE)

    assert set(automl_locked) == set(extras["automl"])
    assert set(analysis_locked) <= set(extras["analysis"]), (
        "The analysis lock pins packages the analysis extra does not declare: "
        f"{sorted(set(analysis_locked) - set(extras['analysis']))}"
    )
    all_optional = {**automl_locked, **analysis_locked}
    assert "autogluon" not in all_optional
    assert "torchvision" not in all_optional


def test_locked_versions_satisfy_the_declared_ranges() -> None:
    """
    Every locked version is a valid solution of the declared range.

    This is the link between the two halves of the dependency policy: the
    metadata says what HABIT supports, the locks say what the Windows bundle
    ships. A lock that falls outside its declared range means the shipped
    environment is not one the metadata claims to support.

    ``packaging`` is always importable here: matplotlib, a default HABIT
    dependency, requires it.
    """
    from packaging.specifiers import SpecifierSet

    declared = _declared_ranges()
    violations: List[str] = []
    for lock_file in (CPU_LOCK_FILE, GPU_LOCK_FILE, AUTOML_LOCK_FILE, ANALYSIS_LOCK_FILE):
        for name, version in _requirements_map(lock_file).items():
            specifier = declared.get(name)
            if specifier is None:
                continue
            if not SpecifierSet(specifier).contains(version, prereleases=True):
                violations.append(
                    f"{lock_file.name}: {name}=={version} violates {specifier}"
                )
    assert not violations, "Locked versions outside their declared range:\n" + "\n".join(
        violations
    )


def test_all_requirement_contracts_are_unique_and_exactly_pinned() -> None:
    """Every package line in a distributable contract must be exact and unique."""
    files = (
        CPU_LOCK_FILE,
        GPU_LOCK_FILE,
        AUTOML_LOCK_FILE,
        ANALYSIS_LOCK_FILE,
    )
    for path in files:
        requirement_lines = _read_requirement_lines(path)
        parsed = [_parse_exact_requirement(line) for line in requirement_lines]
        names = [name for name, _version in parsed]
        assert len(names) == len(set(names)), f"Duplicate package in {path.name}"


def test_vendor_manifest_has_valid_static_and_dynamic_hash_policies() -> None:
    """Static assets need real hashes; the build-produced HABIT wheel must not."""
    manifest = json.loads(VENDOR_ASSETS_FILE.read_text(encoding="utf-8"))
    static_assets = manifest["static_assets"]
    dynamic_assets = manifest["dynamic_assets"]

    assert manifest["schema"] == "habit.vendor-assets/v1"
    assert manifest["hash_algorithm"] == "sha256"
    assert all(HASH_PATTERN.fullmatch(asset["sha256"]) for asset in static_assets)
    assert len({asset["id"] for asset in static_assets}) == len(static_assets)

    static_by_id = {asset["id"]: asset for asset in static_assets}
    assert static_by_id["micromamba"]["package_path"] == (
        "installer/vendor/micromamba/micromamba.exe"
    )
    assert static_by_id["micromamba"]["sha256"] == (
        "8a51f88ec02600488ea20c3acd93fbd4da6c0f03fc499aa53fd234c6749b94b0"
    )
    assert static_by_id["micromamba"]["source"]["kind"] == "repository-vendor"
    assert static_by_id["micromamba"]["source"]["version"] == "2.8.1-0"

    assert dynamic_assets == [
        {
            "id": "habit-wheel",
            "package_path_glob": "dist/habitat_analysis-*.whl",
            "hash_policy": "compute-during-build",
        }
    ]
    assert "sha256" not in dynamic_assets[0]


def test_existing_vendor_asset_hashes_match_manifest() -> None:
    """Repository-supplied vendor asset hashes must match their tracked bytes."""
    manifest = json.loads(VENDOR_ASSETS_FILE.read_text(encoding="utf-8"))
    static_by_id = {
        asset["id"]: asset for asset in manifest["static_assets"]
    }
    repository_asset_ids: Set[str] = set(static_by_id)

    for asset_id in repository_asset_ids:
        asset = static_by_id[asset_id]
        source_path = PROJECT_ROOT / asset["package_path"]
        assert source_path.is_file(), f"Missing vendor source: {source_path}"
        assert _sha256(source_path) == asset["sha256"]

    # Micromamba is statically linked, so its complete package license payload
    # must remain beside the executable and be staged into every release.
    micromamba_license_dir = (
        PROJECT_ROOT / "installer" / "vendor" / "micromamba" / "licenses"
    )
    assert micromamba_license_dir.is_dir()
    assert any(path.is_file() for path in micromamba_license_dir.rglob("*"))
