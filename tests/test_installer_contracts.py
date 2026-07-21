"""Contract tests for the reproducible Windows Python 3.10 installer inputs."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
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
TORCH_LAYER_PACKAGES = {"torch"}
FORBIDDEN_DEFAULT_PACKAGES = {
    "autogluon",
    "gradio",
    "lifelines",
    "networkx",
    "opencv-python",
    "plotly",
    "pyarrow",
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


def _project_dependency_map() -> Dict[str, str]:
    """
    Read the PEP 621 dependency array without requiring a TOML test dependency.

    The project dependency array intentionally contains only quoted strings, so
    its TOML syntax is also a safe Python list literal. This keeps the contract
    test runnable in a newly bootstrapped Python 3.10 environment before test
    tooling installs a TOML parser.

    Returns:
        Dict[str, str]: Canonical direct dependency names and exact versions.
    """
    text = PYPROJECT_FILE.read_text(encoding="utf-8")
    project_match = re.search(
        r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    assert project_match is not None, "pyproject.toml must contain [project]."
    dependencies_match = re.search(
        r"(?ms)^dependencies\s*=\s*(\[.*?\])\s*$",
        project_match.group(1),
    )
    assert dependencies_match is not None
    dependencies = ast.literal_eval(dependencies_match.group(1))
    assert isinstance(dependencies, list)

    result: Dict[str, str] = {}
    for dependency in dependencies:
        assert isinstance(dependency, str)
        name, version = _parse_exact_requirement(dependency)
        assert name not in result, f"Duplicate project dependency: {name}"
        result[name] = version
    return result


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
    """Package metadata must allow only 3.10 and provision exactly 3.10.20."""
    pyproject_text = PYPROJECT_FILE.read_text(encoding="utf-8")
    environment_text = ENVIRONMENT_FILE.read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10,<3.11"' in pyproject_text
    assert 'python = ">=3.10,<3.11"' in pyproject_text
    assert re.search(r"(?m)^\s*-\s*python=3\.10\.20\s*$", environment_text)


def test_heavy_features_are_declared_only_as_targeted_optional_extras() -> None:
    """Package metadata must keep heavy feature families out of the default set."""
    pyproject_text = PYPROJECT_FILE.read_text(encoding="utf-8")

    assert '"autogluon.tabular[lightgbm,catboost]==1.5.0"' in pyproject_text
    assert '"autogluon==1.5.0"' not in pyproject_text
    assert '"torch>=2.4,<3"' in pyproject_text
    assert "torchvision" not in pyproject_text


def test_setup_reads_runtime_metadata_and_keeps_c_extension() -> None:
    """Legacy setuptools builds must consume pyproject and compile the extension."""
    install_requires = _setup_keyword("install_requires")
    python_requires = _setup_keyword("python_requires")
    extension_modules = _setup_keyword("ext_modules")

    assert isinstance(install_requires, ast.Call)
    assert isinstance(install_requires.func, ast.Name)
    assert install_requires.func.id == "_read_runtime_dependencies"
    assert isinstance(python_requires, ast.Call)
    assert isinstance(python_requires.func, ast.Name)
    assert python_requires.func.id == "_read_python_requirement"
    assert isinstance(extension_modules, ast.List)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Extension"
        for node in extension_modules.elts
    )


def test_project_dependencies_are_exactly_pinned_and_feature_scoped() -> None:
    """Project metadata must be the sole default dependency declaration."""
    project_dependencies = _project_dependency_map()

    assert not FORBIDDEN_DEFAULT_PACKAGES.intersection(project_dependencies)
    assert "autogluon" not in project_dependencies
    assert "pyradiomics" in project_dependencies


def test_cpu_network_lock_covers_every_network_direct_dependency() -> None:
    """The default lock adds only dependencies of the offline PyRadiomics wheel."""
    project_dependencies = _project_dependency_map()
    expected_network_dependencies = {
        name: version
        for name, version in project_dependencies.items()
        if name not in OFFLINE_RUNTIME_PACKAGES
    }
    expected_network_dependencies.update(PYRADIOMICS_NETWORK_DEPENDENCIES)

    assert _requirements_map(CPU_LOCK_FILE) == expected_network_dependencies
    assert "pyradiomics" not in _requirements_map(CPU_LOCK_FILE)
    assert "habit" not in _requirements_map(CPU_LOCK_FILE)


def test_gpu_lock_defines_only_the_compatible_torch_replacement_layer() -> None:
    """The GPU lock must contain only HABIT's CUDA compute dependency."""
    expected_gpu_layer = {
        "torch": "2.4.0+cu121",
    }

    assert _requirements_map(GPU_LOCK_FILE) == expected_gpu_layer
    assert set(expected_gpu_layer) == TORCH_LAYER_PACKAGES


def test_optional_profiles_install_only_feature_scoped_packages() -> None:
    """Optional locks must not reintroduce umbrella AutoGluon or TorchVision."""
    assert _requirements_map(AUTOML_LOCK_FILE) == {
        "autogluon-tabular": "1.5.0",
    }
    assert _requirements_map(ANALYSIS_LOCK_FILE) == {
        "pyarrow": "20.0.0",
        "krippendorff": "0.8.2",
        "shap": "0.49.1",
        "plotly": "6.8.0",
        "lifelines": "0.30.0",
    }
    all_optional = {
        **_requirements_map(AUTOML_LOCK_FILE),
        **_requirements_map(ANALYSIS_LOCK_FILE),
    }
    assert "autogluon" not in all_optional
    assert "torchvision" not in all_optional


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
            "package_path_glob": "dist/HABIT-*.whl",
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
