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
"""Regression tests for HABIT's non-editable distribution contents."""

import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Set

import pytest
from setuptools import find_packages


PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The EXACT set of distributions a bare ``pip install habitat-analysis`` may
#: install. This is a whitelist, and the assertion below is an equality check
#: on purpose: the mechanism mirrors ``tests/api/test_public_api.py`` guarding
#: the public symbol surface. Casually adding a required dependency -- the way
#: matplotlib, pyarrow, scikit-image, pydicom, seaborn, openpyxl and chardet
#: accumulated before 1.1.0 -- must turn this test red and force an explicit
#: decision, because every entry here is bytes every user downloads forever.
#:
#: To ADD an entry you must be able to complete the sentence "the habitat
#: kernel cannot run at all without this". Otherwise it belongs in an extra in
#: ``[project.optional-dependencies]``, gated by
#: ``habit.utils.optional_deps.require``.
REQUIRED_DEPENDENCIES: Dict[str, str] = {
    # --- the habitat kernel: imaging + clustering mathematics ---------------
    "numpy": "Every array in every contract; the C extension builds against it.",
    "scipy": "Distances, statistics and morphology under the habitat kernels.",
    "pandas": "FeatureTable IS a DataFrame; every table crosses pandas.",
    "scikit-learn": (
        "KMeans / GMM habitat fitters, preprocessing and metrics -- the "
        "clustering itself, not an optional backend."
    ),
    "SimpleITK": (
        "Reads and writes every medical image format and owns physical-space "
        "geometry; no image enters HABIT without it."
    ),
    # --- plumbing every code path crosses ----------------------------------
    "pydantic": "Validates every Spec and params model at construction time.",
    "PyYAML": "Every CLI configuration file is YAML.",
    "click": "Implements the `habit` console script declared in project.scripts.",
    "tqdm": "Backs habit.utils.progress_utils, used by every long-running map.",
    "joblib": "Parallel cohort map and model persistence.",
    "kneed": (
        "Elbow selection of the habitat count in habit.kernels."
        "cluster_selection -- kernel logic rather than a backend, and 0.1 MB "
        "with no transitive dependencies."
    ),
}

#: Optional packages the bare-install smoke test hides from the interpreter.
#: ``radiomics`` is included even though it is never a pip dependency: the
#: point of the test is that the habitat kernel path needs none of them.
BLOCKED_OPTIONAL_MODULES: tuple[str, ...] = (
    "matplotlib",
    "seaborn",
    "pydicom",
    "pyarrow",
    "openpyxl",
    "skimage",
    "radiomics",
    "napari",
)


def _habit_packages() -> Set[str]:
    """
    Discover packages with the same restriction used by ``setup.py``.

    Returns:
        Set[str]: Importable package names that belong to HABIT.
    """
    return set(
        find_packages(
            where=str(PROJECT_ROOT),
            include=("habit", "habit.*"),
        )
    )


def test_machine_learning_statistics_is_distributable() -> None:
    """The evaluation dependency must be included in non-editable installs."""
    packages = _habit_packages()
    assert "habit.compat.engines.machine_learning.statistics" in packages


def test_distribution_excludes_repository_tests() -> None:
    """User installations must not expose the repository's test packages."""
    packages = _habit_packages()
    assert not any(name == "tests" or name.startswith("tests.") for name in packages)


def test_manifest_prunes_developer_trees() -> None:
    """Published sdists must not ship tests/docs/demo trees."""
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for prune in ("tests", "demo_data", "docs", "developer", ".github"):
        assert f"prune {prune}" in manifest, f"MANIFEST.in must prune {prune}"


def test_package_version_and_python_support_are_consistent() -> None:
    """Build metadata and Poetry metadata must describe the tested runtime."""
    version_scope: dict[str, object] = {}
    version_file = PROJECT_ROOT / "habit" / "_version.py"
    exec(
        compile(version_file.read_text(encoding="utf-8"), str(version_file), "exec"),
        version_scope,
    )
    package_version = str(version_scope["__version__"])
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    version_match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        pyproject_text,
        flags=re.MULTILINE,
    )
    assert version_match is not None
    assert version_match.group(1) == package_version
    assert 'python = ">=3.10,<3.15"' in pyproject_text
    assert 'requires-python = ">=3.10,<3.15"' in pyproject_text
    # PyPI long_description must stay English; Chinese README remains in the
    # repository / sdist for bilingual readers but is not the packaging default.
    assert 'readme = "README_en.md"' in pyproject_text
    assert (PROJECT_ROOT / "README_en.md").is_file()
    assert (PROJECT_ROOT / "habit" / "py.typed").is_file()


def test_manifest_includes_bilingual_readmes() -> None:
    """Sdist must ship both READMEs even though PyPI renders the English one."""
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include README.md" in manifest
    assert "include README_en.md" in manifest


def _declared_required_dependencies() -> Set[str]:
    """
    Parse the distribution names in ``[project.dependencies]``.

    The array holds only quoted strings, so its TOML syntax is a safe Python
    list literal; that keeps this contract runnable before a TOML parser is
    installed (``tomllib`` is 3.11+, HABIT supports 3.10).

    Returns:
        Set[str]: Distribution names, environment markers and version
        specifiers stripped, de-duplicated (a marker-split entry such as
        pyarrow's counts once).
    """
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
    assert section is not None, "pyproject.toml must contain [project]."
    array = re.search(
        r"(?ms)^dependencies\s*=\s*(\[.*?\])\s*$",
        section.group(1),
    )
    assert array is not None, "[project] must declare dependencies."
    requirements = ast.literal_eval(array.group(1))
    assert isinstance(requirements, list)

    names: Set[str] = set()
    for requirement in requirements:
        assert isinstance(requirement, str)
        base = requirement.split(";", 1)[0].strip()
        name = re.split(r"[<>=!~\[]", base, maxsplit=1)[0].strip()
        assert name, f"Unparsable dependency: {requirement}"
        names.add(name)
    return names


def test_required_dependencies_match_the_explicit_whitelist() -> None:
    """
    ``[project.dependencies]`` must equal ``REQUIRED_DEPENDENCIES`` exactly.

    Turning a required dependency into an optional extra is a breaking change
    for users, and turning an optional one into a required dependency is a
    silent tax on every user. Neither should happen without editing the
    whitelist above -- which is where the justification lives.
    """
    declared = _declared_required_dependencies()
    expected = set(REQUIRED_DEPENDENCIES)

    added = sorted(declared - expected)
    removed = sorted(expected - declared)
    assert not added, (
        f"New REQUIRED dependencies: {added}. Every user downloads these "
        "forever. Move them to [project.optional-dependencies] and gate them "
        "with habit.utils.optional_deps.require, or -- if the habitat kernel "
        "truly cannot run without them -- add them to REQUIRED_DEPENDENCIES "
        "with the reason."
    )
    assert not removed, (
        f"Dependencies dropped from the required set: {removed}. That breaks "
        "existing installs; update REQUIRED_DEPENDENCIES in the same commit "
        "and document the migration path in CHANGELOG.md."
    )


def test_every_optional_extra_is_known_to_the_require_helper() -> None:
    """
    ``OPTIONAL_EXTRA_MODULES`` must cover every extra declared in pyproject.

    ``require(..., extra=...)`` validates its argument against that mapping, so
    an extra missing from it could never appear in an install hint.
    """
    from habit.utils.optional_deps import OPTIONAL_EXTRA_MODULES

    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    assert section is not None, "pyproject.toml must declare extras."
    declared = {
        extra
        for extra, _array in re.findall(
            r"(?ms)^([A-Za-z0-9_-]+)\s*=\s*(\[.*?\])\s*$", section.group(1)
        )
    }
    # ``all`` and ``full`` are user-facing meta-extras; no single missing
    # module ever points at them, so the helper does not list them.
    meta_extras = {"all", "full"}
    only_in_pyproject = sorted(declared - meta_extras - set(OPTIONAL_EXTRA_MODULES))
    only_in_helper = sorted(set(OPTIONAL_EXTRA_MODULES) - declared)
    assert declared - meta_extras == set(OPTIONAL_EXTRA_MODULES), (
        "pyproject extras and habit.utils.optional_deps.OPTIONAL_EXTRA_MODULES "
        f"disagree: only in pyproject {only_in_pyproject}, "
        f"only in the helper {only_in_helper}."
    )


#: Script executed in a fresh interpreter with the optional packages hidden.
#:
#: Blocking is done with a ``sys.meta_path`` finder rather than by building a
#: clean virtualenv: it proves the same property (the kernel path imports none
#: of these) at a cost CI can pay on every commit. The real clean-install check
#: is the ``bare-install`` job in .github/workflows/tests.yml, which pip-installs
#: the wheel with no extras and then runs this very test.
_BARE_INSTALL_SMOKE_SCRIPT = '''
import sys
from importlib.abc import MetaPathFinder
from typing import Any, Optional, Sequence

BLOCKED = set({blocked!r})


class _MissingExtraFinder(MetaPathFinder):
    """Make the optional packages look uninstalled, however they are imported."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Any = None,
    ) -> None:
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(
                "hidden by the bare-install contract test", name=fullname
            )
        return None


sys.meta_path.insert(0, _MissingExtraFinder())

import habit
import habit.recipes as recipes
from habit.datasets import make_synthetic_cohort
from habit.exceptions import OptionalDependencyError
from habit.spec.specs import HabitatSpec, Spec

# 1) The full two-step training path must run start to finish.
#    `supervoxelizer` is named explicitly: the default supervoxel backend is
#    "kmeans" (feature clustering), and this test must not start silently
#    depending on that default staying non-SLIC.
cohort = make_synthetic_cohort(
    n_subjects=3, modalities=("T1", "T2"), shape=(10, 10, 10), rng=0
)
spec = HabitatSpec(
    name="bare_install_two_step",
    voxel_feature_extractor=Spec(name="raw", params={{"modalities": ["T1", "T2"]}}),
    supervoxelizer=Spec(name="kmeans", params={{"n_supervoxels": 5}}),
    habitat_model_fitter=Spec(
        name="kmeans",
        params={{
            "min_habitats": 2,
            "max_habitats": 3,
            "validation": "silhouette",
            "n_init": 3,
        }},
    ),
    habitat_assigner=Spec(name="nearest_centroid", params={{}}),
    random_seed=42,
)
result = recipes.two_step(cohort, spec)
assert result.habitat_model is not None, "two_step produced no habitat model"
assert result.habitat_model.n_habitats >= 2

# 2) Not one blocked package may have been imported along the way. A blocked
#    import raises, so this also catches a swallowed ImportError.
leaked = sorted(name for name in BLOCKED if name in sys.modules)
assert not leaked, "kernel path imported optional packages: " + repr(leaked)

# 3) Reaching for a blocked backend must fail with OptionalDependencyError and
#    a copy-pasteable pip command, never a bare ModuleNotFoundError.
from habit.utils.optional_deps import require

for module, extra in (
    ("matplotlib.pyplot", "viz"),
    ("seaborn", "viz"),
    ("pydicom", "dicom"),
    ("pyarrow", "tables"),
    ("openpyxl", "tables"),
    ("skimage.segmentation", "slic"),
):
    try:
        require(module, extra=extra, purpose="the bare-install contract test")
    except OptionalDependencyError as exc:
        message = str(exc)
        assert 'pip install "habitat-analysis[' + extra + ']"' in message, message
        assert module in message, message
    else:
        raise AssertionError(module + " was expected to be unavailable")

# 4) The parquet default must fail loudly and offer BOTH exits, never silently
#    fall back to CSV.
import pandas as pd
from habit.utils.habitats_results_io import save_habitats_results

try:
    save_habitats_results(pd.DataFrame({{"a": [1]}}), sys.argv[1], "parquet")
except OptionalDependencyError as exc:
    message = str(exc)
    assert 'pip install "habitat-analysis[tables]"' in message, message
    assert "habitats_results_format: csv" in message, message
else:
    raise AssertionError("parquet export must not succeed without pyarrow")

# CSV, which needs no extra, must still work.
csv_path = save_habitats_results(pd.DataFrame({{"a": [1]}}), sys.argv[1], "csv")
assert csv_path.name == "habitats.csv", csv_path

print("BARE_INSTALL_SMOKE_OK")
'''


@pytest.mark.unit
def test_bare_install_runs_the_habitat_kernel_path(tmp_path: Path) -> None:
    """
    The REQUIRED dependency set alone must be enough to train a habitat model.

    This is the load-bearing half of the diet: the whitelist above states what
    HABIT claims to need, and this proves the claim. It runs in a fresh
    interpreter because ``sys.modules`` is process-global state the pytest
    process has already polluted.

    Args:
        tmp_path: pytest-provided scratch directory, used as the habitats
            results output directory.
    """
    script = _BARE_INSTALL_SMOKE_SCRIPT.format(blocked=list(BLOCKED_OPTIONAL_MODULES))
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode == 0, output
    assert "BARE_INSTALL_SMOKE_OK" in output, output
