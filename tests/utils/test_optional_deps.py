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
"""Tests for the optional-dependency import gate and its install hints."""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from types import ModuleType
from typing import Any, Iterator, Optional, Sequence
from unittest import mock

import pytest

from habit.exceptions import OptionalDependencyError
from habit.utils.optional_deps import (
    DISTRIBUTION_NAME,
    INSTALLATION_DOCS_URL,
    OPTIONAL_EXTRA_MODULES,
    install_command,
    optional_dependency_hint,
    pyradiomics_install_hint,
    require,
    require_excel_backend,
    require_parquet_backend,
    require_pyradiomics,
    windows_pyradiomics_wheel_url,
)


class _BlockingFinder(MetaPathFinder):
    """Meta-path finder that makes one package root look uninstalled."""

    def __init__(self, root: str) -> None:
        """
        Args:
            root: Top-level package name to hide, including its submodules.
        """
        self._root = root

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Any = None,
    ) -> None:
        """
        Raise for the hidden package, defer to the next finder otherwise.

        Args:
            fullname: Fully qualified module name being imported.
            path: Parent package ``__path__``, unused.
            target: Existing module for reloads, unused.

        Returns:
            None: Always, so the remaining finders run for other modules.

        Raises:
            ModuleNotFoundError: When ``fullname`` is inside the hidden root.
        """
        if fullname.split(".")[0] == self._root:
            raise ModuleNotFoundError(f"hidden by test: {fullname}", name=fullname)
        return None


@pytest.fixture
def hide_module() -> Iterator[Any]:
    """
    Provide a callable that hides a top-level package for one test.

    Yields:
        A function ``hide(root: str) -> None`` that installs the blocking
        finder and drops any already-imported copy of ``root`` from
        ``sys.modules``; both are undone at teardown.
    """
    finders: list[MetaPathFinder] = []
    saved: dict[str, ModuleType] = {}

    def hide(root: str) -> None:
        for name in [n for n in sys.modules if n.split(".")[0] == root]:
            saved[name] = sys.modules.pop(name)
        finder = _BlockingFinder(root)
        finders.append(finder)
        sys.meta_path.insert(0, finder)

    try:
        yield hide
    finally:
        for finder in finders:
            sys.meta_path.remove(finder)
        sys.modules.update(saved)


def test_install_command_quotes_the_extra_for_every_shell() -> None:
    """The hint must be pasteable into zsh / PowerShell, which glob ``[``."""
    assert install_command("viz") == 'pip install "habitat-analysis[viz]"'
    assert install_command("tables") == f'pip install "{DISTRIBUTION_NAME}[tables]"'


def test_install_command_rejects_an_undeclared_extra() -> None:
    """A typo must fail loudly rather than print an unresolvable command."""
    with pytest.raises(ValueError, match="Unknown HABIT extra"):
        install_command("vizz")


def test_hint_names_the_module_the_purpose_and_the_command() -> None:
    """The message must answer what, why and how in one read."""
    hint = optional_dependency_hint(
        "matplotlib.pyplot", extra="viz", purpose="publication figures"
    )
    assert "matplotlib.pyplot" in hint
    assert "publication figures" in hint
    assert 'pip install "habitat-analysis[viz]"' in hint
    assert INSTALLATION_DOCS_URL in hint


def test_hint_lists_alternatives_when_a_dependency_free_route_exists() -> None:
    """Escape routes such as the CSV switch must appear beside the command."""
    hint = optional_dependency_hint(
        "pyarrow",
        extra="tables",
        purpose="parquet export",
        alternatives=("set habitats_results_format: csv",),
    )
    assert "Alternatively:" in hint
    assert "habitats_results_format: csv" in hint


def test_require_returns_the_requested_submodule() -> None:
    """``require`` must mirror ``import a.b as x`` and return the submodule."""
    module = require("json.decoder", extra="viz", purpose="a unit test")
    assert module.__name__ == "json.decoder"


def test_require_rejects_an_undeclared_extra_before_importing() -> None:
    """The extra is validated first, so the error is identical either way."""
    with pytest.raises(ValueError, match="Unknown HABIT extra"):
        require("json", extra="not-an-extra", purpose="a unit test")


@pytest.mark.parametrize(
    ("module", "extra"),
    [
        ("matplotlib.pyplot", "viz"),
        ("seaborn", "viz"),
        ("pydicom", "dicom"),
        ("pyarrow", "tables"),
        ("openpyxl", "tables"),
        ("skimage.segmentation", "slic"),
    ],
)
def test_require_raises_optional_dependency_error_with_pip_command(
    module: str, extra: str, hide_module: Any
) -> None:
    """Every demoted dependency must fail with the extra's install command."""
    hide_module(module.split(".")[0])
    with pytest.raises(OptionalDependencyError) as exc_info:
        require(module, extra=extra, purpose="a unit test")
    message = str(exc_info.value)
    assert f'pip install "habitat-analysis[{extra}]"' in message
    assert module in message


def test_require_does_not_mask_an_unrelated_missing_module(
    hide_module: Any,
) -> None:
    """
    A missing dependency *of* an installed package must propagate untouched.

    Claiming "install the extra" when the extra IS installed but broken in some
    unrelated way would send the user down the wrong path.
    """
    hide_module("nonexistent_inner_dependency")
    with mock.patch(
        "habit.utils.optional_deps.importlib.import_module",
        side_effect=ModuleNotFoundError(
            "No module named 'something_else'", name="something_else"
        ),
    ):
        with pytest.raises(ModuleNotFoundError) as exc_info:
            require("json", extra="viz", purpose="a unit test")
    assert not isinstance(exc_info.value, OptionalDependencyError)


def test_require_reports_a_broken_install_distinctly(hide_module: Any) -> None:
    """A present-but-unimportable package needs a different first sentence."""
    with mock.patch(
        "habit.utils.optional_deps.importlib.import_module",
        side_effect=ImportError("undefined symbol: PyArrow_Whatever"),
    ):
        with pytest.raises(OptionalDependencyError) as exc_info:
            require("pyarrow", extra="tables", purpose="parquet export")
    message = str(exc_info.value)
    assert "installed but failed to import" in message
    assert 'pip install "habitat-analysis[tables]"' in message


def test_table_backend_wrappers_point_at_the_tables_extra(
    hide_module: Any,
) -> None:
    """The pandas engine wrappers must name ``tables`` and the CSV escape."""
    hide_module("openpyxl")
    hide_module("pyarrow")

    with pytest.raises(OptionalDependencyError) as excel_info:
        require_excel_backend(purpose="reading a spreadsheet")
    assert 'pip install "habitat-analysis[tables]"' in str(excel_info.value)
    assert ".csv" in str(excel_info.value)

    with pytest.raises(OptionalDependencyError) as parquet_info:
        require_parquet_backend(
            purpose="reading a parquet table",
            alternatives=("use CSV",),
        )
    assert 'pip install "habitat-analysis[tables]"' in str(parquet_info.value)


def test_optional_extra_modules_covers_the_demoted_packages() -> None:
    """
    Each demoted package must map to exactly one extra.

    The reverse direction (every pyproject extra is present in the mapping) is
    asserted in ``tests/test_packaging_contracts.py``.
    """
    module_to_extra = {
        module: extra
        for extra, modules in OPTIONAL_EXTRA_MODULES.items()
        for module in modules
    }
    assert module_to_extra["matplotlib"] == "viz"
    assert module_to_extra["seaborn"] == "viz"
    assert module_to_extra["pydicom"] == "dicom"
    assert module_to_extra["pyarrow"] == "tables"
    assert module_to_extra["openpyxl"] == "tables"
    assert module_to_extra["skimage"] == "slic"
    # numba is required, but ``[accel]`` stays an empty alias like
    # ``[radiomics]`` so the extras matrix still lists it.
    assert module_to_extra["numba"] == "accel"
    # kneed stays a REQUIRED dependency, so it must not claim an extra.
    assert "kneed" not in module_to_extra


def test_pyradiomics_hint_points_at_separate_install() -> None:
    """The public install recipe must document separate PyRadiomics install."""
    hint = pyradiomics_install_hint(python_version=(3, 10))
    assert "pip install pyradiomics" in hint
    assert "conda-forge" in hint
    assert INSTALLATION_DOCS_URL in hint
    assert "habit.install_radiomics" not in hint
    assert "v1.0.2" in hint


def test_pyradiomics_hint_warns_on_unsupported_windows_python() -> None:
    """Unsupported Windows CPython must get an explicit wheel-coverage warning."""
    with mock.patch("habit.utils.optional_deps.sys.platform", "win32"):
        hint = pyradiomics_install_hint(python_version=(3, 9))
    assert "3.9" in hint
    assert "prebuilt PyRadiomics wheels" in hint


def test_pyradiomics_hint_includes_matching_windows_wheel() -> None:
    """Supported Windows CPython hints include the concrete Release wheel URL."""
    with mock.patch("habit.utils.optional_deps.sys.platform", "win32"):
        hint = pyradiomics_install_hint(python_version=(3, 12))
    expected = windows_pyradiomics_wheel_url(python_version=(3, 12))
    assert expected in hint


@pytest.mark.parametrize("minor", [10, 11, 12, 13, 14])
def test_windows_wheel_url_maps_supported_cpython(minor: int) -> None:
    """Each supported CPython minor must map to the matching Release asset."""
    url = windows_pyradiomics_wheel_url(python_version=(3, minor))
    tag = f"cp3{minor}"
    assert url.startswith(
        "https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/"
    )
    assert f"pyradiomics-3.1.0-{tag}-{tag}-win_amd64.whl" in url


def test_windows_wheel_url_rejects_unsupported_python() -> None:
    """Unsupported interpreters must fail loudly instead of inventing a URL."""
    with pytest.raises(ValueError, match="No HABIT prebuilt PyRadiomics wheel"):
        windows_pyradiomics_wheel_url(python_version=(3, 9))


def test_require_pyradiomics_raises_optional_dependency_error() -> None:
    """Missing radiomics must surface OptionalDependencyError, not ModuleNotFoundError."""
    with mock.patch(
        "habit.utils.optional_deps.importlib.import_module",
        side_effect=ModuleNotFoundError(name="radiomics"),
    ), mock.patch(
        "habit.utils.optional_deps.sys.platform",
        "linux",
    ):
        with pytest.raises(OptionalDependencyError) as exc_info:
            require_pyradiomics()
    message = str(exc_info.value)
    assert INSTALLATION_DOCS_URL in message
    assert "habit.install_radiomics" not in message


def test_require_pyradiomics_does_not_auto_install_on_windows() -> None:
    """On Windows, a missing install must raise immediately (no auto-download)."""
    with mock.patch(
        "habit.utils.optional_deps.importlib.import_module",
        side_effect=ModuleNotFoundError(name="radiomics"),
    ), mock.patch(
        "habit.utils.optional_deps.sys.platform",
        "win32",
    ):
        with pytest.raises(OptionalDependencyError) as exc_info:
            require_pyradiomics()
    assert "win_amd64" in str(exc_info.value)
    assert "habit.install_radiomics" not in str(exc_info.value)


def test_require_pyradiomics_returns_module_when_available() -> None:
    """When PyRadiomics is installed (CI / developer envs), import succeeds."""
    pytest.importorskip("radiomics")
    module = require_pyradiomics()
    assert isinstance(module, ModuleType)
    assert module.__name__ == "radiomics"
