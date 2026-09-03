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
"""Import gate for every optional third-party backend HABIT can use.

HABIT's required dependency set is deliberately small: only the habitat kernel
(numpy / scipy / pandas / scikit-learn / SimpleITK / networkx / numba) plus
the thin plumbing layer (pydantic / PyYAML / click / tqdm / joblib / kneed)
is installed by a bare ``pip install habitat-analysis``. Everything else --
plotting, DICOM, parquet/xlsx tables, SLIC supervoxels, tabular ML, survival
analysis, ANTs registration, PyRadiomics -- lives behind a pip extra.

:func:`require` is the single entry point every optional backend must go
through. It converts the raw ``ModuleNotFoundError`` a missing extra would
otherwise produce into an :class:`~habit.exceptions.OptionalDependencyError`
carrying a copy-pasteable install command, so a user who hits a missing
backend never has to guess which extra provides it.

PyRadiomics is the one backend that cannot be expressed as a pip extra: PyPI
has no usable Windows binaries for CPython 3.10-3.14 (broken 3.1.0 sdist; no
3.0.1 win_amd64 wheels), and PyPI rejects direct wheel URLs in uploaded
metadata. It therefore keeps its own hint builder
(:func:`pyradiomics_install_hint`) and its own thin gate
(:func:`require_pyradiomics`), which point at the prebuilt wheels published on
the HABIT GitHub Release ``v1.0.2`` (Windows) or PyPI / conda-forge
(macOS / Linux).
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Mapping, Optional, Sequence

from habit.exceptions import OptionalDependencyError

#: PyPI distribution name. The import package is ``habit``, but the pip
#: command a user must run names the distribution, so every generated hint
#: has to use this string and not ``habit``.
DISTRIBUTION_NAME: str = "habitat-analysis"

#: Every pip extra declared in ``pyproject.toml``
#: ``[project.optional-dependencies]``, mapped to the top-level importable
#: modules it makes available.
#:
#: This is the machine-readable half of the extras matrix: :func:`require`
#: validates its ``extra`` argument against these keys so a typo in a call
#: site cannot ship an install command that does not resolve, and
#: ``tests/test_packaging_contracts.py`` cross-checks the keys against the
#: extras actually declared in ``pyproject.toml``.
#:
#: ``radiomics`` maps to ``radiomics`` even though the extra is an empty
#: documented alias (PyRadiomics is installed separately); the mapping keeps
#: the module -> extra direction complete for diagnostics.
#: The ``all`` / ``full`` meta-extras are intentionally absent: they exist for
#: users, never as the target of a single missing-module hint.
OPTIONAL_EXTRA_MODULES: Mapping[str, tuple[str, ...]] = {
    "viz": ("matplotlib", "seaborn"),
    "view": ("napari",),
    "dicom": ("pydicom",),
    "tables": ("pyarrow", "openpyxl"),
    "slic": ("skimage",),
    "ml": ("xgboost", "imblearn", "mrmr", "statsmodels"),
    "analysis": (
        "krippendorff",
        "shap",
        "plotly",
        "pingouin",
        "lifelines",
        "sksurv",
    ),
    "automl": ("autogluon",),
    "registration": ("ants",),
    "torch": ("torch",),
    "monai": ("monai",),
    "radiomics": ("radiomics",),
    # Empty documented alias: numba is now required. Mapping is kept so
    # extras coverage stays complete and ``require("numba", extra="accel")``
    # still resolves; kernels keep a silent ``try: import numba`` fallback.
    "accel": ("numba",),
}

#: CPython minor versions for which HABIT publishes ``win_amd64`` wheels.
SUPPORTED_WINDOWS_CPYTHON_MINORS: tuple[int, ...] = (10, 11, 12, 13, 14)

#: GitHub Release tag that hosts the HABIT-built PyRadiomics Windows wheels.
PYRADIOMICS_WHEEL_RELEASE: str = "v1.0.2"

#: Upstream PyRadiomics version encoded in the Release wheel filenames.
PYRADIOMICS_WHEEL_VERSION: str = "3.1.0"

#: Docs page with the platform-specific install recipe and wheel table.
INSTALLATION_DOCS_URL: str = (
    "https://lichao312214129.github.io/HABIT/tutorial/installation.html"
)

#: Human-readable install recipe shown whenever PyRadiomics is missing.
PYRADIOMICS_INSTALL_HINT: str = (
    "PyRadiomics is required for radiomics / voxel_radiomics / "
    "supervoxel_radiomics workflows.\n"
    "\n"
    "Install PyRadiomics separately (HABIT does not pull it via pip extras):\n"
    "\n"
    "  Windows (prebuilt wheel from the HABIT GitHub Release v1.0.2;\n"
    "  pick the URL matching your CPython — do NOT use bare\n"
    "  ``pip install pyradiomics``, which downloads a broken sdist):\n"
    "    pip install https://github.com/lichao312214129/HABIT/releases/"
    "download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl\n"
    "    # … or cp311 / cp312 / cp313 / cp314 — see the Installation docs\n"
    "\n"
    "  macOS / Linux:\n"
    "    pip install \"pyradiomics>=3.0.1,<3.2\"\n"
    "    # or: conda install -c conda-forge pyradiomics\n"
    "\n"
    f"Full wheel table and notes: {INSTALLATION_DOCS_URL}"
)


def install_command(extra: str) -> str:
    """
    Build the pip command that installs one HABIT extra.

    Args:
        extra: Extra name declared in ``[project.optional-dependencies]``.

    Returns:
        str: A copy-pasteable command, quoted so shells that treat ``[`` as a
        glob character (zsh, PowerShell) do not mangle it.

    Raises:
        ValueError: When ``extra`` is not a declared HABIT extra.
    """
    if extra not in OPTIONAL_EXTRA_MODULES:
        known = ", ".join(sorted(OPTIONAL_EXTRA_MODULES))
        raise ValueError(
            f"Unknown HABIT extra {extra!r}. Declared extras: {known}."
        )
    return f'pip install "{DISTRIBUTION_NAME}[{extra}]"'


def optional_dependency_hint(
    module: str,
    *,
    extra: str,
    purpose: str,
    alternatives: Sequence[str] = (),
) -> str:
    """
    Compose the user-facing message for a missing optional dependency.

    The message answers the only three questions a user has at that moment:
    what is missing, what was it needed for, and what exactly do I type.

    Args:
        module: Importable module name that could not be imported, for
            example ``matplotlib.pyplot``.
        extra: HABIT extra that provides the module.
        purpose: One-line description of what HABIT needed the module for,
            phrased to complete "... is required for <purpose>".
        alternatives: Extra escape routes to list after the install command,
            for example a configuration switch that avoids the dependency
            entirely. Each entry is rendered as its own bullet.

    Returns:
        str: Multi-line message ending with the docs URL.

    Raises:
        ValueError: When ``extra`` is not a declared HABIT extra.
    """
    lines = [
        f"{module} is required for {purpose}, but it is not installed.",
        "",
        "It is an OPTIONAL HABIT dependency. Install the extra that "
        "provides it:",
        "",
        f"  {install_command(extra)}",
    ]
    if alternatives:
        lines.extend(["", "Alternatively:"])
        lines.extend(f"  - {alternative}" for alternative in alternatives)
    lines.extend(
        [
            "",
            "Every extra and what it unlocks: " f"{INSTALLATION_DOCS_URL}",
        ]
    )
    return "\n".join(lines)


def require(
    module: str,
    *,
    extra: str,
    purpose: str,
    alternatives: Sequence[str] = (),
) -> ModuleType:
    """
    Import an optional module or raise :class:`OptionalDependencyError`.

    Every optional backend in HABIT is imported through this function instead
    of a bare ``import``. The point is the failure mode: a bare import raises
    ``ModuleNotFoundError: No module named 'matplotlib'``, which tells a user
    nothing about which HABIT extra to install. This raises
    ``OptionalDependencyError`` with the exact pip command instead.

    Nothing is installed or downloaded; the function only imports and, on
    failure, explains.

    Args:
        module: Module to import. May be a submodule (``matplotlib.pyplot``);
            the returned object is then that submodule, matching
            ``import matplotlib.pyplot as plt`` semantics rather than
            ``import matplotlib.pyplot``.
        extra: HABIT extra that provides the module.
        purpose: One-line description of what the module is needed for, used
            verbatim in the error message.
        alternatives: Optional escape routes shown alongside the pip command
            (see :func:`optional_dependency_hint`).

    Returns:
        ModuleType: The imported module (or submodule) object.

    Raises:
        OptionalDependencyError: When the module is absent, or present but
            broken (failed C extension, wrong ABI, incompatible version).
        ValueError: When ``extra`` is not a declared HABIT extra.
    """
    # Validate the extra before attempting the import so a typo fails the
    # same way whether or not the dependency happens to be installed.
    install_command(extra)
    root = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = str(exc.name or "")
        # Only claim "the extra is missing" when the failure really is about
        # the requested distribution. A ModuleNotFoundError raised from deep
        # inside an installed package (a genuine bug in that package, or a
        # different missing dependency of it) must propagate untouched.
        if missing and missing != root and not missing.startswith(f"{root}."):
            raise
        raise OptionalDependencyError(
            optional_dependency_hint(
                module,
                extra=extra,
                purpose=purpose,
                alternatives=alternatives,
            )
        ) from exc
    except ImportError as exc:
        raise OptionalDependencyError(
            f"{module} is installed but failed to import "
            f"({type(exc).__name__}: {exc}).\n\n"
            + optional_dependency_hint(
                module,
                extra=extra,
                purpose=purpose,
                alternatives=alternatives,
            )
        ) from exc


def require_excel_backend(*, purpose: str) -> ModuleType:
    """
    Gate an ``.xlsx`` read/write behind the ``tables`` extra.

    ``openpyxl`` is never imported by HABIT directly -- it is the engine
    ``pandas.read_excel`` / ``DataFrame.to_excel`` load at run time. Without
    this gate, a missing openpyxl surfaces as pandas' own
    ``ImportError: Missing optional dependency 'openpyxl'``, which names
    neither HABIT nor the extra that provides it.

    Args:
        purpose: What the spreadsheet is being read or written for.

    Returns:
        ModuleType: The imported ``openpyxl`` module (callers normally ignore
        it and let pandas resolve the engine itself).

    Raises:
        OptionalDependencyError: When openpyxl is not installed.
    """
    return require(
        "openpyxl",
        extra="tables",
        purpose=purpose,
        alternatives=(
            "convert the spreadsheet to .csv -- HABIT reads and writes CSV "
            "with no optional dependency at all",
        ),
    )


def require_parquet_backend(
    *,
    purpose: str,
    alternatives: Sequence[str] = (),
) -> ModuleType:
    """
    Gate a parquet read/write behind the ``tables`` extra.

    Like openpyxl, ``pyarrow`` is a pandas run-time engine rather than a HABIT
    import, so the gate has to sit at the call site.

    Args:
        purpose: What the parquet table is being read or written for.
        alternatives: Escape routes to list in addition to installing the
            extra, for example the configuration switch that selects CSV.

    Returns:
        ModuleType: The imported ``pyarrow`` module.

    Raises:
        OptionalDependencyError: When pyarrow is not installed.
    """
    return require(
        "pyarrow",
        extra="tables",
        purpose=purpose,
        alternatives=alternatives,
    )


def windows_pyradiomics_wheel_url(
    *,
    python_version: Optional[tuple[int, int]] = None,
    release: str = PYRADIOMICS_WHEEL_RELEASE,
    pyradiomics_version: str = PYRADIOMICS_WHEEL_VERSION,
) -> str:
    """
    Return the GitHub Release URL for the prebuilt Windows PyRadiomics wheel.

    Args:
        python_version: ``(major, minor)`` interpreter tag; defaults to the
            running interpreter.
        release: HABIT GitHub Release tag that hosts the wheel assets.
        pyradiomics_version: Version string embedded in the wheel filename.

    Returns:
        Absolute HTTPS URL of the ``win_amd64`` wheel.

    Raises:
        ValueError: When no wheel exists for the requested CPython version.
    """
    version = python_version or sys.version_info[:2]
    if version[0] != 3 or version[1] not in SUPPORTED_WINDOWS_CPYTHON_MINORS:
        supported = ", ".join(f"3.{minor}" for minor in SUPPORTED_WINDOWS_CPYTHON_MINORS)
        raise ValueError(
            f"No HABIT prebuilt PyRadiomics wheel for Python {version[0]}.{version[1]} "
            f"on Windows. Supported: {supported}."
        )
    tag = f"cp3{version[1]}"
    filename = (
        f"pyradiomics-{pyradiomics_version}-{tag}-{tag}-win_amd64.whl"
    )
    return (
        f"https://github.com/lichao312214129/HABIT/releases/download/"
        f"{release}/{filename}"
    )


def pyradiomics_install_hint(*, python_version: Optional[tuple[int, int]] = None) -> str:
    """
    Return the install hint, with an extra warning on unsupported Windows Python.

    Args:
        python_version: ``(major, minor)`` override; defaults to the running
            interpreter.

    Returns:
        Multi-line install guidance string.
    """
    version = python_version or sys.version_info[:2]
    hint = PYRADIOMICS_INSTALL_HINT
    if sys.platform == "win32":
        if version[0] != 3 or version[1] not in SUPPORTED_WINDOWS_CPYTHON_MINORS:
            supported = ", ".join(
                f"3.{minor}" for minor in SUPPORTED_WINDOWS_CPYTHON_MINORS
            )
            hint += (
                "\n\n"
                f"This interpreter is Python {version[0]}.{version[1]} on Windows. "
                f"HABIT publishes prebuilt PyRadiomics wheels only for {supported}. "
                "Switch to a supported Python, or install PyRadiomics another way."
            )
        else:
            try:
                url = windows_pyradiomics_wheel_url(python_version=version)
            except ValueError:
                pass
            else:
                hint += (
                    "\n\n"
                    f"For this interpreter, the matching wheel is:\n"
                    f"  pip install {url}"
                )
    return hint


def require_pyradiomics() -> ModuleType:
    """
    Import the ``radiomics`` package or raise :class:`OptionalDependencyError`.

    This is the PyRadiomics specialization of :func:`require`. It cannot be
    replaced by ``require("radiomics", extra="radiomics", ...)`` because the
    ``radiomics`` extra is an empty documented alias: the install recipe is
    platform-dependent (a GitHub Release ``win_amd64`` wheel URL on Windows,
    PyPI or conda-forge elsewhere), so the hint has to be computed rather
    than templated from an extra name. See :func:`pyradiomics_install_hint`.

    No automatic download or install is attempted; the error message points at
    the documented separate-install recipe (Windows Release wheel / PyPI /
    conda-forge).

    Returns:
        The imported ``radiomics`` module.

    Raises:
        OptionalDependencyError: When PyRadiomics is not installed / importable.
    """
    try:
        return importlib.import_module("radiomics")
    except ModuleNotFoundError as exc:
        if exc.name not in {None, "radiomics"} and not str(exc.name or "").startswith(
            "radiomics."
        ):
            raise
        raise OptionalDependencyError(pyradiomics_install_hint()) from exc
    except ImportError as exc:
        # Broken partial installs (failed C extension, wrong ABI, …).
        raise OptionalDependencyError(
            "PyRadiomics is installed but failed to import "
            f"({type(exc).__name__}: {exc}).\n\n" + pyradiomics_install_hint()
        ) from exc
