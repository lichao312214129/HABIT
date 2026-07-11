"""Setuptools build configuration for the HABIT package."""

from pathlib import Path
from typing import Dict

import numpy as np
from setuptools import Extension, find_packages, setup

_ROOT = Path(__file__).resolve().parent
_CEXT_SRC = "habit/core/habitat_analysis/clustering_features/supervoxel_cext/src"
_SV_CMATRICES_MODULE = (
    "habit.core.habitat_analysis.clustering_features.supervoxel_cext._sv_cmatrices"
)


def _read_version() -> str:
    """
    Read the package version without importing ``habit`` during the build.

    Importing the package here would execute its public API initialization before
    runtime dependencies are installed in the isolated build environment.

    Returns:
        str: The version declared in ``habit/_version.py``.
    """
    version_scope: Dict[str, object] = {}
    version_file = _ROOT / "habit" / "_version.py"
    exec(
        compile(version_file.read_text(encoding="utf-8"), str(version_file), "exec"),
        version_scope,
    )
    return str(version_scope["__version__"])


setup(
    name="HABIT",
    version=_read_version(),
    description="Habitat Analysis: Biomedical Imaging Toolkit",
    author="lichao19870617@163.com",
    license="HABIT Software License",
    # Restrict discovery to HABIT itself. The repository's ``tests`` directory
    # is an importable package but must not be installed into user environments.
    packages=find_packages(include=("habit", "habit.*")),
    include_package_data=True,
    package_data={
        # Bundled PyRadiomics parameter presets (default params_file fallbacks).
        "habit": ["py.typed"],
        "habit.resources.radiomics": ["*.yaml"],
    },
    ext_modules=[
        Extension(
            _SV_CMATRICES_MODULE,
            [
                f"{_CEXT_SRC}/_sv_cmatrices.c",
                f"{_CEXT_SRC}/sv_cmatrices.c",
            ],
            include_dirs=[_CEXT_SRC, np.get_include()],
        ),
    ],
    install_requires=[
        "click>=8.0",
        "SimpleITK==2.2.1",
        "antspyx==0.4.2",
        "opencv-python",
        "numpy",
        "matplotlib",
        "trimesh",
        "scipy",
        "openpyxl",
        "pandas",
        "pyarrow",
        "tqdm",
        "pyradiomics",
        "scikit-learn",
        "scikit-image",
        "mrmr_selection",
        "pingouin",
        "statsmodels",
        "xgboost",
        "seaborn",
        "shap",
        "pyyaml",
        "lifelines",
    ],
    extras_require={
        "gui": [
            "fastapi>=0.100",
            "uvicorn[standard]>=0.22",
        ],
    },
    entry_points={
        "console_scripts": [
            "habit=habit.cli:cli",
        ],
    },
    python_requires=">=3.10,<3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.10",
    ],
)
