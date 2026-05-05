"""
Shared pytest fixtures for the entire test suite.

Fixtures are organized by scope:
- session-scoped: demo data paths, heavy one-time resources
- function-scoped: temporary directories, synthetic DataFrames
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Project root and demo data paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA_DIR = PROJECT_ROOT / "demo_data"
ML_DATA_DIR = DEMO_DATA_DIR / "ml_data"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def cwd_repo_root(monkeypatch: pytest.MonkeyPatch, project_root: Path) -> None:
    """
    Set process cwd to the repository root for CLI tests.

    Demo YAML often uses paths relative to where users run ``habit`` (usually
    repo root). ``CliRunner`` still inherits pytest's cwd unless overridden.
    """
    monkeypatch.chdir(project_root)


@pytest.fixture(scope="session")
def demo_data_dir() -> Path:
    """Absolute path to demo_data/."""
    return DEMO_DATA_DIR


@pytest.fixture(scope="session")
def ml_data_dir() -> Path:
    """Absolute path to demo_data/ml_data/."""
    return ML_DATA_DIR


@pytest.fixture(scope="session")
def breast_cancer_csv(ml_data_dir: Path) -> Path:
    """Path to the breast-cancer CSV used for ML integration tests."""
    p = ml_data_dir / "breast_cancer_dataset.csv"
    if not p.exists():
        pytest.skip(f"Demo CSV not found: {p}")
    return p


@pytest.fixture(scope="session")
def clinical_feature_csv(ml_data_dir: Path) -> Path:
    """Path to the clinical feature CSV."""
    p = ml_data_dir / "clinical_feature.csv"
    if not p.exists():
        pytest.skip(f"Demo CSV not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Temporary output directory (function scope, auto-cleaned)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> str:
    """Return a temporary output directory as str (matches config expectation)."""
    out = tmp_path / "output"
    out.mkdir()
    return str(out)


# ---------------------------------------------------------------------------
# Synthetic tabular data
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Small reproducible binary-classification dataset (60 samples, 10 features).

    Returns
    -------
    X : pd.DataFrame  (60, 10)
    y : pd.Series     binary 0/1 labels
    """
    rng = np.random.RandomState(42)
    n, p = 60, 10
    X = pd.DataFrame(
        rng.randn(n, p),
        columns=[f"feature_{i}" for i in range(p)],
    )
    y = pd.Series((rng.randn(n) > 0).astype(int), name="label")
    return X, y


@pytest.fixture
def binary_df_with_label(binary_classification_data) -> pd.DataFrame:
    """Single DataFrame with features + 'label' column, suitable for CSV-based tests."""
    X, y = binary_classification_data
    df = X.copy()
    df["label"] = y.values
    df.index.name = "subject_id"
    df = df.reset_index()
    return df


@pytest.fixture
def prediction_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-computed y_true / y_prob pair with known AUC > 0.7 for metric tests.
    """
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([
        rng.beta(2, 5, 50),   # negatives cluster near 0
        rng.beta(5, 2, 50),   # positives cluster near 1
    ])
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Minimal logger
# ---------------------------------------------------------------------------


@pytest.fixture
def logger() -> logging.Logger:
    """Silent logger for use inside unit tests."""
    log = logging.getLogger("test")
    log.setLevel(logging.CRITICAL)  # suppress noise during tests
    return log


# ---------------------------------------------------------------------------
# Minimal MLConfig factory helper
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_ml_config_dict(tmp_path: Path) -> dict:
    """
    Smallest valid dict that can be validated into an MLConfig.
    Tests that need a real CSV can override 'path'.
    """
    dummy_csv = tmp_path / "dummy.csv"
    dummy_csv.write_text("subject_id,feature_0,label\nS1,1.0,0\nS2,2.0,1\n")
    return {
        "input": [
            {
                "path": str(dummy_csv),
                "subject_id_col": "subject_id",
                "label_col": "label",
            }
        ],
        "output": str(tmp_path / "out"),
        "feature_selection_methods": [],
        "models": {
            "LogisticRegression": {"params": {"max_iter": 100}},
        },
    }
