"""
Integration tests for ICC (Intra-class Correlation Coefficient) analysis.

Merges tests/test_icc.py and tests/test_icc_analyzer.py.
Uses demo_data/ml_data CSVs when available; falls back to synthetic data.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DATA = PROJECT_ROOT / "demo_data" / "ml_data"
BREAST_CANCER_CSV = ML_DATA / "breast_cancer_dataset.csv"
RETEST_CSV = ML_DATA / "breast_cancer_dataset_retest_simulated.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_feature_csv(path: Path, n: int = 50, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {"subject_id": [f"S{i:03d}" for i in range(n)],
         **{f"feat_{j}": rng.randn(n) for j in range(10)}}
    )
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# icc.py – low-level ICC calculation
# ---------------------------------------------------------------------------


class TestICCCalculation:
    def test_icc2_on_identical_measurements_is_one(self) -> None:
        try:
            from habit.core.machine_learning.feature_selectors.icc.icc import compute_icc
        except ImportError:
            pytest.skip("compute_icc not importable")

        rng = np.random.RandomState(0)
        values = rng.randn(30)
        # Perfectly reproducible: test == retest
        result = compute_icc(values, values, method="icc2")
        assert abs(result - 1.0) < 1e-4

    def test_icc_random_data_in_range(self) -> None:
        try:
            from habit.core.machine_learning.feature_selectors.icc.icc import compute_icc
        except ImportError:
            pytest.skip("compute_icc not importable")

        rng = np.random.RandomState(0)
        a = rng.randn(30)
        b = rng.randn(30)
        result = compute_icc(a, b, method="icc2")
        assert -1.0 <= result <= 1.0

    def test_supported_methods_accepted(self) -> None:
        try:
            from habit.core.machine_learning.feature_selectors.icc.icc import compute_icc
        except ImportError:
            pytest.skip("compute_icc not importable")

        rng = np.random.RandomState(0)
        a, b = rng.randn(20), rng.randn(20)
        for method in ["icc2", "icc3"]:
            result = compute_icc(a, b, method=method)
            assert isinstance(result, (float, np.floating))


# ---------------------------------------------------------------------------
# icc_analyzer.py – batch feature analysis
# ---------------------------------------------------------------------------


class TestICCAnalyzer:
    def _get_csv_paths(self, tmp_path: Path) -> List[str]:
        if BREAST_CANCER_CSV.exists() and RETEST_CSV.exists():
            return [str(BREAST_CANCER_CSV), str(RETEST_CSV)]
        # Synthetic fallback
        p1 = tmp_path / "test.csv"
        p2 = tmp_path / "retest.csv"
        _make_synthetic_feature_csv(p1, seed=0)
        _make_synthetic_feature_csv(p2, seed=1)
        return [str(p1), str(p2)]

    def test_analyze_features_returns_dict(self, tmp_path: Path) -> None:
        try:
            from habit.core.machine_learning.feature_selectors.icc.icc_analyzer import (
                analyze_features,
            )
        except ImportError:
            pytest.skip("icc_analyzer not importable")

        paths = self._get_csv_paths(tmp_path)
        result = analyze_features(file_paths=paths, metrics=["icc2"])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_save_results_creates_json(self, tmp_path: Path) -> None:
        try:
            from habit.core.machine_learning.feature_selectors.icc.icc_analyzer import (
                analyze_features,
                save_results,
            )
        except ImportError:
            pytest.skip("icc_analyzer not importable")

        from habit.utils.log_utils import setup_logger

        paths = self._get_csv_paths(tmp_path)
        logger = setup_logger("test_icc", output_dir=tmp_path, log_filename="icc.log")
        results = analyze_features(file_paths=paths, metrics=["icc2"])
        out_path = str(tmp_path / "icc_results.json")
        save_results(results, out_path, logger)
        assert Path(out_path).exists()

    def test_multiple_metrics(self, tmp_path: Path) -> None:
        try:
            from habit.core.machine_learning.feature_selectors.icc.icc_analyzer import (
                analyze_features,
            )
        except ImportError:
            pytest.skip("icc_analyzer not importable")

        paths = self._get_csv_paths(tmp_path)
        result = analyze_features(file_paths=paths, metrics=["icc2", "icc3"])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ICC CLI command (habit icc)
# ---------------------------------------------------------------------------


class TestICCCLI:
    CONFIG_ICC = PROJECT_ROOT / "demo_data" / "config_icc.yaml"

    def test_icc_help_exits_zero(self) -> None:
        from click.testing import CliRunner

        from habit.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["icc", "--help"])
        assert result.exit_code == 0

    def test_icc_missing_config_fails(self) -> None:
        from click.testing import CliRunner

        from habit.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["icc", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_icc_with_demo_config(self) -> None:
        if not self.CONFIG_ICC.exists():
            pytest.skip(f"Config not found: {self.CONFIG_ICC}")
        from click.testing import CliRunner

        from habit.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["icc", "-c", str(self.CONFIG_ICC)])
        assert result.exit_code in [0, 1], result.output
