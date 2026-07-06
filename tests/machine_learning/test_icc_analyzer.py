"""
Unit and integration tests for ICC reliability analysis.

Covers:
- Core metric calculation (ICCMetric, analyze_features)
- ICCConfig loading
- ICC-based feature selector
- Habitat test-retest label mapping
- CLI smoke test via config_icc_demo.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from habit.cli import cli
from habit.core.machine_learning.feature_selectors.icc.config import ICCConfig
from habit.core.machine_learning.feature_selectors.icc.icc import run_icc_analysis_from_config
from habit.core.machine_learning.feature_selectors.icc.icc_analyzer import (
    ICCMetric,
    ICCType,
    analyze_features,
    prepare_long_format,
)
from habit.core.machine_learning.feature_selectors.icc.habitat_test_retest_mapper import (
    find_habitat_mapping,
)
from habit.core.machine_learning.feature_selectors.icc_selector import icc_selector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ML = PROJECT_ROOT / "demo_data" / "ml_data"
ICC_DEMO_CONFIG = PROJECT_ROOT / "config" / "auxiliary" / "config_icc_demo.yaml"
ICC_OUTPUT = PROJECT_ROOT / "demo_data" / "results" / "icc" / "icc_radiomics.json"

TEST_CSV = DEMO_ML / "breast_cancer_dataset.csv"
RETEST_CSV = DEMO_ML / "breast_cancer_dataset_retest_simulated.csv"
HABITATS_TEST = DEMO_ML / "habitats_test.csv"
HABITATS_RETEST = DEMO_ML / "habitats_retest.csv"


@pytest.mark.unit
@pytest.mark.ml
class TestICCMetricCore:
    """Low-level ICC metric calculations."""

    def test_icc3_perfect_agreement(self) -> None:
        """Identical raters should yield ICC(3,1) ~= 1.0."""
        subjects: list[str] = [f"subj{i:03d}" for i in range(10)]
        rater_a: pd.DataFrame = pd.DataFrame({"feature_a": np.arange(10.0)}, index=subjects)
        rater_b: pd.DataFrame = rater_a.copy()
        common_index: pd.Index = pd.Index(subjects)
        long_data: pd.DataFrame = prepare_long_format(
            [rater_a, rater_b],
            "feature_a",
            common_index,
            ["test", "retest"],
        )

        result = ICCMetric(icc_type=ICCType.ICC3).calculate(
            long_data, targets="target", raters="reader", ratings="value"
        )
        assert result.value == pytest.approx(1.0, abs=1e-6)

    def test_analyze_features_demo_csvs(self) -> None:
        """Demo test-retest CSV pair should produce high ICC for known feature."""
        assert TEST_CSV.is_file(), f"Missing demo CSV: {TEST_CSV}"
        assert RETEST_CSV.is_file(), f"Missing demo CSV: {RETEST_CSV}"

        results = analyze_features(
            file_paths=[str(TEST_CSV), str(RETEST_CSV)],
            metrics=["icc2", "icc3"],
            selected_features=["compactness error"],
        )

        group_key = "breast_cancer_dataset_vs_breast_cancer_dataset_retest_simulated"
        assert group_key in results
        feature_result = results[group_key]["compactness error"]
        icc3_value: float = feature_result["ICC3"]["value"]
        assert icc3_value >= 0.75
        assert icc3_value == pytest.approx(0.9403620326397929, rel=1e-4)


@pytest.mark.unit
@pytest.mark.ml
class TestICCConfig:
    """Configuration schema and orchestration."""

    def test_demo_config_loads(self) -> None:
        """config_icc_demo.yaml must parse into ICCConfig."""
        config = ICCConfig.from_file(str(ICC_DEMO_CONFIG))
        assert config.input.type == "files"
        assert len(config.parse_file_groups()) == 1
        assert config.metrics is not None
        assert "icc3" in config.metrics

    def test_run_from_config_writes_json(self, tmp_path: Path) -> None:
        """End-to-end orchestration writes valid JSON with ICC results."""
        out_json = tmp_path / "icc_out.json"
        config = ICCConfig.from_file(str(ICC_DEMO_CONFIG))
        # Override output to tmp to avoid clobbering committed artifacts.
        config.output.path = str(out_json)
        config.metrics = ["icc3"]
        config.selected_features = ["compactness error"]

        run_icc_analysis_from_config(config)

        assert out_json.is_file()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        group = payload[
            "breast_cancer_dataset_vs_breast_cancer_dataset_retest_simulated"
        ]
        assert group["compactness error"]["ICC3"]["value"] >= 0.75


@pytest.mark.unit
@pytest.mark.ml
class TestICCSelector:
    """ICC-based feature filtering for ML pipelines."""

    def test_icc_selector_uses_demo_results(self) -> None:
        """Selector should return stable features from demo ICC JSON."""
        if not ICC_OUTPUT.is_file():
            pytest.skip("Run ICC demo first or use test_cli_icc_demo")

        group_name = (
            "breast_cancer_dataset_vs_breast_cancer_dataset_retest_simulated"
        )
        selected = icc_selector(
            icc_results_path=str(ICC_OUTPUT),
            groups=[group_name],
            threshold=0.75,
            metric="ICC3",
        )
        assert "compactness error" in selected
        assert len(selected) == 13


@pytest.mark.unit
@pytest.mark.ml
class TestHabitatTestRetestMapper:
    """Habitat label remapping before ICC on feature tables."""

    def test_find_habitat_mapping_identity_on_identical_tables(self) -> None:
        """Identical test/retest tables should map each label to itself."""
        assert HABITATS_TEST.is_file()
        assert HABITATS_RETEST.is_file()

        mapping: dict[int, int] = find_habitat_mapping(
            test_habitat_table=str(HABITATS_TEST),
            retest_habitat_table=str(HABITATS_RETEST),
            similarity_method="pearson",
        )
        assert mapping == {1: 1, 2: 2, 3: 3, 4: 4}


@pytest.mark.integration
@pytest.mark.ml
class TestICCMLIntegration:
    """ML pipeline with ICC feature selection on demo data."""

    def test_ml_train_with_icc_demo_config(self) -> None:
        """habit model train should succeed when ICC JSON matches feature column names."""
        if not ICC_OUTPUT.is_file():
            pytest.skip("Run habit icc -c config/auxiliary/config_icc_demo.yaml first")

        runner = CliRunner()
        config_path = (
            PROJECT_ROOT
            / "config"
            / "machine_learning"
            / "config_machine_learning_radiomics_icc_demo.yaml"
        )
        result = runner.invoke(
            cli,
            ["model", "-c", str(config_path), "-m", "train"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output


@pytest.mark.integration
@pytest.mark.ml
class TestICCCLI:
    """CLI entry point for habit icc."""

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["icc", "--help"])
        assert result.exit_code == 0

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["icc", "-c", "nonexistent_icc.yaml"])
        assert result.exit_code != 0

    def test_cli_icc_demo(self) -> None:
        """Full CLI run on bundled demo_data CSVs."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["icc", "-c", str(ICC_DEMO_CONFIG)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert ICC_OUTPUT.is_file()
