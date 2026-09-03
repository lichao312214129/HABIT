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
Programmatic Python API tests for HABIT domain entry functions.

These tests mock heavy pipeline execution (BatchProcessor.run, workflow.run,
etc.) so they stay fast and do not require demo imaging data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require_config(relative_path: str) -> Path:
    """Return an existing config path or skip the test."""
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        pytest.skip(f"Config not found: {path}")
    return path


class TestPreprocessingAPI:
    """Tests for the public batch preprocessing API."""

    def test_public_runner_invokes_recipe(self, cwd_repo_root: None) -> None:
        """Validated configurations reach the L4 recipe without YAML reloading."""
        from habit.api.preprocessing import PreprocessingConfig, run_preprocess

        cfg_path = _require_config("config/preprocessing/config_preprocessing_demo.yaml")
        config = PreprocessingConfig.from_file(str(cfg_path))

        with patch("habit.recipes.preprocess.preprocess_images") as mock_recipe:
            run_preprocess(config)
            mock_recipe.assert_called_once_with(config, logger=None)


class TestDicomSortAPI:
    """Tests for ``run_dicom_sort`` config loading."""

    def test_dicom_sort_config_from_file(self, cwd_repo_root: None) -> None:
        """DicomSortConfig.from_file resolves paths relative to the YAML."""
        from habit.schemas.workflows.dicom_sort import DicomSortConfig

        cfg_path = _require_config("config/dicom_sort/config_sort_dicom.yaml")
        config = DicomSortConfig.from_file(str(cfg_path))
        assert config.data_dir
        assert config.out_dir or config.output_dir


class TestHabitatAPI:
    """Tests for habitat analysis programmatic entry points."""

    def test_apply_habitat_cli_overrides(self, cwd_repo_root: None) -> None:
        """CLI-style flags should mutate the loaded config in place."""
        from habit.compat.engines.habitat_analysis.config_schemas import HabitatAnalysisConfig
        from habit.compat.engines.habitat_analysis.run import apply_habitat_cli_overrides

        cfg_path = _require_config("config/habitat/config_habitat_two_step.yaml")
        config = HabitatAnalysisConfig.from_file(str(cfg_path))
        apply_habitat_cli_overrides(
            config,
            mode="predict",
            pipeline_path="/tmp/pipeline.pkl",
            debug=True,
            resume=True,
        )
        assert config.run_mode == "predict"
        assert config.pipeline_path == "/tmp/pipeline.pkl"
        assert config.debug is True
        assert config.resume is True

    def test_predict_mode_requires_pipeline_path(self, cwd_repo_root: None) -> None:
        """Predict without pipeline_path must fail before heavy work starts."""
        from habit.compat.engines.habitat_analysis.config_schemas import HabitatAnalysisConfig
        from habit.compat.engines.habitat_analysis.run import run_habitat_analysis_from_config

        cfg_path = _require_config("config/habitat/config_habitat_two_step.yaml")
        config = HabitatAnalysisConfig.from_file(str(cfg_path))
        config.run_mode = "predict"
        config.pipeline_path = None

        with pytest.raises(ValueError, match="pipeline_path"):
            run_habitat_analysis_from_config(config)

    def test_habitat_config_from_demo_yaml(self, cwd_repo_root: None) -> None:
        """Demo habitat YAML should load through the public schema."""
        from habit.compat.engines.habitat_analysis.config_schemas import HabitatAnalysisConfig

        cfg_path = _require_config("config/habitat/config_habitat_two_step.yaml")
        config = HabitatAnalysisConfig.from_file(str(cfg_path))
        assert config.habitat_segmentation.clustering_mode


class TestFeatureExtractionAPI:
    """Tests for feature extraction programmatic entry."""

    def test_public_run_feature_extraction_delegates(
        self,
        cwd_repo_root: None,
    ) -> None:
        """``habit.api.habitat.run_feature_extraction`` delegates to the L4 recipe."""
        from habit.api.habitat import FeatureExtractionConfig, run_feature_extraction

        cfg_path = _require_config(
            "config/feature_extraction/config_extract_features_demo.yaml"
        )
        config = FeatureExtractionConfig.from_file(str(cfg_path))

        with patch(
            "habit.recipes.features.extract_habitat_features"
        ) as mock_run:
            mock_run.return_value = MagicMock(run_id="extract-run")
            run_feature_extraction(config)
            mock_run.assert_called_once_with(
                config,
                plugin_configs=None,
                logger=None,
            )


class TestRadiomicsAndAnalysisAPI:
    """Tests for radiomics, model comparison, and ICC entry points."""

    def test_radiomics_config_from_demo_yaml(self, cwd_repo_root: None) -> None:
        """Radiomics schema loads through the public config module."""
        from habit.compat.engines.habitat_analysis.config_schemas import RadiomicsConfig

        cfg_path = _require_config(
            "config/radiomics/config_traditional_radiomics.yaml"
        )
        config = RadiomicsConfig.from_file(str(cfg_path))
        assert config.paths.out_dir

    def test_public_run_model_comparison_delegates(
        self,
        cwd_repo_root: None,
    ) -> None:
        """``habit.api.machine_learning.run_model_comparison`` delegates to the v1 recipe."""
        from habit.api.machine_learning import (
            ModelComparisonConfig,
            run_model_comparison,
        )

        cfg_path = _require_config(
            "config/model_comparison/config_model_comparison_demo.yaml"
        )
        config = ModelComparisonConfig.from_file(str(cfg_path))

        # The delegate is habit.recipes.comparison.compare_models, not the v0.1
        # ModelComparison engine; habit.api.machine_learning imports it inside
        # the function body, so patching the recipe module is what intercepts
        # the call.
        with patch("habit.recipes.comparison.compare_models") as mock_run:
            run_model_comparison(config)
            mock_run.assert_called_once_with(
                config,
                logger=None,
                output_dir=None,
            )

    def test_icc_config_from_demo_yaml(self, cwd_repo_root: None) -> None:
        """ICC schema loads through ``habit.api.analysis``."""
        from habit.api.analysis import ICCConfig

        cfg_path = _require_config("config/auxiliary/config_icc_demo.yaml")
        config = ICCConfig.from_file(str(cfg_path))
        assert config.output.path

    def test_public_run_icc_analysis_delegates(self, cwd_repo_root: None) -> None:
        """``habit.api.analysis.run_icc_analysis`` delegates to its L4 recipe."""
        from habit.api.analysis import ICCConfig, run_icc_analysis

        cfg_path = _require_config("config/auxiliary/config_icc_demo.yaml")
        config = ICCConfig.from_file(str(cfg_path))

        with patch(
            "habit.recipes.icc_runner.run_icc_analysis_from_config"
        ) as mock_run:
            run_icc_analysis(config)
            mock_run.assert_called_once_with(config)


class TestMachineLearningAPI:
    """Tests for ML programmatic entry points."""

    def test_apply_ml_mode_override_noop_when_same_mode(
        self,
        cwd_repo_root: None,
    ) -> None:
        """Same-mode override should return the original config instance."""
        from habit.compat.engines.machine_learning.config_schemas import MLConfig
        from habit.compat.engines.machine_learning.run import apply_ml_mode_override

        cfg_path = _require_config(
            "config/machine_learning/config_machine_learning_radiomics.yaml"
        )
        config = MLConfig.from_file(str(cfg_path))
        updated = apply_ml_mode_override(config, mode="train")
        assert updated.run_mode == "train"
        assert updated is config

    def test_run_ml_from_config_invokes_workflow_run(
        self,
        cwd_repo_root: None,
    ) -> None:
        """Holdout workflow.run() is the single ML dispatch entry."""
        from habit.compat.engines.machine_learning.config_schemas import MLConfig
        from habit.compat.engines.machine_learning.run import run_ml_from_config

        cfg_path = _require_config(
            "config/machine_learning/config_machine_learning_radiomics.yaml"
        )
        config = MLConfig.from_file(str(cfg_path))

        with patch("habit.compat.engines.machine_learning.run.MLConfigurator") as mock_cfg_cls:
            mock_workflow = MagicMock()
            mock_cfg_cls.return_value.create_ml_workflow.return_value = mock_workflow
            run_ml_from_config(config)
            mock_workflow.run.assert_called_once()

    def test_run_kfold_requires_train_mode(self) -> None:
        """K-fold runner rejects non-train configs without touching data."""
        from habit.compat.engines.machine_learning.run import run_kfold_from_config

        config = MagicMock()
        config.run_mode = "predict"
        config.output = "/tmp/kfold_out"
        with pytest.raises(ValueError, match="run_mode='train'"):
            run_kfold_from_config(config)

    def test_model_comparison_config_from_demo_yaml(
        self,
        cwd_repo_root: None,
    ) -> None:
        """Model comparison schema loads the demo YAML."""
        from habit.compat.engines.machine_learning.config_schemas import ModelComparisonConfig

        cfg_path = _require_config(
            "config/model_comparison/config_model_comparison_demo.yaml"
        )
        config = ModelComparisonConfig.from_file(str(cfg_path))
        assert config.output_dir
        assert len(config.files_config) >= 2
