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
    """Tests for ``run_preprocess_from_config`` and BatchProcessor config object."""

    def test_batch_processor_accepts_config_without_yaml_reload(self) -> None:
        """Validated config objects must not trigger a second YAML load."""
        from habit.core.preprocessing.config_schemas import PreprocessingConfig
        from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

        cfg_path = _require_config("config/preprocessing/config_preprocessing_demo.yaml")
        config = PreprocessingConfig.from_file(str(cfg_path))

        with patch(
            "habit.core.preprocessing.image_processor_pipeline.load_config"
        ) as mock_load:
            processor = BatchProcessor(config=config)
            mock_load.assert_not_called()

        assert processor.config_obj.out_dir == config.out_dir

    def test_run_preprocess_from_config_invokes_processor_run(
        self,
        cwd_repo_root: None,
    ) -> None:
        """Domain runner should delegate to BatchProcessor.run()."""
        from habit.core.preprocessing.config_schemas import PreprocessingConfig
        from habit.core.preprocessing.run import run_preprocess_from_config

        cfg_path = _require_config("config/preprocessing/config_preprocessing_demo.yaml")
        config = PreprocessingConfig.from_file(str(cfg_path))

        with patch(
            "habit.core.preprocessing.image_processor_pipeline.BatchProcessor"
        ) as mock_cls:
            mock_processor = MagicMock()
            mock_cls.return_value = mock_processor
            run_preprocess_from_config(config)
            mock_processor.run.assert_called_once()


class TestDicomSortAPI:
    """Tests for ``run_dicom_sort`` config loading."""

    def test_dicom_sort_config_from_file(self, cwd_repo_root: None) -> None:
        """DicomSortConfig.from_file resolves paths relative to the YAML."""
        from habit.core.dicom_sort import DicomSortConfig

        cfg_path = _require_config("config/dicom_sort/config_sort_dicom.yaml")
        config = DicomSortConfig.from_file(str(cfg_path))
        assert config.data_dir
        assert config.out_dir or config.output_dir


class TestHabitatAPI:
    """Tests for habitat analysis programmatic entry points."""

    def test_apply_habitat_cli_overrides(self, cwd_repo_root: None) -> None:
        """CLI-style flags should mutate the loaded config in place."""
        from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
        from habit.core.habitat_analysis.run import apply_habitat_cli_overrides

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
        from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
        from habit.core.habitat_analysis.run import run_habitat_analysis_from_config

        cfg_path = _require_config("config/habitat/config_habitat_two_step.yaml")
        config = HabitatAnalysisConfig.from_file(str(cfg_path))
        config.run_mode = "predict"
        config.pipeline_path = None

        with pytest.raises(ValueError, match="pipeline_path"):
            run_habitat_analysis_from_config(config)

    def test_habitat_config_from_demo_yaml(self, cwd_repo_root: None) -> None:
        """Demo habitat YAML should load through the public schema."""
        from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

        cfg_path = _require_config("config/habitat/config_habitat_two_step.yaml")
        config = HabitatAnalysisConfig.from_file(str(cfg_path))
        assert config.HabitatSegmentation.clustering_mode


class TestFeatureExtractionAPI:
    """Tests for feature extraction programmatic entry."""

    def test_run_feature_extraction_delegates_to_extractor(
        self,
        cwd_repo_root: None,
    ) -> None:
        """Domain runner should call HabitatMapAnalyzer.run()."""
        from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig
        from habit.core.habitat_analysis.run import run_feature_extraction_from_config

        cfg_path = _require_config(
            "config/feature_extraction/config_extract_features_demo.yaml"
        )
        config = FeatureExtractionConfig.from_file(str(cfg_path))

        with patch("habit.core.habitat_analysis.run.HabitatConfigurator") as mock_cfg_cls:
            mock_extractor = MagicMock()
            mock_cfg_cls.return_value.create_feature_extractor.return_value = (
                mock_extractor
            )
            run_feature_extraction_from_config(config)
            mock_extractor.run.assert_called_once_with(
                feature_types=config.feature_types,
                n_habitats=config.n_habitats,
            )


class TestMachineLearningAPI:
    """Tests for ML programmatic entry points."""

    def test_apply_ml_mode_override_noop_when_same_mode(
        self,
        cwd_repo_root: None,
    ) -> None:
        """Same-mode override should return the original config instance."""
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.run import apply_ml_mode_override

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
        from habit.core.machine_learning.config_schemas import MLConfig
        from habit.core.machine_learning.run import run_ml_from_config

        cfg_path = _require_config(
            "config/machine_learning/config_machine_learning_radiomics.yaml"
        )
        config = MLConfig.from_file(str(cfg_path))

        with patch("habit.core.machine_learning.run.MLConfigurator") as mock_cfg_cls:
            mock_workflow = MagicMock()
            mock_cfg_cls.return_value.create_ml_workflow.return_value = mock_workflow
            run_ml_from_config(config)
            mock_workflow.run.assert_called_once()

    def test_run_kfold_requires_train_mode(self) -> None:
        """K-fold runner rejects non-train configs without touching data."""
        from habit.core.machine_learning.run import run_kfold_from_config

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
        from habit.core.machine_learning.config_schemas import ModelComparisonConfig

        cfg_path = _require_config(
            "config/model_comparison/config_model_comparison_demo.yaml"
        )
        config = ModelComparisonConfig.from_file(str(cfg_path))
        assert config.output_dir
        assert len(config.files_config) >= 2
