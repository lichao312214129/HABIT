"""Unit tests for supervoxel radiomics TorchRadiomics wiring."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor import (
    SupervoxelRadiomicsExtractor,
    _log_supervoxel_feature_class_summary,
    _should_log_supervoxel_progress,
)


class TestSupervoxelProgressLogging(unittest.TestCase):
    def test_should_log_every_supervoxel_when_count_is_small(self) -> None:
        self.assertTrue(_should_log_supervoxel_progress(0, 50))
        self.assertTrue(_should_log_supervoxel_progress(49, 50))

    def test_should_log_sparse_progress_when_count_is_large(self) -> None:
        self.assertTrue(_should_log_supervoxel_progress(0, 500))
        self.assertTrue(_should_log_supervoxel_progress(499, 500))
        self.assertFalse(_should_log_supervoxel_progress(3, 500))
        self.assertTrue(_should_log_supervoxel_progress(24, 500))


class TestSupervoxelFeatureClassLogging(unittest.TestCase):
    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor.logger"
    )
    def test_log_supervoxel_feature_class_summary(self, mock_logger: object) -> None:
        _log_supervoxel_feature_class_summary(
            [
                "original_firstorder_Energy-T2",
                "original_glcm_Contrast-T2",
            ],
            ["firstorder", "glcm"],
            subject="sub1",
            image_name="T2",
            backend="torch",
            matrix_backend="habit_native_c",
        )
        self.assertEqual(mock_logger.info.call_count, 2)
        first_message = mock_logger.info.call_args_list[0][0][0]
        self.assertIn("feature class finished", first_message)
        self.assertEqual(mock_logger.info.call_args_list[0][0][3], "torch")
        self.assertEqual(mock_logger.info.call_args_list[0][0][4], "habit_native_c")


class TestSupervoxelRadiomicsTorchBackend(unittest.TestCase):
    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor."
        "extract_supervoxel_features_pyradiomics"
    )
    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor."
        "create_radiomics_feature_extractor"
    )
    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor."
        "injected_torch_radiomics"
    )
    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor."
        "resolve_voxel_radiomics_backend"
    )
    def test_auto_falls_back_to_pyradiomics_union_bin_path(
        self,
        mock_resolve_backend: MagicMock,
        mock_injected_torch: MagicMock,
        mock_create_extractor: MagicMock,
        mock_extract_pyradiomics: MagicMock,
    ) -> None:
        mock_resolve_backend.return_value = ("pyradiomics", None)
        mock_injected_torch.return_value.__enter__ = MagicMock(return_value=None)
        mock_injected_torch.return_value.__exit__ = MagicMock(return_value=False)

        mock_extractor = MagicMock()
        mock_extractor.enabledFeatures = {"firstorder": [], "glcm": []}
        mock_extractor.settings = {"binWidth": 25}
        mock_create_extractor.return_value = mock_extractor

        mock_extract_pyradiomics.return_value = pd.DataFrame(
            {
                "SupervoxelID": [1, 2],
                "original_firstorder_Energy": [1.0, 2.0],
            }
        )

        image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
        sv_map = sitk.GetImageFromArray(
            np.array([[[0, 1], [2, 0]], [[0, 1], [2, 0]]], dtype=np.uint8)
        )

        extractor = SupervoxelRadiomicsExtractor(params_file="dummy.yaml")
        result = extractor.extract_features(
            image,
            sv_map,
            useTorchRadiomics="auto",
        )

        mock_injected_torch.assert_called_once_with(enabled=False)
        mock_extract_pyradiomics.assert_called_once()
        self.assertEqual(sorted(result["SupervoxelID"].tolist()), [1, 2])

    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor."
        "extract_batched_supervoxel_features"
    )
    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor."
        "create_radiomics_feature_extractor"
    )
    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor."
        "injected_torch_radiomics"
    )
    @patch(
        "habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor."
        "resolve_voxel_radiomics_backend"
    )
    def test_uses_batched_torch_path_when_backend_is_torch(
        self,
        mock_resolve_backend: MagicMock,
        mock_injected_torch: MagicMock,
        mock_create_extractor: MagicMock,
        mock_extract_batched: MagicMock,
    ) -> None:
        mock_resolve_backend.return_value = ("torch", "cuda:0")
        mock_injected_torch.return_value.__enter__ = MagicMock(return_value=None)
        mock_injected_torch.return_value.__exit__ = MagicMock(return_value=False)

        mock_extractor = MagicMock()
        mock_extractor.enabledFeatures = {"firstorder": []}
        mock_extractor.settings = {"binWidth": 25}
        mock_create_extractor.return_value = mock_extractor

        mock_extract_batched.return_value = pd.DataFrame(
            {
                "SupervoxelID": [1],
                "original_firstorder_Energy-T2": [1.0],
            }
        )

        image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
        sv_map = sitk.GetImageFromArray(
            np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], dtype=np.uint8)
        )

        extractor = SupervoxelRadiomicsExtractor(params_file="dummy.yaml")
        result = extractor.extract_features(
            image,
            sv_map,
            subject="sub1",
            image="T2",
            useTorchRadiomics="true",
            supervoxelBatch=32,
        )

        mock_injected_torch.assert_called_once_with(enabled=True)
        mock_extract_batched.assert_called_once()
        call_kwargs = mock_extract_batched.call_args.kwargs
        self.assertEqual(call_kwargs["device"], "cuda:0")
        self.assertEqual(call_kwargs["batch_size"], 32)
        self.assertIn("SupervoxelID", result.columns)


if __name__ == "__main__":
    unittest.main()
