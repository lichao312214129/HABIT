"""Unit tests for supervoxel radiomics TorchRadiomics wiring."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import SimpleITK as sitk

from habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor import (
    SupervoxelRadiomicsExtractor,
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


class TestSupervoxelRadiomicsTorchBackend(unittest.TestCase):
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
    def test_uses_torch_injection_when_backend_is_torch(
        self,
        mock_resolve_backend: MagicMock,
        mock_injected_torch: MagicMock,
        mock_create_extractor: MagicMock,
    ) -> None:
        mock_resolve_backend.return_value = ("torch", "cuda:0")
        mock_injected_torch.return_value.__enter__ = MagicMock(return_value=None)
        mock_injected_torch.return_value.__exit__ = MagicMock(return_value=False)

        mock_extractor = MagicMock()
        mock_extractor.enabledFeatures = {"firstorder": []}
        mock_extractor.execute.return_value = {
            "original_firstorder_Energy": 1.0,
        }
        mock_create_extractor.return_value = mock_extractor

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
        )

        mock_injected_torch.assert_called_once_with(enabled=True)
        mock_extractor.settings.update.assert_called_once()
        settings = mock_extractor.settings.update.call_args[0][0]
        self.assertEqual(settings["device"], "cuda:0")
        self.assertGreaterEqual(mock_extractor.execute.call_count, 1)
        self.assertIn("SupervoxelID", result.columns)

        # Scheme A: pass the multi-label supervoxel map directly (not a rebuilt binary mask).
        for call in mock_extractor.execute.call_args_list:
            self.assertIs(call.args[1], sv_map)
            self.assertEqual(call.kwargs.get("label"), 1)

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
    def test_execute_uses_each_supervoxel_label_from_map(
        self,
        mock_resolve_backend: MagicMock,
        mock_injected_torch: MagicMock,
        mock_create_extractor: MagicMock,
    ) -> None:
        mock_resolve_backend.return_value = ("pyradiomics", None)
        mock_injected_torch.return_value.__enter__ = MagicMock(return_value=None)
        mock_injected_torch.return_value.__exit__ = MagicMock(return_value=False)

        mock_extractor = MagicMock()
        mock_extractor.enabledFeatures = {"firstorder": []}
        mock_extractor.execute.return_value = {"original_firstorder_Energy": 1.0}
        mock_create_extractor.return_value = mock_extractor

        image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
        sv_map = sitk.GetImageFromArray(
            np.array([[[0, 1], [2, 0]], [[0, 3], [3, 0]]], dtype=np.uint8)
        )

        extractor = SupervoxelRadiomicsExtractor(params_file="dummy.yaml")
        result = extractor.extract_features(image, sv_map, useTorchRadiomics="false")

        labels_used = {call.kwargs["label"] for call in mock_extractor.execute.call_args_list}
        self.assertEqual(labels_used, {1, 2, 3})
        self.assertEqual(sorted(result["SupervoxelID"].tolist()), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
