"""Unit tests for voxel radiomics extractor logging helpers."""

from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from habit.core.habitat_analysis.clustering_features.voxel_radiomics_extractor import (
    _group_voxel_feature_keys_by_class,
    _log_voxel_feature_class_summary,
)
from habit.utils.log_utils import radiomics_feature_class_logging, resolve_radiomics_logging_level


class TestVoxelRadiomicsExtractorHelpers(unittest.TestCase):
    def test_group_voxel_feature_keys_by_class(self) -> None:
        keys = [
            "original_firstorder_Energy",
            "original_firstorder_Entropy",
            "original_glcm_Contrast",
            "original_glrlm_RunLengthNonUniformity",
        ]
        grouped = _group_voxel_feature_keys_by_class(
            keys,
            ["firstorder", "glcm", "glrlm"],
        )
        self.assertEqual(len(grouped["firstorder"]), 2)
        self.assertEqual(len(grouped["glcm"]), 1)
        self.assertEqual(len(grouped["glrlm"]), 1)

    @patch(
        "habit.core.habitat_analysis.clustering_features.voxel_radiomics_extractor.logger"
    )
    def test_log_voxel_feature_class_summary(self, mock_logger: object) -> None:
        _log_voxel_feature_class_summary(
            [
                "original_firstorder_Energy",
                "original_glcm_Contrast",
            ],
            ["firstorder", "glcm"],
            subject="sub1",
            image_name="T2",
        )
        self.assertEqual(mock_logger.info.call_count, 2)
        first_message = mock_logger.info.call_args_list[0][0][0]
        self.assertIn("feature class finished", first_message)

    def test_radiomics_feature_class_logging_restores_level(self) -> None:
        radiomics_logger = logging.getLogger("radiomics")
        radiomics_logger.setLevel(logging.WARNING)
        with radiomics_feature_class_logging():
            self.assertEqual(radiomics_logger.level, logging.INFO)
        self.assertEqual(radiomics_logger.level, logging.WARNING)

    def test_resolve_radiomics_logging_level_debug(self) -> None:
        self.assertEqual(
            resolve_radiomics_logging_level(True),
            logging.DEBUG,
        )
        self.assertEqual(
            resolve_radiomics_logging_level(False),
            logging.INFO,
        )

    def test_radiomics_feature_class_logging_debug_level(self) -> None:
        radiomics_logger = logging.getLogger("radiomics")
        radiomics_logger.setLevel(logging.WARNING)
        with radiomics_feature_class_logging(level=logging.DEBUG):
            self.assertEqual(radiomics_logger.level, logging.DEBUG)
        self.assertEqual(radiomics_logger.level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
