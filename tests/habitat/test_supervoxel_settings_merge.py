"""Tests for forwarding supervoxel_level.params into radiomics settings."""

from __future__ import annotations

import unittest

from habit.core.habitat_analysis.clustering_features.supervoxel_cext import (
    resolve_use_supervoxel_cext,
    supervoxel_cext_matrix_backend_label,
)
from habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_settings import (
    merge_supervoxel_settings,
)
from habit.core.common.configs.base import ConfigValidationError
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig


class TestMergeSupervoxelSettings(unittest.TestCase):
    def test_use_supervoxel_cext_false_is_forwarded(self) -> None:
        kwargs = {
            "useSupervoxelCext": False,
            "supervoxelUnionBboxCrop": True,
        }
        merged = merge_supervoxel_settings({}, kwargs)
        self.assertIs(merged["useSupervoxelCext"], False)
        self.assertFalse(resolve_use_supervoxel_cext(merged))
        self.assertEqual(
            supervoxel_cext_matrix_backend_label(merged),
            "torch_cmatrices",
        )

    def test_use_supervoxel_cext_auto_defaults_when_missing(self) -> None:
        merged = merge_supervoxel_settings({}, {"supervoxelUnionBboxCrop": True})
        self.assertNotIn("useSupervoxelCext", merged)


class TestSupervoxelLevelConfigSchema(unittest.TestCase):
    def test_rejects_invalid_use_supervoxel_cext(self) -> None:
        with self.assertRaises(ConfigValidationError):
            HabitatAnalysisConfig(
                data_dir="data",
                out_dir="out",
                FeatureConstruction={
                    "voxel_level": {"method": "firstorder()", "params": {}},
                    "supervoxel_level": {
                        "method": "supervoxel_radiomics(T2)",
                        "params": {"useSupervoxelCext": "maybe"},
                    },
                },
                HabitatSegmentation={"clustering_mode": "two_step"},
            )

    def test_accepts_bool_false_use_supervoxel_cext(self) -> None:
        cfg = HabitatAnalysisConfig(
            data_dir="data",
            out_dir="out",
            FeatureConstruction={
                "voxel_level": {"method": "firstorder()", "params": {}},
                "supervoxel_level": {
                    "method": "supervoxel_radiomics(T2)",
                    "params": {"useSupervoxelCext": False},
                },
            },
            HabitatSegmentation={"clustering_mode": "two_step"},
        )
        self.assertIs(
            cfg.FeatureConstruction.supervoxel_level.params["useSupervoxelCext"],
            False,
        )


if __name__ == "__main__":
    unittest.main()
