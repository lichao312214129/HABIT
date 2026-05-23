"""Tests for forwarding voxel_level.params to voxel feature extractors."""

from __future__ import annotations

import unittest

from habit.core.habitat_analysis.services.feature_service import resolve_voxel_step_params


class TestResolveVoxelStepParams(unittest.TestCase):
    def test_merge_global_params_not_in_method_expression(self) -> None:
        step_params = {
            "params_file": "./parameter.yaml",
            "kernelRadius": 3,
        }
        voxel_params = {
            "params_file": "./parameter.yaml",
            "kernelRadius": 3,
            "voxelBatch": 1000,
            "useTorchRadiomics": "auto",
            "torchGpus": [0, 1],
        }
        resolved = resolve_voxel_step_params(step_params, voxel_params)
        self.assertEqual(resolved["voxelBatch"], 1000)
        self.assertEqual(resolved["useTorchRadiomics"], "auto")
        self.assertEqual(resolved["torchGpus"], [0, 1])

    def test_resolve_placeholder_from_global_params(self) -> None:
        step_params = {"params_file": "params_file", "kernelRadius": "kernelRadius"}
        voxel_params = {
            "params_file": "./parameter.yaml",
            "kernelRadius": 3,
            "voxelBatch": 512,
        }
        resolved = resolve_voxel_step_params(step_params, voxel_params)
        self.assertEqual(resolved["params_file"], "./parameter.yaml")
        self.assertEqual(resolved["kernelRadius"], 3)
        self.assertEqual(resolved["voxelBatch"], 512)


if __name__ == "__main__":
    unittest.main()
