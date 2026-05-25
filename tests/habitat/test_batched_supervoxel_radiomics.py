import unittest

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from habit.core.habitat_analysis.clustering_features.batched_supervoxel_radiomics import (
    extract_batched_supervoxel_firstorder,
    extract_batched_supervoxel_features,
)


class TestBatchedSupervoxelRadiomics(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.image_array = np.random.randint(0, 100, size=(10, 10, 10)).astype(np.float32)
        self.image = sitk.GetImageFromArray(self.image_array)

        self.sv_array = np.zeros((10, 10, 10), dtype=np.uint8)
        self.sv_array[1:4, 1:4, 1:4] = 1
        self.sv_array[5:8, 5:8, 5:8] = 2
        self.sv_array[8:10, 1:3, 1:3] = 3
        self.sv_map = sitk.GetImageFromArray(self.sv_array)
        self.sv_map.CopyInformation(self.image)
        self.labels = np.array([1, 2, 3])
        self.settings = {"binWidth": 25, "voxelArrayShift": 0}

    def test_batched_firstorder_matches_sequential_batches(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        df_batched = extract_batched_supervoxel_firstorder(
            image=self.image,
            supervoxel_map=self.sv_map,
            labels=self.labels,
            device="cuda:0",
            dtype=torch.float64,
            batch_size=2,
            settings=self.settings,
        )

        df_ref = extract_batched_supervoxel_firstorder(
            image=self.image,
            supervoxel_map=self.sv_map,
            labels=self.labels,
            device="cuda:0",
            dtype=torch.float64,
            batch_size=1,
            settings=self.settings,
        )

        cols = [c for c in df_batched.columns if c != "SupervoxelID"]
        merged = df_batched.merge(df_ref, on="SupervoxelID", suffixes=("_batched", "_ref"))
        for col in cols:
            np.testing.assert_allclose(
                merged[f"{col}_batched"].to_numpy(),
                merged[f"{col}_ref"].to_numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Feature {col} mismatch",
            )

    def test_batched_glcm_runs(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        enabled = {"glcm": ["Contrast", "Correlation", "JointEntropy"]}
        df = extract_batched_supervoxel_features(
            image=self.image,
            supervoxel_map=self.sv_map,
            labels=self.labels,
            enabled_features=enabled,
            settings=self.settings,
            device="cuda:0",
            dtype_name="float64",
            batch_size=2,
        )
        self.assertEqual(len(df), 3)
        self.assertIn("original_glcm_Contrast", df.columns)

    def test_batched_all_texture_classes_match_batch_one(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        enabled = {
            "firstorder": ["Mean"],
            "glcm": ["Contrast"],
            "glrlm": ["ShortRunEmphasis"],
            "glszm": ["SmallAreaEmphasis"],
            "ngtdm": ["Coarseness"],
            "gldm": ["SmallDependenceEmphasis"],
        }

        df_batched = extract_batched_supervoxel_features(
            image=self.image,
            supervoxel_map=self.sv_map,
            labels=self.labels,
            enabled_features=enabled,
            settings=self.settings,
            device="cuda:0",
            dtype_name="float64",
            batch_size=2,
        )

        df_ref = extract_batched_supervoxel_features(
            image=self.image,
            supervoxel_map=self.sv_map,
            labels=self.labels,
            enabled_features=enabled,
            settings=self.settings,
            device="cuda:0",
            dtype_name="float64",
            batch_size=1,
        )

        cols = [c for c in df_batched.columns if c != "SupervoxelID"]
        merged = df_batched.merge(df_ref, on="SupervoxelID", suffixes=("_batched", "_ref"))
        for col in cols:
            np.testing.assert_allclose(
                merged[f"{col}_batched"].to_numpy(),
                merged[f"{col}_ref"].to_numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Feature {col} mismatch",
            )


if __name__ == "__main__":
    unittest.main()
