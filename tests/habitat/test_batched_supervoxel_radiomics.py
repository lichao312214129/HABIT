import unittest

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from habit.core.habitat_analysis.clustering_features.batched_supervoxel_radiomics import (
    extract_batched_supervoxel_firstorder,
    extract_batched_supervoxel_features,
    _prepare_supervoxel_volumes,
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

    def test_union_bbox_crop_reduces_volume_and_preserves_features(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        embedded_image_array = np.zeros((20, 20, 20), dtype=np.float32)
        embedded_sv_array = np.zeros((20, 20, 20), dtype=np.uint8)
        embedded_image_array[5:15, 5:15, 5:15] = self.image_array
        embedded_sv_array[5:15, 5:15, 5:15] = self.sv_array

        embedded_image = sitk.GetImageFromArray(embedded_image_array)
        embedded_sv = sitk.GetImageFromArray(embedded_sv_array)
        embedded_image.SetSpacing(self.image.GetSpacing())
        embedded_image.SetOrigin(self.image.GetOrigin())
        embedded_sv.SetSpacing(self.image.GetSpacing())
        embedded_sv.SetOrigin(self.image.GetOrigin())

        cropped_image, _, cropped_sv, pad_distance, did_crop = _prepare_supervoxel_volumes(
            embedded_image,
            embedded_sv,
            {"padDistance": 1},
        )
        self.assertTrue(did_crop)
        self.assertEqual(pad_distance, 1)
        self.assertLess(np.prod(cropped_image.GetSize()), np.prod(embedded_image.GetSize()))

        enabled = {"firstorder": ["Mean"], "glcm": ["Contrast"]}
        df_tight = extract_batched_supervoxel_features(
            image=self.image,
            supervoxel_map=self.sv_map,
            labels=self.labels,
            enabled_features=enabled,
            settings=self.settings,
            device="cuda:0",
            dtype_name="float64",
            batch_size=2,
        )
        df_cropped = extract_batched_supervoxel_features(
            image=cropped_image,
            supervoxel_map=cropped_sv,
            labels=self.labels,
            enabled_features=enabled,
            settings={**self.settings, "supervoxelUnionBboxCrop": False},
            device="cuda:0",
            dtype_name="float64",
            batch_size=2,
        )

        cols = [c for c in df_tight.columns if c != "SupervoxelID"]
        merged = df_tight.merge(df_cropped, on="SupervoxelID", suffixes=("_tight", "_crop"))
        for col in cols:
            np.testing.assert_allclose(
                merged[f"{col}_tight"].to_numpy(),
                merged[f"{col}_crop"].to_numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Feature {col} mismatch after union bbox crop",
            )


if __name__ == "__main__":
    unittest.main()
