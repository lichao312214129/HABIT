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
"""Numeric parity: native C+CPU formulas vs union-bin PyRadiomics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
import SimpleITK as sitk

from habit.kernels.radiomics.cext import cext_backend
from habit.kernels.radiomics.native_batch import extract_native_supervoxel_features
from habit.kernels.radiomics.supervoxel_batch import (
    extract_supervoxel_features_pyradiomics,
)


def _two_blob_case() -> tuple[sitk.Image, sitk.Image, np.ndarray]:
    """Two compact labels inside a padded box (union crop leaves interior ROI)."""
    rng = np.random.default_rng(11)
    image = rng.normal(loc=20.0, scale=40.0, size=(18, 24, 24)).astype(np.float64)
    sv_map = np.zeros((18, 24, 24), dtype=np.int32)
    sv_map[4:10, 4:12, 4:12] = 1
    sv_map[10:16, 12:20, 12:20] = 2
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetSpacing((1.0, 1.0, 2.0))
    sitk_map = sitk.GetImageFromArray(sv_map)
    sitk_map.CopyInformation(sitk_image)
    return sitk_image, sitk_map, np.asarray([1, 2], dtype=np.int64)


def _settings(use_cext: object = "auto") -> Dict[str, object]:
    return {
        "binWidth": 12,
        "voxelArrayShift": 300,
        "normalize": False,
        "distances": [1],
        "force2D": False,
        "symmetricalGLCM": True,
        "gldm_a": 0,
        "use_supervoxel_cext": use_cext,
        "supervoxel_union_bbox_crop": True,
        "padDistance": 1,
    }


ENABLED = {
    "firstorder": ["Mean", "Energy", "TotalEnergy", "Entropy", "10Percentile"],
    "glcm": ["Id", "JointEntropy", "Autocorrelation", "JointAverage", "MCC"],
    "glrlm": ["ShortRunEmphasis", "HighGrayLevelRunEmphasis"],
    "glszm": ["ZonePercentage", "SmallAreaEmphasis"],
    "gldm": ["SmallDependenceEmphasis", "HighGrayLevelEmphasis"],
    "ngtdm": ["Coarseness"],
}


@pytest.mark.unit
def test_energy_matches_pyradiomics_union_bin() -> None:
    """Energy / TotalEnergy follow voxelArrayShift and prod(spacing)."""
    from radiomics.firstorder import RadiomicsFirstOrder

    image, sv_map, _labels = _two_blob_case()
    settings = _settings()
    native = extract_native_supervoxel_features(
        image,
        sv_map,
        np.asarray([1], dtype=np.int64),
        enabled_features={"firstorder": ["Energy", "TotalEnergy", "Mean"]},
        settings=settings,
        union_bin=True,
    )
    mask = sitk.BinaryThreshold(sv_map, 1, 1, 1, 0)
    calc = RadiomicsFirstOrder(
        image,
        mask,
        binWidth=12,
        voxelArrayShift=300,
    )
    calc._initCalculation(None)
    energy = float(np.asarray(calc.getEnergyFeatureValue()).reshape(-1)[0])
    total = float(np.asarray(calc.getTotalEnergyFeatureValue()).reshape(-1)[0])
    mean = float(np.asarray(calc.getMeanFeatureValue()).reshape(-1)[0])
    assert native.iloc[0]["original_firstorder_Mean"] == pytest.approx(mean, rel=0, abs=1e-12)
    assert native.iloc[0]["original_firstorder_Energy"] == pytest.approx(energy, rel=1e-12, abs=0)
    assert native.iloc[0]["original_firstorder_TotalEnergy"] == pytest.approx(
        total, rel=1e-12, abs=0
    )


@pytest.mark.unit
def test_native_matches_union_bin_calculator_path() -> None:
    """Native formulas match the previous union-bin calculator path."""
    assert cext_backend() == "native"
    image, sv_map, labels = _two_blob_case()
    native = extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features=ENABLED,
        settings=_settings("auto"),
        union_bin=True,
    )
    reference = extract_supervoxel_features_pyradiomics(
        image,
        sv_map,
        labels,
        enabled_features=ENABLED,
        settings=_settings(False),
        union_bin=True,
    )
    shift_invariant = [
        "original_firstorder_Mean",
        "original_glcm_Id",
        "original_glcm_JointEntropy",
        "original_glrlm_ShortRunEmphasis",
        "original_glszm_ZonePercentage",
        "original_firstorder_Energy",
    ]
    absolute = [
        "original_glcm_Autocorrelation",
        "original_glcm_JointAverage",
        "original_glrlm_HighGrayLevelRunEmphasis",
    ]
    for col in shift_invariant:
        np.testing.assert_allclose(
            native[col].to_numpy(dtype=np.float64),
            reference[col].to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
            err_msg=col,
        )
    for col in absolute:
        np.testing.assert_allclose(
            native[col].to_numpy(dtype=np.float64),
            reference[col].to_numpy(dtype=np.float64),
            rtol=1e-8,
            atol=1e-8,
            err_msg=col,
        )
    assert np.isfinite(native["original_glrlm_ShortRunEmphasis"]).all()


@pytest.mark.unit
def test_glcm_features_are_present() -> None:
    """The pxSuby bridge must not swallow the GLCM class."""
    image, sv_map, labels = _two_blob_case()
    frame = extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features={"glcm": ["Id", "Idm", "InverseVariance", "Autocorrelation", "MCC"]},
        settings=_settings(),
        union_bin=True,
    )
    for col in (
        "original_glcm_Id",
        "original_glcm_Idm",
        "original_glcm_InverseVariance",
        "original_glcm_Autocorrelation",
        "original_glcm_MCC",
    ):
        assert col in frame.columns
        assert np.isfinite(frame[col]).all()
