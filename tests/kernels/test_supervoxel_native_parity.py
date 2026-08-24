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
    np.testing.assert_allclose(
        native["original_glcm_MCC"].to_numpy(dtype=np.float64),
        reference["original_glcm_MCC"].to_numpy(dtype=np.float64),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.unit
def test_c_mcc_matches_pruned_eigvals() -> None:
    """C Jacobi on the similar Gram matrix matches eigvals of Q."""
    from habit.kernels.radiomics.cext import glcm_mcc
    from habit.kernels.radiomics.cpu_formulas import _mcc_pruned_eigvals

    rng = np.random.default_rng(0)
    n_labels, n_gray, n_angles = 6, 20, 7
    p_raw = rng.integers(0, 8, size=(n_labels, n_gray, n_gray, n_angles)).astype(
        np.float64
    )
    p_raw[:, 12:, :, :] = 0.0
    p_sym = p_raw + np.transpose(p_raw, (0, 2, 1, 3))
    sum_p = p_sym.sum(axis=(1, 2)).astype(np.float64)
    empty = sum_p == 0
    sum_p[empty] = np.nan
    p_norm = p_sym / sum_p[:, None, None, :]
    px = p_norm.sum(axis=2, keepdims=True)
    py = p_norm.sum(axis=1, keepdims=True)
    reference = _mcc_pruned_eigvals(
        p_norm, px, py, empty, n_gray, n_labels, n_angles, float(np.spacing(1))
    )
    computed = np.asarray(glcm_mcc(p_raw, 1), dtype=np.float64)
    np.testing.assert_allclose(computed, reference, rtol=1e-5, atol=1e-6)


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


def _split_island_case() -> tuple[sitk.Image, sitk.Image, np.ndarray]:
    """Same label on both sides of another label (union-crop re-entry)."""
    image = np.full((8, 8, 8), 0.0, dtype=np.float64)
    sv_map = np.zeros((8, 8, 8), dtype=np.int32)
    # Two 2x2 islands of label 1, separated by a 2x2 block of label 2.
    image[3:5, 3:5, 2] = 10.0
    image[3:5, 3:5, 3] = 80.0
    image[3:5, 3:5, 4] = 10.0
    sv_map[3:5, 3:5, 2] = 1
    sv_map[3:5, 3:5, 3] = 2
    sv_map[3:5, 3:5, 4] = 1
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_map = sitk.Cast(sitk.GetImageFromArray(sv_map), sitk.sitkUInt8)
    sitk_map.CopyInformation(sitk_image)
    return sitk_image, sitk_map, np.asarray([1, 2], dtype=np.int64)


@pytest.mark.unit
def test_glrlm_split_islands_match_execute() -> None:
    """Leaving a label must end the walk; re-entry is a new execute()-style run."""
    from habit.domain.habitat_features._radiomics import (
        build_pyradiomics_extractor,
        execute_radiomics,
    )

    image, sv_map, labels = _split_island_case()
    settings = {
        "binWidth": 25,
        "voxelArrayShift": 0,
        "normalize": False,
        "distances": [1],
        "force2D": False,
        "symmetricalGLCM": True,
        "gldm_a": 0,
        "use_supervoxel_cext": "auto",
        "supervoxel_union_bbox_crop": True,
        "padDistance": 1,
    }
    native = extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features={"glrlm": ["ShortRunEmphasis", "RunPercentage", "RunEntropy"]},
        settings=settings,
        union_bin=False,
    )
    extractor = build_pyradiomics_extractor(
        None,
        {
            "imageType": {"Original": {}},
            "featureClass": {"glrlm": ["ShortRunEmphasis", "RunPercentage", "RunEntropy"]},
            "setting": {"binWidth": 25, "voxelArrayShift": 0, "normalize": False},
        },
        owner="test_glrlm_islands",
    )
    for label in (1, 2):
        executed = execute_radiomics(
            extractor, image, sv_map, label=label, use_torch_radiomics=False
        )
        row = native.loc[native["supervoxel_id"] == label].iloc[0]
        for name in (
            "original_glrlm_ShortRunEmphasis",
            "original_glrlm_RunPercentage",
            "original_glrlm_RunEntropy",
        ):
            assert float(row[name]) == pytest.approx(
                float(executed[name]), rel=1e-8, abs=1e-8
            )


@pytest.mark.unit
def test_default_union_bin_false_matches_execute_gray_level_features() -> None:
    """Default per-label bin must match execute() on i-dependent features."""
    from habit.domain.habitat_features._radiomics import (
        build_pyradiomics_extractor,
        execute_radiomics,
    )

    image, sv_map, labels = _two_blob_case()
    native = extract_native_supervoxel_features(
        image,
        sv_map,
        labels,
        enabled_features={
            "glcm": ["Autocorrelation", "JointAverage"],
            "glrlm": ["HighGrayLevelRunEmphasis"],
        },
        settings=_settings(),
    )
    extractor = build_pyradiomics_extractor(
        None,
        {
            "imageType": {"Original": {}},
            "featureClass": {
                "glcm": ["Autocorrelation", "JointAverage"],
                "glrlm": ["HighGrayLevelRunEmphasis"],
            },
            "setting": {"binWidth": 12, "voxelArrayShift": 300, "normalize": False},
        },
        owner="test_default_union_bin",
    )
    for label in labels:
        executed = execute_radiomics(
            extractor, image, sv_map, label=int(label), use_torch_radiomics=False
        )
        row = native.loc[native["supervoxel_id"] == int(label)].iloc[0]
        for name in (
            "original_glcm_Autocorrelation",
            "original_glcm_JointAverage",
            "original_glrlm_HighGrayLevelRunEmphasis",
        ):
            assert float(row[name]) == pytest.approx(
                float(executed[name]), rel=1e-8, abs=1e-8
            )
