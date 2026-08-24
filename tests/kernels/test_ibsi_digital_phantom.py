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
"""IBSI-1 Phase 1 digital phantom: HABIT / PyRadiomics vs official refs.

Phantom and 3-D-averaged reference values come from the Image Biomarker
Standardisation Initiative (theibsi/data_sets, theibsi/ibsi-doc). Texture
aggregation is IBSI ``3D, averaged`` (PyRadiomics / HABIT default), not
``3D, merged``.

Kurtosis is the documented PyRadiomics convention: Pearson kurtosis =
IBSI excess kurtosis + 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from habit.kernels.radiomics.native_batch import extract_native_supervoxel_features
from habit.kernels.radiomics.supervoxel_batch import DEFAULT_FEATURES_BY_CLASS

radiomics = pytest.importorskip("radiomics")
from radiomics.featureextractor import RadiomicsFeatureExtractor

PHANTOM_DIR = Path(__file__).resolve().parents[1] / "data" / "ibsi_1_digital_phantom"
REF_CSV = PHANTOM_DIR / "ibsi_phase1_3d_averaged.csv"

# IBSI published values use a few significant figures. Allow 2% or 0.02 abs.
IBSI_RTOL = 0.02
IBSI_ATOL = 0.02
# Native C + CPU formulas vs FeatureExtractor.execute() on this phantom.
NATIVE_PYRAD_RTOL = 1e-10
NATIVE_PYRAD_ATOL = 1e-12

SHAPE_FEATURES = (
    "MeshVolume",
    "VoxelVolume",
    "SurfaceArea",
    "SurfaceVolumeRatio",
    "Sphericity",
    "Maximum3DDiameter",
)

PHASE1_SETTINGS: Dict[str, object] = {
    "binWidth": 1.0,
    "voxelArrayShift": 0,
    "normalize": False,
    "force2D": False,
    "symmetricalGLCM": True,
    "distances": [1],
    "gldm_a": 0,
    "padDistance": 1,
    "supervoxel_union_bbox_crop": True,
    "additionalInfo": False,
}


def _load_refs() -> pd.DataFrame:
    frame = pd.read_csv(REF_CSV, comment="#")
    return frame.loc[frame["note"].fillna("").astype(str) != "skip"].reset_index(drop=True)


def _scalar(value: object) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else float("nan")


def _extract() -> Tuple[pd.Series, Mapping[str, float]]:
    image = sitk.ReadImage(str(PHANTOM_DIR / "phantom.nii.gz"))
    mask = sitk.ReadImage(str(PHANTOM_DIR / "mask.nii.gz"))
    sv_map = sitk.Cast(mask, sitk.sitkInt32)
    sv_map.CopyInformation(image)
    enabled = {name: list(feats) for name, feats in DEFAULT_FEATURES_BY_CLASS.items()}
    native = extract_native_supervoxel_features(
        image,
        sv_map,
        np.asarray([1], dtype=np.int64),
        enabled_features=enabled,
        settings=PHASE1_SETTINGS,
        union_bin=True,
    )
    # Pass a params dict so PyRadiomics does not log default settings as
    # ``%s`` + a dict (pytest log capture treats that dict as ``%(key)s``).
    extractor = RadiomicsFeatureExtractor(
        {
            "setting": {
                "binWidth": 1.0,
                "voxelArrayShift": 0,
                "normalize": False,
                "force2D": False,
                "symmetricalGLCM": True,
                "distances": [1],
                "gldm_a": 0,
                "padDistance": 1,
            }
        }
    )
    extractor.disableAllFeatures()
    for class_name, feats in enabled.items():
        extractor.enableFeaturesByName(**{class_name: feats})
    extractor.enableFeaturesByName(shape=list(SHAPE_FEATURES))
    for key, value in PHASE1_SETTINGS.items():
        if key in (
            "binWidth",
            "voxelArrayShift",
            "normalize",
            "force2D",
            "symmetricalGLCM",
            "distances",
            "gldm_a",
            "padDistance",
        ):
            extractor.settings[key] = value
    pyrad = extractor.execute(image, sv_map, label=1)
    numeric = {
        str(key): _scalar(value)
        for key, value in pyrad.items()
        if str(key).startswith("original_")
    }
    return native.iloc[0], numeric


@pytest.mark.unit
def test_official_phantom_files_present() -> None:
    """The IBSI NIfTI pair must be the official 5x4x4 / 74-voxel phantom."""
    assert (PHANTOM_DIR / "phantom.nii.gz").is_file()
    assert (PHANTOM_DIR / "mask.nii.gz").is_file()
    image = sitk.ReadImage(str(PHANTOM_DIR / "phantom.nii.gz"))
    mask = sitk.ReadImage(str(PHANTOM_DIR / "mask.nii.gz"))
    assert tuple(int(v) for v in image.GetSize()) == (5, 4, 4)
    assert tuple(float(v) for v in image.GetSpacing()) == (2.0, 2.0, 2.0)
    roi = int((sitk.GetArrayFromImage(mask) > 0).sum())
    assert roi == 74


@pytest.mark.unit
def test_ibsi_phase1_pyradiomics_matches_official_refs() -> None:
    """Traditional / each_habitat path: PyRadiomics vs IBSI 3-D averaged."""
    _native, pyrad = _extract()
    refs = _load_refs()
    mismatches = []
    for row in refs.itertuples(index=False):
        name = str(row.feature)
        ibsi = float(row.ibsi_value)
        if name.endswith("Kurtosis"):
            continue
        if name.startswith("original_shape_"):
            got = float(pyrad.get(name, float("nan")))
        else:
            got = float(pyrad.get(name, float("nan")))
        if not np.isfinite(got):
            mismatches.append(f"{name}: missing")
            continue
        if not np.isclose(got, ibsi, rtol=IBSI_RTOL, atol=IBSI_ATOL):
            mismatches.append(f"{name}: IBSI={ibsi} PyRad={got}")
    assert not mismatches, "PyRadiomics vs IBSI:\n" + "\n".join(mismatches)


@pytest.mark.unit
def test_ibsi_phase1_habit_native_matches_official_refs() -> None:
    """Native C+CPU formulas vs IBSI 3-D averaged (same 2% / 0.02 gate)."""
    native, _pyrad = _extract()
    refs = _load_refs()
    mismatches = []
    for row in refs.itertuples(index=False):
        name = str(row.feature)
        ibsi = float(row.ibsi_value)
        if name.endswith("Kurtosis") or name.startswith("original_shape_"):
            continue
        if name not in native.index:
            continue
        got = float(native[name])
        if not np.isclose(got, ibsi, rtol=IBSI_RTOL, atol=IBSI_ATOL):
            mismatches.append(f"{name}: IBSI={ibsi} HABIT={got}")
    assert not mismatches, "HABIT native vs IBSI:\n" + "\n".join(mismatches)


@pytest.mark.unit
def test_ibsi_phase1_habit_native_matches_pyradiomics() -> None:
    """Native path equals execute() on the official phantom, including GLRLM."""
    native, pyrad = _extract()
    mismatches = []
    for name in native.index:
        if not str(name).startswith("original_"):
            continue
        if name not in pyrad:
            continue
        habit = float(native[name])
        got = float(pyrad[name])
        if not np.isclose(habit, got, rtol=NATIVE_PYRAD_RTOL, atol=NATIVE_PYRAD_ATOL):
            mismatches.append(f"{name}: HABIT={habit} PyRad={got}")
    assert not mismatches, "HABIT native vs PyRadiomics:\n" + "\n".join(mismatches)


@pytest.mark.unit
def test_ibsi_phase1_kurtosis_is_pearson_not_excess() -> None:
    """PyRadiomics / HABIT Kurtosis = IBSI excess kurtosis + 3."""
    native, pyrad = _extract()
    refs = _load_refs()
    ibsi = float(refs.loc[refs["feature"] == "original_firstorder_Kurtosis", "ibsi_value"].iloc[0])
    habit = float(native["original_firstorder_Kurtosis"])
    pyrad_k = float(pyrad["original_firstorder_Kurtosis"])
    assert habit == pytest.approx(ibsi + 3.0, rel=0, abs=1e-3)
    assert pyrad_k == pytest.approx(ibsi + 3.0, rel=0, abs=1e-3)
    assert habit == pytest.approx(pyrad_k, rel=0, abs=1e-12)
