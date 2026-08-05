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
"""
Deterministic synthetic demo-data generator for HABIT coverage tests.

The real ``demo_data`` dataset is gitignored and unavailable in CI, so the
cloud coverage suite builds its own look-alike tree:

.. code-block:: text

    <root>/images/subj001..subj004/{delay2,delay3,delay5}/image.nrrd
    <root>/masks/subj001..subj004/delay2/mask.nrrd
    <root>/ml_data/clinical.csv
    <root>/ml_data/radiomics_features.csv
    <root>/ml_data/radiomics_features_retest.csv
    <root>/ml_data/icc_measurements_test.csv
    <root>/ml_data/icc_measurements_retest.csv

Every image is a 64x64x16 float32 volume with 1x1x2 mm spacing. A smooth
random background fills the field of view, and inside the ellipsoid tumor
mask 2-3 compact subregions carry distinct mean intensities, so habitat
clustering has real structure to recover. The same planted means drive all
three modalities through fixed linear combinations, keeping the modalities
correlated-but-different exactly like multi-phase DCE data.

Everything is deterministic given the seed: per-subject generators derive
from ``seed + subject_index`` and the table generators use their own
independent streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

#: Subject ids in canonical (sorted) order.
SUBJECT_IDS: Tuple[str, ...] = ("subj001", "subj002", "subj003", "subj004")

#: Modality keys mirroring the demo DCE phases.
MODALITIES: Tuple[str, ...] = ("delay2", "delay3", "delay5")

#: Mask folder key used by the conventional directory layout (roi key).
ROI_KEY: str = "delay2"

#: Volume shape in SimpleITK (x, y, z) order.
IMAGE_SHAPE_XYZ: Tuple[int, int, int] = (64, 64, 16)

#: Voxel spacing in millimetres (x, y, z).
IMAGE_SPACING_MM: Tuple[float, float, float] = (1.0, 1.0, 2.0)

#: Number of rows in the synthetic radiomics feature table (ML-ready).
ML_TABLE_ROWS: int = 60

#: Number of numeric features in the synthetic radiomics feature table.
ML_TABLE_FEATURES: int = 20

#: Name of the one radiomics feature correlated with the outcome label.
ML_SIGNAL_FEATURE: str = "feature_03"


@dataclass(frozen=True)
class SyntheticTree:
    """
    Paths and ground truth of one generated synthetic tree.

    Attributes:
        root: Dataset root holding ``images/`` and ``masks/``.
        subjects: Subject ids that were generated.
        modalities: Modality keys generated per subject.
        roi: Mask folder key under ``masks/<subject>/``.
        clinical_csv: Clinical table for the four imaging subjects.
        radiomics_csv: ML-ready feature table (60 rows, label column).
        radiomics_retest_csv: Noisy remeasurement of ``radiomics_csv``.
        icc_test_csv: Paired measurement table (test) for ICC analysis.
        icc_retest_csv: Paired measurement table (retest) for ICC analysis.
        subregion_means: ``subject -> (mean_a, mean_b, mean_c)`` planted
            delay2 subregion means; subjects with two subregions repeat the
            second value for the third slot.
    """

    root: Path
    subjects: Tuple[str, ...]
    modalities: Tuple[str, ...]
    roi: str
    clinical_csv: Path
    radiomics_csv: Path
    radiomics_retest_csv: Path
    icc_test_csv: Path
    icc_retest_csv: Path
    subregion_means: Dict[str, Tuple[float, float, float]]


def _ellipsoid(
    shape_zyx: Tuple[int, int, int],
    center_xyz: Tuple[float, float, float],
    radii_xyz: Tuple[float, float, float],
) -> np.ndarray:
    """
    Build a boolean ellipsoid mask.

    Args:
        shape_zyx: Array shape in NumPy (z, y, x) order.
        center_xyz: Ellipsoid centre in (x, y, z) voxel coordinates.
        radii_xyz: Ellipsoid radii in (x, y, z) voxels.

    Returns:
        Boolean array of ``shape_zyx``; True inside the ellipsoid.
    """
    grids_zyx = np.ogrid[tuple(slice(0, n) for n in shape_zyx)]
    # Map the (z, y, x) index grids onto the (x, y, z) parameter order.
    x_grid, y_grid, z_grid = grids_zyx[2], grids_zyx[1], grids_zyx[0]
    cx, cy, cz = center_xyz
    rx, ry, rz = radii_xyz
    normalised = (
        ((x_grid - cx) / rx) ** 2
        + ((y_grid - cy) / ry) ** 2
        + ((z_grid - cz) / rz) ** 2
    )
    return normalised <= 1.0


def _plant_subregions(
    rng: np.random.Generator,
    tumor_mask: np.ndarray,
) -> List[np.ndarray]:
    """
    Plant 2-3 compact subregion masks inside the tumor ellipsoid.

    Args:
        rng: Subject-local random generator (jitter makes every subject
            slightly different while staying deterministic).
        tumor_mask: Boolean tumor mask in (z, y, x) order.

    Returns:
        List of boolean subregion masks, each a subset of ``tumor_mask``.
    """
    shape_zyx = tumor_mask.shape
    jitter = lambda scale: float(rng.uniform(-scale, scale))  # noqa: E731
    # Three candidate compact ellipsoids at fixed offsets from the tumour
    # centre; radii stay well inside the 13x13x4 tumour envelope.
    candidates = [
        ((26.0 + jitter(1.5), 27.0 + jitter(1.5), 8.0 + jitter(1.0)), (5.0, 5.0, 2.5)),
        ((38.0 + jitter(1.5), 36.0 + jitter(1.5), 8.0 + jitter(1.0)), (5.0, 4.5, 2.5)),
        ((31.0 + jitter(1.5), 40.0 + jitter(1.5), 7.0 + jitter(1.0)), (4.5, 4.0, 2.0)),
    ]
    n_regions = 3 if rng.random() > 0.25 else 2
    subregions: List[np.ndarray] = []
    for center_xyz, radii_xyz in candidates[:n_regions]:
        region = _ellipsoid(shape_zyx, center_xyz, radii_xyz) & tumor_mask
        subregions.append(region)
    return subregions


def _modality_mean(base_mean: float, modality_index: int) -> float:
    """
    Map a delay2 base intensity onto another modality's mean.

    The fixed linear combinations keep modalities correlated yet distinct,
    mimicking wash-in / wash-out of DCE phases.

    Args:
        base_mean: Planted mean intensity for delay2.
        modality_index: 0 for delay2, 1 for delay3, 2 for delay5.

    Returns:
        The modality-specific mean intensity.
    """
    scales = (1.00, 0.85, 0.70)
    offsets = (0.0, 15.0, 30.0)
    return base_mean * scales[modality_index] + offsets[modality_index]


def _make_subject_volumes(
    seed: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Tuple[float, float, float]]:
    """
    Build one subject's three modality images plus the tumor mask.

    Args:
        seed: Per-subject seed (``base seed + subject index``).

    Returns:
        ``(images, mask, region_means)`` where ``images`` maps modality key
        to a float32 (z, y, x) array, ``mask`` is a uint8 (z, y, x) array,
        and ``region_means`` holds the planted delay2 means of the three
        subregion slots (the third repeats the second when only two
        subregions were planted).
    """
    rng = np.random.default_rng(seed)
    shape_zyx = (IMAGE_SHAPE_XYZ[2], IMAGE_SHAPE_XYZ[1], IMAGE_SHAPE_XYZ[0])

    # Tumour ellipsoid with a small deterministic per-subject jitter.
    center_xyz = (
        32.0 + float(rng.uniform(-2.0, 2.0)),
        32.0 + float(rng.uniform(-2.0, 2.0)),
        8.0 + float(rng.uniform(-1.0, 1.0)),
    )
    tumor_mask = _ellipsoid(shape_zyx, center_xyz, (13.0, 13.0, 4.0))
    subregions = _plant_subregions(rng, tumor_mask)

    # Distinct planted means per subregion (delay2 reference).
    base_means = [120.0 + float(rng.uniform(-8, 8)),
                  175.0 + float(rng.uniform(-8, 8)),
                  225.0 + float(rng.uniform(-8, 8))]
    tumour_base = 80.0 + float(rng.uniform(-5, 5))

    images: Dict[str, np.ndarray] = {}
    for modality_index, modality in enumerate(MODALITIES):
        # Smooth random background: low-pass filtered noise, mean ~45.
        background = gaussian_filter(
            rng.normal(0.0, 1.0, size=shape_zyx), sigma=3.0
        )
        background = background * 12.0 + 45.0
        volume = background.astype(np.float32)

        # Tumour interior: base level plus per-subregion distinct means,
        # with modest voxel noise so clustering sees compact clusters.
        volume[tumor_mask] = _modality_mean(tumour_base, modality_index)
        for region, base_mean in zip(subregions, base_means):
            volume[region] = _modality_mean(base_mean, modality_index)
        volume[tumor_mask] += rng.normal(
            0.0, 6.0, size=int(tumor_mask.sum())
        ).astype(np.float32)
        images[modality] = volume

    mask = tumor_mask.astype(np.uint8)
    means = tuple(base_means[i] if i < len(subregions) else base_means[-1]
                  for i in range(3))
    return images, mask, means  # type: ignore[return-value]


def _write_nrrd(path: Path, array: np.ndarray) -> None:
    """
    Write one NumPy array as NRRD with the standard spacing, creating parents.

    Args:
        path: Destination ``.nrrd`` file path.
        array: Voxel data in (z, y, x) order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(IMAGE_SPACING_MM)
    sitk.WriteImage(image, str(path))


def _write_image_tree(root: Path, seed: int) -> Dict[str, Tuple[float, float, float]]:
    """
    Write ``images/`` and ``masks/`` for all subjects.

    Args:
        root: Dataset root to populate.
        seed: Base seed; subject ``i`` uses ``seed + i``.

    Returns:
        Mapping of subject id to its planted delay2 subregion means.
    """
    subregion_means: Dict[str, Tuple[float, float, float]] = {}
    for index, subject_id in enumerate(SUBJECT_IDS):
        images, mask, means = _make_subject_volumes(seed + index)
        for modality in MODALITIES:
            _write_nrrd(
                root / "images" / subject_id / modality / "image.nrrd",
                images[modality],
            )
        _write_nrrd(root / "masks" / subject_id / ROI_KEY / "mask.nrrd", mask)
        subregion_means[subject_id] = means
    return subregion_means


def _write_clinical_csv(
    path: Path,
    subregion_means: Dict[str, Tuple[float, float, float]],
    seed: int,
) -> None:
    """
    Write the clinical table for the four imaging subjects.

    The binary outcome is correlated with the first subregion's planted
    delay2 mean intensity (subjects above the cohort median are labelled 1);
    two noise covariates carry no signal.

    Args:
        path: Destination CSV path.
        subregion_means: Planted means keyed by subject id.
        seed: Seed for the noise covariates.
    """
    rng = np.random.default_rng(seed)
    subjects = sorted(subregion_means)
    signal = np.array([subregion_means[s][0] for s in subjects])
    outcome = (signal >= np.median(signal)).astype(int)
    frame = pd.DataFrame(
        {
            "subject_id": subjects,
            "outcome": outcome,
            "age": np.round(55.0 + rng.normal(0.0, 10.0, size=len(subjects)), 1),
            "noise_score": np.round(rng.uniform(0.0, 1.0, size=len(subjects)), 4),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _make_radiomics_frame(seed: int) -> pd.DataFrame:
    """
    Build the ML-ready synthetic radiomics feature table.

    ``ML_TABLE_ROWS`` subjects x ``ML_TABLE_FEATURES`` numeric features;
    ``ML_SIGNAL_FEATURE`` is correlated with the binary ``label`` column
    (label itself derives from a balanced latent split, so every
    stratified CV fold sees both classes).

    Args:
        seed: Seed for the table generator.

    Returns:
        DataFrame with ``subject_id``, ``feature_XX`` columns and ``label``.
    """
    rng = np.random.default_rng(seed)
    n_rows = ML_TABLE_ROWS
    latent = rng.normal(0.0, 1.0, size=n_rows)
    label = (latent >= np.median(latent)).astype(int)
    data: Dict[str, np.ndarray] = {}
    for feature_index in range(ML_TABLE_FEATURES):
        name = f"feature_{feature_index:02d}"
        if name == ML_SIGNAL_FEATURE:
            # Signal feature: class means separated by 3.0 plus unit noise.
            # (2.0 gave marginal CV AUC ~0.69 on 60 rows, flaky vs the 0.7
            # assertion threshold; 3.0 leaves comfortable headroom.)
            data[name] = 3.0 * label + rng.normal(0.0, 1.0, size=n_rows)
        else:
            data[name] = rng.normal(0.0, 1.0, size=n_rows)
    frame = pd.DataFrame(data)
    frame.insert(0, "subject_id", [f"subj{i + 1:03d}" for i in range(n_rows)])
    frame["label"] = label
    return frame


def _write_radiomics_csvs(test_path: Path, retest_path: Path, seed: int) -> None:
    """
    Write the radiomics feature table and its noisy retest twin.

    Args:
        test_path: Destination for the base feature table.
        retest_path: Destination for the retest table (same schema, small
            additive noise so ICC stays high but below 1).
        seed: Seed for both the base table and the retest noise.
    """
    frame = _make_radiomics_frame(seed)
    rng = np.random.default_rng(seed + 1)
    feature_cols = [c for c in frame.columns if c.startswith("feature_")]
    retest = frame.copy()
    retest[feature_cols] = frame[feature_cols] + rng.normal(
        0.0, 0.1, size=frame[feature_cols].shape
    )
    test_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(test_path, index=False)
    retest.to_csv(retest_path, index=False)


def _write_icc_csvs(test_path: Path, retest_path: Path, seed: int) -> None:
    """
    Write paired measurement tables for ICC analysis (no label column).

    Eight numeric measurements on 30 subjects; the retest table adds small
    Gaussian noise, giving high-but-not-perfect agreement.

    Args:
        test_path: Destination for the test measurements.
        retest_path: Destination for the retest measurements.
        seed: Seed for both tables.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_features = 30, 8
    subjects = [f"subj{i + 1:03d}" for i in range(n_rows)]
    columns = [f"measure_{i:02d}" for i in range(n_features)]
    base = rng.normal(0.0, 1.0, size=(n_rows, n_features))
    noise = rng.normal(0.0, 0.05, size=(n_rows, n_features))
    test_frame = pd.DataFrame(base, columns=columns)
    retest_frame = pd.DataFrame(base + noise, columns=columns)
    test_frame.insert(0, "subject_id", subjects)
    retest_frame.insert(0, "subject_id", subjects)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_frame.to_csv(test_path, index=False)
    retest_frame.to_csv(retest_path, index=False)


def make_synthetic_tree(root: Path, seed: int = 42) -> SyntheticTree:
    """
    Generate the complete synthetic demo-data tree under ``root``.

    Args:
        root: Dataset root; created when absent. Holds ``images/``,
            ``masks/`` and ``ml_data/`` afterwards.
        seed: Base seed for every generator (default 42).

    Returns:
        A :class:`SyntheticTree` describing every written artefact.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    subregion_means = _write_image_tree(root, seed)
    ml_dir = root / "ml_data"
    clinical_csv = ml_dir / "clinical.csv"
    radiomics_csv = ml_dir / "radiomics_features.csv"
    radiomics_retest_csv = ml_dir / "radiomics_features_retest.csv"
    icc_test_csv = ml_dir / "icc_measurements_test.csv"
    icc_retest_csv = ml_dir / "icc_measurements_retest.csv"
    _write_clinical_csv(clinical_csv, subregion_means, seed + 100)
    _write_radiomics_csvs(radiomics_csv, radiomics_retest_csv, seed + 200)
    _write_icc_csvs(icc_test_csv, icc_retest_csv, seed + 300)
    return SyntheticTree(
        root=root,
        subjects=SUBJECT_IDS,
        modalities=MODALITIES,
        roi=ROI_KEY,
        clinical_csv=clinical_csv,
        radiomics_csv=radiomics_csv,
        radiomics_retest_csv=radiomics_retest_csv,
        icc_test_csv=icc_test_csv,
        icc_retest_csv=icc_retest_csv,
        subregion_means=subregion_means,
    )


def make_dicom_series(series_dir: Path, n_slices: int = 5, seed: int = 42) -> Path:
    """
    Synthesise a tiny valid DICOM CT series with pydicom.

    Each slice is a 16x16 uint16 image carrying a deterministic gradient
    plus seeded noise; the usual patient/study/series UIDs and geometry
    tags are populated so DICOM readers accept the files.

    Args:
        series_dir: Directory receiving ``slice_001.dcm`` ... files.
        n_slices: Number of slices to write.
        seed: Noise seed.

    Returns:
        The ``series_dir`` path.
    """
    pydicom = pytest_import_pydicom()
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    rng = np.random.default_rng(seed)
    series_dir = Path(series_dir)
    series_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_reference_uid = generate_uid()
    for slice_index in range(n_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        dataset = Dataset()
        dataset.file_meta = file_meta
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = "CT"
        dataset.PatientName = "SYNTHETIC^COVERAGE"
        dataset.PatientID = "subj_dicom_001"
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.FrameOfReferenceUID = frame_of_reference_uid
        dataset.StudyDescription = "SyntheticCoverageStudy"
        dataset.SeriesDescription = "SyntheticCT"
        dataset.StudyDate = "20260101"
        dataset.SeriesNumber = 1
        dataset.InstanceNumber = slice_index + 1
        dataset.ImagePositionPatient = [0.0, 0.0, float(slice_index) * 2.5]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 2.5
        dataset.Rows = 16
        dataset.Columns = 16
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        gradient = np.tile(np.arange(16, dtype=np.uint16) * 16, (16, 1))
        noise = rng.integers(0, 32, size=(16, 16)).astype(np.uint16)
        dataset.PixelData = (gradient + noise).tobytes()

        # enforce_file_format writes the 128-byte preamble, the 'DICM'
        # prefix and the file-meta group so strict readers accept the file.
        pydicom.dcmwrite(
            str(series_dir / f"slice_{slice_index + 1:03d}.dcm"),
            dataset,
            enforce_file_format=True,
        )
    return series_dir


def pytest_import_pydicom():
    """
    Import pydicom or skip the calling test with a documented reason.

    Returns:
        The imported ``pydicom`` module.
    """
    import importlib

    import pytest

    try:
        return importlib.import_module("pydicom")
    except ImportError:
        pytest.skip("pydicom is not installed; DICOM synthesis unavailable")
