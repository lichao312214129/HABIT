# Smoke test for itk-elastix core (no HABIT imports).
#
# Default: load fixed/moving from a subject folder (same layout as habit io_utils:
#   <subject>/<modality>/<one volume file>).
# Fallback: ``--synthetic`` uses tiny in-memory 3-D blobs.
#
# Install: pip install itk-elastix SimpleITK
#
# Usage:
#   python demo_data/temp/debug_elastic.py
#   python demo_data/temp/debug_elastic.py --synthetic

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import itk
import numpy as np
import SimpleITK as sitk

# Repository demo elastix parameter file (affine example from elastix distribution).
_PAR0040: Path = Path(__file__).resolve().parents[2] / "demo_data" / "Par0040affine.txt"

# Real data: one subject under demo_data/temp/data/images/
_DEFAULT_SUBJECT_DIR: Path = Path(
    r"F:\work\habit_project\demo_data\preprocessed\resample_02\images\subj001"
)
_DEFAULT_FIXED_MODALITY: str = "delay2"
_DEFAULT_MOVING_MODALITY: str = "delay3"


def _first_volume_file_in_dir(modality_dir: Path) -> Path:
    """Pick the first non-hidden file under a modality directory (Deterministic order)."""
    if not modality_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {modality_dir}")
    candidates: list[Path] = sorted(
        p for p in modality_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if not candidates:
        raise FileNotFoundError(f"No volume files under {modality_dir}")
    if len(candidates) > 1:
        print(f"Warning: multiple files in {modality_dir}; using {candidates[0].name}")
    return candidates[0]


def _sitk_to_itk(image_sitk: sitk.Image):
    """Convert SimpleITK image to ITK image (spacing, origin, direction preserved).

    Mirrors the pattern in ``habit.core.preprocessing.registration.elastix_backend`` so
    elastix sees the same geometry as the rest of the project.
    """
    ndim: int = image_sitk.GetDimension()
    is_vector: bool = image_sitk.GetNumberOfComponentsPerPixel() > 1
    arr: np.ndarray = sitk.GetArrayFromImage(image_sitk)
    itk_image = itk.GetImageFromArray(arr, is_vector=is_vector)
    itk_image.SetSpacing(image_sitk.GetSpacing())
    itk_image.SetOrigin(image_sitk.GetOrigin())
    direction_2d: np.ndarray = np.reshape(
        np.asarray(image_sitk.GetDirection(), dtype=np.float64), (ndim, ndim)
    )
    itk_image.SetDirection(itk.GetMatrixFromArray(direction_2d))
    return itk_image


def _itk_to_sitk(itk_image: Any) -> sitk.Image:
    """Convert ITK image (elastix output) to SimpleITK for writing NIfTI with project-consistent metadata."""
    is_vector: bool = itk_image.GetNumberOfComponentsPerPixel() > 1
    arr: np.ndarray = itk.GetArrayFromImage(itk_image)
    result: sitk.Image = sitk.GetImageFromArray(arr, isVector=is_vector)
    result.SetOrigin(tuple(itk_image.GetOrigin()))
    result.SetSpacing(tuple(itk_image.GetSpacing()))
    result.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten())
    return result


def _make_synthetic_itk_pair() -> tuple[object, object, int]:
    """Return (fixed_itk, moving_itk, ndim) for a quick elastix smoke test without disk."""
    shape: tuple[int, int, int] = (24, 24, 12)
    z_idx, y_idx, x_idx = np.ogrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    cx, cy, cz = shape[2] / 2.0, shape[1] / 2.0, shape[0] / 2.0
    fixed_arr: np.ndarray = np.exp(
        -((x_idx - cx) ** 2 + (y_idx - cy) ** 2 + (z_idx - cz) ** 2) / 50.0
    ).astype(np.float32)
    moving_arr: np.ndarray = np.exp(
        -((x_idx - cx - 2.0) ** 2 + (y_idx - cy + 1.0) ** 2 + (z_idx - cz) ** 2) / 50.0
    ).astype(np.float32)
    return itk.GetImageFromArray(fixed_arr), itk.GetImageFromArray(moving_arr), 3


def _load_itk_pair_from_subject(subject_dir: Path, fixed_mod: str, moving_mod: str) -> tuple[object, object, int]:
    """Load two modalities from disk and return ITK images plus spatial dimension."""
    fixed_path: Path = _first_volume_file_in_dir(subject_dir / fixed_mod)
    moving_path: Path = _first_volume_file_in_dir(subject_dir / moving_mod)
    print(f"Fixed  ({fixed_mod}): {fixed_path}")
    print(f"Moving ({moving_mod}): {moving_path}")

    fixed_sitk: sitk.Image = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving_sitk: sitk.Image = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    ndim: int = int(fixed_sitk.GetDimension())

    return _sitk_to_itk(fixed_sitk), _sitk_to_itk(moving_sitk), ndim


def main() -> None:
    if not _PAR0040.is_file():
        raise FileNotFoundError(
            f"Parameter file not found: {_PAR0040}. Clone repo with demo_data or pass a .txt path."
        )

    use_synthetic: bool = "--synthetic" in sys.argv

    if use_synthetic:
        fixed_image, moving_image, ndim = _make_synthetic_itk_pair()
        print("Using synthetic 3-D volumes (--synthetic).")
    else:
        fixed_image, moving_image, ndim = _load_itk_pair_from_subject(
            _DEFAULT_SUBJECT_DIR, _DEFAULT_FIXED_MODALITY, _DEFAULT_MOVING_MODALITY
        )

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(str(_PAR0040))
    for i in range(parameter_object.GetNumberOfParameterMaps()):
        pmap = parameter_object.GetParameterMap(i)
        pmap["FixedImageDimension"] = [str(ndim)]
        pmap["MovingImageDimension"] = [str(ndim)]
        parameter_object.SetParameterMap(i, pmap)

    result_image, transform = itk.elastix_registration_method(
        fixed_image=fixed_image,
        moving_image=moving_image,
        parameter_object=parameter_object,
        log_to_console=True,
    )

    out_arr: np.ndarray = itk.GetArrayFromImage(result_image)
    print("itk-elastix core OK.")
    print("  output shape (z,y,x):", out_arr.shape)
    print("  transform type:", type(transform).__name__)

    # Save moving warped to fixed space (same grid as fixed image).
    result_sitk: sitk.Image = _itk_to_sitk(result_image)
    out_dir: Path = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    name_tag: str = "synthetic" if use_synthetic else f"{_DEFAULT_MOVING_MODALITY}_to_{_DEFAULT_FIXED_MODALITY}"
    out_path: Path = out_dir / f"elastix_registered_{name_tag}.nii.gz"
    sitk.WriteImage(result_sitk, str(out_path))
    print(f"Wrote registered image: {out_path}")


if __name__ == "__main__":
    main()
