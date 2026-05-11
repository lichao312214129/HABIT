# Debug script: exercise habit ElastixRegistrationBackend (elastix / transformix CLI) on disk images.
#
# Edit the CONFIG block below, then run from repository root:
#   python demo_data/temp/debug_elastix_habit.py
#
# Set RUN_RAW_ITK = True alongside RUN_HABIT_BACKEND to compare with bare itk.elastix_registration_method.
#
# Requires: SimpleITK; elastix + transformix on PATH (or set elastix_path in code). Optional: itk-elastix for RUN_RAW_ITK.

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import SimpleITK as sitk

# --- repository root on sys.path -------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_SUBJECT_DIR: Path = _REPO_ROOT / "demo_data" / "preprocessed" / "resample_02" / "images" / "subj001"
_DEFAULT_PAR: Path = _REPO_ROOT / "demo_data" / "Par0040affine.txt"

# ---------------------------------------------------------------------------
# CONFIG — change these values only (no command-line arguments).
#
# SUBJECT_DIR: folder with one subfolder per modality (e.g. delay2/, delay3/).
# USE_SYNTHETIC_MASKS: True => all-ones uint8 masks; False => no masks (like debug_elastic.py).
# ---------------------------------------------------------------------------
SUBJECT_DIR: Path = _DEFAULT_SUBJECT_DIR
FIXED_MODALITY: str = "delay2"
MOVING_MODALITY: str = "delay3"
PAR_PATH: Path = _DEFAULT_PAR

# If True, pass all-ones uint8 masks on fixed/moving grids (like use_mask in pipeline).
# If False, no masks are passed (closest to demo_data/temp/debug_elastic.py).
USE_SYNTHETIC_MASKS: bool = True

# Elastix console logging for the raw-ITK code path only (habit backend keeps log_to_console=False).
LOG_TO_CONSOLE_RAW_ITK: bool = False

# Default: only habit ElastixRegistrationBackend (set RUN_RAW_ITK True to compare with bare itk).
RUN_RAW_ITK: bool = False
RUN_HABIT_BACKEND: bool = True


def _first_volume_file_in_dir(modality_dir: Path) -> Path:
    """Return the first lexicographic non-hidden file under modality_dir."""
    if not modality_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {modality_dir}")
    candidates: list[Path] = sorted(
        p for p in modality_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if not candidates:
        raise FileNotFoundError(f"No volume files under {modality_dir}")
    return candidates[0]


def load_sitk_pair(
    subject_dir: Path,
    fixed_mod: str,
    moving_mod: str,
) -> Tuple[sitk.Image, sitk.Image]:
    """Load fixed/moving volumes as sitkFloat32 (same dtype habit uses before elastix)."""
    fixed_path: Path = _first_volume_file_in_dir(subject_dir / fixed_mod)
    moving_path: Path = _first_volume_file_in_dir(subject_dir / moving_mod)
    fixed: sitk.Image = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving: sitk.Image = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    return fixed, moving


def make_all_ones_mask_fixed(reference: sitk.Image) -> sitk.Image:
    """All-ones uint8 mask with geometry copied from reference (matches pipeline use_mask).

    Args:
        reference (sitk.Image): Grid template (fixed or moving image).

    Returns:
        sitk.Image: Scalar uint8 mask with identical meta-data, values 1 everywhere.
    """
    mask_arr: np.ndarray = np.ones(tuple(reversed(reference.GetSize())), dtype=np.uint8)
    m: sitk.Image = sitk.GetImageFromArray(mask_arr)
    m.SetOrigin(reference.GetOrigin())
    m.SetSpacing(reference.GetSpacing())
    m.SetDirection(reference.GetDirection())
    return sitk.Cast(m, sitk.sitkUInt8)


def _sitk_to_itk(itk: Any, image_sitk: sitk.Image) -> Any:
    """Same conversion strategy as legacy itk-based habit backend (for raw itk path only)."""
    ndim: int = int(image_sitk.GetDimension())
    is_vector: bool = image_sitk.GetNumberOfComponentsPerPixel() > 1
    arr: np.ndarray = sitk.GetArrayFromImage(image_sitk)
    itk_image = itk.GetImageFromArray(arr, is_vector=is_vector)
    itk_image.SetSpacing(image_sitk.GetSpacing())
    itk_image.SetOrigin(image_sitk.GetOrigin())
    direction_2d: np.ndarray = np.reshape(
        np.asarray(image_sitk.GetDirection(), dtype=np.float64),
        (ndim, ndim),
    )
    itk_image.SetDirection(itk.GetMatrixFromArray(direction_2d))
    return itk_image


def run_raw_elastix(
    fixed_sitk: sitk.Image,
    moving_sitk: sitk.Image,
    par_path: Path,
    fixed_mask: Optional[sitk.Image],
    moving_mask: Optional[sitk.Image],
    log_to_console: bool,
) -> Tuple[Any, Any, float]:
    """Call itk.elastix_registration_method directly (no HABIT)."""
    import itk  # noqa: WPS433 — local import to mirror lazy import timing

    t0: float = time.perf_counter()
    ndim: int = int(fixed_sitk.GetDimension())
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(str(par_path))
    for i in range(parameter_object.GetNumberOfParameterMaps()):
        pmap = parameter_object.GetParameterMap(i)
        pmap["FixedImageDimension"] = [str(ndim)]
        pmap["MovingImageDimension"] = [str(ndim)]
        parameter_object.SetParameterMap(i, pmap)

    fixed_itk = _sitk_to_itk(itk, sitk.Cast(fixed_sitk, sitk.sitkFloat32))
    moving_itk = _sitk_to_itk(itk, sitk.Cast(moving_sitk, sitk.sitkFloat32))
    kwargs: dict = dict(
        fixed_image=fixed_itk,
        moving_image=moving_itk,
        parameter_object=parameter_object,
        log_to_console=log_to_console,
    )
    if fixed_mask is not None:
        kwargs["fixed_mask"] = _sitk_to_itk(itk, sitk.Cast(fixed_mask, sitk.sitkUInt8))
    if moving_mask is not None:
        kwargs["moving_mask"] = _sitk_to_itk(itk, sitk.Cast(moving_mask, sitk.sitkUInt8))

    result_itk, transform = itk.elastix_registration_method(**kwargs)
    elapsed: float = time.perf_counter() - t0
    return result_itk, transform, elapsed


def run_habit_elastix_backend(
    fixed_sitk: sitk.Image,
    moving_sitk: sitk.Image,
    par_path: Path,
    fixed_mask: Optional[sitk.Image],
    moving_mask: Optional[sitk.Image],
) -> Tuple[sitk.Image, Any, float]:
    """Call habit.preprocessing ElastixRegistrationBackend.register_image."""
    from habit.core.preprocessing.registration.elastix_backend import ElastixRegistrationBackend

    t0: float = time.perf_counter()
    # BaseRegistrationBackend requires these six kwargs (pipeline passes them via RegistrationPreprocessor).
    # Elastix CLI ignores transform/metric strings here; they are placeholders for direct backend construction.
    backend = ElastixRegistrationBackend(
        fixed_image_key="debug_fixed",
        type_of_transform="Rigid",
        metric="",
        optimizer=None,
        reg_params={},
        sitk_reg_params={},
        elastix_parameter_files=[str(par_path)],
    )
    registered, transform = backend.register_image(
        fixed_image_sitk=fixed_sitk,
        moving_image_sitk=moving_sitk,
        fixed_mask_sitk=fixed_mask,
        moving_mask_sitk=moving_mask,
    )
    elapsed: float = time.perf_counter() - t0
    return registered, transform, elapsed


def _print_banner(title: str) -> None:
    print(f"\n=== {title} ===")


def _run_and_report(
    label: str,
    runner: Callable[[], Tuple[Any, ...]],
) -> None:
    """Execute runner and print success or re-raise with context."""
    _print_banner(label)
    try:
        out = runner()
        print("OK", out[-1] if out else "")
    except BaseException as exc:  # noqa: WPS424 — debug script: show full cause chain
        print(f"FAILED: {type(exc).__name__}: {exc}")
        raise


def main() -> None:
    if not PAR_PATH.is_file():
        raise FileNotFoundError(f"Parameter file not found: {PAR_PATH}")

    fixed_sitk, moving_sitk = load_sitk_pair(SUBJECT_DIR, FIXED_MODALITY, MOVING_MODALITY)
    print("Loaded:", SUBJECT_DIR)
    print(
        "  fixed :",
        FIXED_MODALITY,
        "size",
        fixed_sitk.GetSize(),
        "spacing",
        fixed_sitk.GetSpacing(),
    )
    print(
        "  moving:",
        MOVING_MODALITY,
        "size",
        moving_sitk.GetSize(),
        "spacing",
        moving_sitk.GetSpacing(),
    )

    fixed_mask: Optional[sitk.Image] = None
    moving_mask: Optional[sitk.Image] = None
    if USE_SYNTHETIC_MASKS:
        fixed_mask = make_all_ones_mask_fixed(fixed_sitk)
        moving_mask = make_all_ones_mask_fixed(moving_sitk)
        print("Using synthetic all-ones uint8 masks on fixed/moving grids.")

    if RUN_RAW_ITK:

        def _raw() -> Tuple[Any, Any, float]:
            _, _, elapsed = run_raw_elastix(
                fixed_sitk,
                moving_sitk,
                PAR_PATH,
                fixed_mask,
                moving_mask,
                log_to_console=LOG_TO_CONSOLE_RAW_ITK,
            )
            return (None, None, f"raw itk elapsed {elapsed:.3f}s")

        _run_and_report("raw itk.elastix_registration_method", _raw)

    if RUN_HABIT_BACKEND:

        def _habit() -> Tuple[sitk.Image, Any, float]:
            reg, _tfm, elapsed = run_habit_elastix_backend(
                fixed_sitk,
                moving_sitk,
                PAR_PATH,
                fixed_mask,
                moving_mask,
            )
            return (
                reg,
                _tfm,
                f"HABIT ElastixRegistrationBackend elapsed {elapsed:.3f}s size_out={reg.GetSize()}",
            )

        _run_and_report("habit ElastixRegistrationBackend.register_image", _habit)

    print("\nDone.")


if __name__ == "__main__":
    main()
