# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import SimpleITK as sitk

from habit.core.preprocessing.elastix_cli_runner import (
    ElastixCliRunner,
    merge_elastix_parameter_file_text,
)
from habit.core.preprocessing.registration.base import BaseRegistrationBackend
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)


def _transform_parameters_txt_sort_key(path: Path) -> Tuple[int, str]:
    """Sort ``TransformParameters.<n>.txt`` by numeric stage index, not lexicographically.

    Args:
        path: A transform-parameter file in the elastix output directory.

    Returns:
        Tuple[int, str]: Numeric suffix (``-1`` if unknown) and name for tie-breaking.
    """
    match = re.match(r"(?i)TransformParameters\.(\d+)\.txt$", path.name)
    if match:
        return (int(match.group(1)), path.name.lower())
    return (-1, path.name.lower())


def _find_elastix_result_image(out_dir: Path) -> Path:
    """Return the primary registered intensity image produced in an elastix ``-out`` directory.

    Elastix often writes **MetaImage** pairs ``result.N.mhd`` + ``result.N.raw``. SimpleITK must
    open the ``.mhd`` header; it cannot infer dimensions from a bare ``.raw`` file. When multiple
    numbered outputs exist (multi-resolution), the highest index is usually the final stage.

    Args:
        out_dir: Elastix output directory passed to ``-out``.

    Returns:
        Path: Absolute path to the result image to load with SimpleITK.

    Raises:
        RuntimeError: If no readable ``result.*`` image is present.
    """
    candidates: List[Path] = []
    for path in out_dir.iterdir():
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if not name_lower.startswith("result."):
            continue
        if "transformparameters" in name_lower:
            continue
        if "iteration" in name_lower and "info" in name_lower:
            continue
        # MetaImage raw voxel file — never pass this alone to ImageFileReader; use the .mhd.
        if name_lower.endswith(".raw"):
            continue
        candidates.append(path)

    if not candidates:
        raise RuntimeError(
            f"elastix produced no readable result.* image under {out_dir} "
            "(expected .mhd / .mha / .nii / .nii.gz, not lone .raw). "
            "Check parameter files and elastix stderr."
        )

    def format_tier(path: Path) -> int:
        """Larger = prefer (NIfTI over MetaImage header, etc.)."""
        n = path.name.lower()
        if n.endswith(".nii.gz"):
            return 50
        if n.endswith(".nii"):
            return 40
        if n.endswith(".mha"):
            return 30
        if n.endswith(".mhd"):
            return 20
        if n.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            return 10
        return 5

    def resolution_index(path: Path) -> int:
        """E.g. result.1.mhd -> 1; result.nii -> 0 (unnumbered last-resort sort key)."""
        match = re.match(r"(?i)result\.(\d+)\.", path.name)
        return int(match.group(1)) if match else 0

    candidates.sort(
        key=lambda p: (format_tier(p), resolution_index(p), p.name.lower())
    )
    return candidates[-1].resolve()


def _find_final_transform_parameters(out_dir: Path) -> Path:
    """Pick the last ``TransformParameters.<n>.txt`` from a completed elastix run.

    Sequential ``-p`` stages emit multiple transform parameter files; transformix / mask
    warping must use the final map in the chain (highest numeric suffix).

    Args:
        out_dir: Elastix ``-out`` directory.

    Returns:
        Path: Absolute path to the final transform-parameter file.

    Raises:
        RuntimeError: If no transform-parameter file exists.
    """
    numbered = list(out_dir.glob("TransformParameters.*.txt"))

    def suffix_index(path: Path) -> int:
        # TransformParameters.0.txt -> stem "TransformParameters.0"
        part = path.stem.split(".")[-1]
        return int(part)

    if numbered:
        numbered.sort(key=suffix_index)
        return numbered[-1].resolve()

    legacy = out_dir / "TransformParameters.txt"
    if legacy.is_file():
        return legacy.resolve()

    raise RuntimeError(
        f"No TransformParameters*.txt found under {out_dir}; elastix may have failed silently."
    )


def _ensure_nearest_neighbor_resample(tp_text: str) -> str:
    """Force label-friendly interpolation in a transform-parameter file for transformix.

    For binary / label masks, elastix / transformix should resample with nearest neighbor
    so boundaries do not blur. See elastix parameter ``ResampleInterpolator``.

    Args:
        tp_text: Full text of a ``TransformParameters.N.txt`` file.

    Returns:
        str: Updated text (always ends with a newline).
    """
    line_pattern = re.compile(
        r"^\s*\(\s*ResampleInterpolator\s+.*\)\s*\r?$",
        re.MULTILINE,
    )
    replacement = "(ResampleInterpolator FinalNearestNeighborInterpolator)"
    if line_pattern.search(tp_text):
        updated = line_pattern.sub(replacement, tp_text, count=1)
    else:
        text = tp_text.rstrip()
        if text:
            updated = text + "\n" + replacement + "\n"
        else:
            updated = replacement + "\n"
    return updated


class ElastixRegistrationBackend(BaseRegistrationBackend):
    """Registration and mask warping via external **elastix** / **transformix** executables.

    Uses the same CLI as documented in the elastix manual (``-f``, ``-m``, ``-out``, ``-p``,
    optional ``-fMask`` / ``-mMask``; transformix ``-in``, ``-out``, ``-tp``). This avoids
    the itk-elastix Python bindings, which can fail on some Windows environments
    (e.g. connection resets in ITK worker pipelines).

    YAML / ``reg_params`` keys:

    - ``elastix_parameter_files``: required at run time for this backend.
    - ``elastix_parameter_overrides``: optional dict merged into each parameter file.
    - ``elastix_path``: optional path or directory for ``elastix`` (like ``dcm2niix_path``).
    - ``transformix_path``: optional path or directory for ``transformix``.
    - ``elastix_threads``: optional positive int passed as ``-threads`` to both tools.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._parameter_files: List[str] = []
        self._overrides: Dict[str, Any] = {}

        reg_params = dict(kwargs.get("reg_params") or {})

        raw_files = kwargs.pop("elastix_parameter_files", None)
        if raw_files is None:
            raw_files = reg_params.pop("elastix_parameter_files", None)
        if raw_files is not None:
            if isinstance(raw_files, (str, Path)):
                raw_files = [raw_files]
            self._parameter_files = [str(p) for p in raw_files]

        raw_overrides = kwargs.pop("elastix_parameter_overrides", None)
        if raw_overrides is None:
            raw_overrides = reg_params.pop("elastix_parameter_overrides", None)
        if raw_overrides is not None:
            self._overrides = dict(raw_overrides)

        elastix_path = kwargs.pop("elastix_path", None)
        if elastix_path is None:
            elastix_path = reg_params.pop("elastix_path", None)

        transformix_path = kwargs.pop("transformix_path", None)
        if transformix_path is None:
            transformix_path = reg_params.pop("transformix_path", None)

        threads_raw = kwargs.pop("elastix_threads", None)
        if threads_raw is None:
            threads_raw = reg_params.pop("elastix_threads", None)
        self._threads: Optional[int] = (
            int(threads_raw) if threads_raw is not None else None
        )

        kwargs["reg_params"] = reg_params
        super().__init__(**kwargs)

        self._runner = ElastixCliRunner(
            elastix_path=elastix_path,
            transformix_path=transformix_path,
        )
        # Scratch dirs from ``register_image``; removed in ``cleanup_elastix_work_dirs`` after
        # the registration preprocessor finishes one subject (mask warping needs files first).
        self._elastix_work_dirs: List[str] = []

    def cleanup_elastix_work_dirs(self) -> None:
        """Delete elastix CLI temporary trees tracked for this backend instance.

        Invoked by ``RegistrationPreprocessor.__call__`` in a ``finally`` block so scratch
        space is reclaimed even when registration or mask warping raises. Safe to call
        multiple times.

        Note:
            ``*_transform_files`` paths pointing under these directories become invalid after
            this runs.
        """
        for d in self._elastix_work_dirs:
            shutil.rmtree(d, ignore_errors=True)
            logger.debug("Removed elastix temp directory: %s", d)
        self._elastix_work_dirs.clear()

    def _prepare_parameter_files(self, spatial_dim: int, staging_dir: Path) -> List[str]:
        """Copy user parameter files into ``staging_dir`` with dimension + override merges.

        Args:
            spatial_dim: 2 or 3 for typical HABIT volumes.
            staging_dir: Writable directory (unique per registration call).

        Returns:
            List[str]: Absolute paths to merged parameter files in pipeline order.
        """
        if not self._parameter_files:
            raise ValueError(
                "backend='elastix' requires elastix_parameter_files "
                "(YAML: Preprocessing.registration.elastix_parameter_files)."
            )

        prepared: List[str] = []
        for idx, file_path in enumerate(self._parameter_files):
            path = Path(file_path)
            if not path.is_file():
                raise FileNotFoundError(f"elastix parameter file not found: {file_path}")
            body = path.read_text(encoding="utf-8", errors="replace")
            merged = merge_elastix_parameter_file_text(
                body, spatial_dim=spatial_dim, overrides=self._overrides
            )
            out = staging_dir / f"habit_elastix_param_{idx}.txt"
            out.write_text(merged, encoding="utf-8")
            prepared.append(str(out.resolve()))
        return prepared

    def register_image(
        self,
        fixed_image_sitk: sitk.Image,
        moving_image_sitk: sitk.Image,
        fixed_mask_sitk: Optional[sitk.Image] = None,
        moving_mask_sitk: Optional[sitk.Image] = None,
        fixed_image_ants: Optional[Any] = None,
    ) -> Tuple[sitk.Image, List[str]]:
        spatial_dim: int = int(fixed_image_sitk.GetDimension())

        work_dir = Path(
            tempfile.mkdtemp(prefix="habit_elastix_cli_")
        ).resolve()
        self._elastix_work_dirs.append(str(work_dir))
        logger.debug("elastix CLI work directory: %s", work_dir)

        fixed_path = work_dir / "habit_fixed.nii.gz"
        moving_path = work_dir / "habit_moving.nii.gz"
        sitk.WriteImage(
            sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
            str(fixed_path),
        )
        sitk.WriteImage(
            sitk.Cast(moving_image_sitk, sitk.sitkFloat32),
            str(moving_path),
        )

        fixed_mask_path: Optional[Path] = None
        moving_mask_path: Optional[Path] = None
        if fixed_mask_sitk is not None:
            fixed_mask_path = work_dir / "habit_fixed_mask.nii.gz"
            sitk.WriteImage(
                sitk.Cast(fixed_mask_sitk, sitk.sitkUInt8),
                str(fixed_mask_path),
            )
        if moving_mask_sitk is not None:
            moving_mask_path = work_dir / "habit_moving_mask.nii.gz"
            sitk.WriteImage(
                sitk.Cast(moving_mask_sitk, sitk.sitkUInt8),
                str(moving_mask_path),
            )

        param_paths = self._prepare_parameter_files(spatial_dim, work_dir)
        out_dir = work_dir / "elastix_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._runner.run_elastix(
            fixed_image=fixed_path,
            moving_image=moving_path,
            out_dir=out_dir,
            parameter_files=param_paths,
            fixed_mask=fixed_mask_path,
            moving_mask=moving_mask_path,
            threads=self._threads,
        )

        result_path = _find_elastix_result_image(out_dir)
        final_tp = _find_final_transform_parameters(out_dir)
        registered = sitk.ReadImage(str(result_path))
        registered = sitk.Cast(registered, sitk.sitkFloat32)

        # Return the final transform file; prior maps stay alongside it in ``out_dir`` for
        # InitialTransformParametersFileName resolution when running transformix.
        return registered, [str(final_tp)]

    def apply_transform_mask(
        self,
        fixed_reference_sitk: sitk.Image,
        moving_mask_sitk: sitk.Image,
        transform_files: List[str],
        fixed_image_ants: Optional[Any] = None,
    ) -> sitk.Image:
        if not transform_files:
            raise ValueError(
                "transform_files must contain the final elastix TransformParameters path"
            )
        final_tp = Path(transform_files[0]).resolve()
        if not final_tp.is_file():
            raise FileNotFoundError(f"transform parameter file not found: {final_tp}")

        chain_dir = final_tp.parent
        tp_named = list(chain_dir.glob("TransformParameters.*.txt"))
        if not tp_named and (chain_dir / "TransformParameters.txt").is_file():
            tp_named = [chain_dir / "TransformParameters.txt"]
        if not tp_named:
            raise RuntimeError(
                f"No TransformParameters files next to {final_tp}; "
                "cannot resample mask (elastix chain missing)."
            )

        mask_stage = Path(
            tempfile.mkdtemp(prefix="habit_transformix_mask_")
        ).resolve()
        try:
            for src in sorted(tp_named, key=_transform_parameters_txt_sort_key):
                dest = mask_stage / src.name
                shutil.copy2(src, dest)

            local_final = mask_stage / final_tp.name
            if not local_final.is_file():
                raise RuntimeError(f"Expected copied transform file missing: {local_final}")
            raw_tp = local_final.read_text(encoding="utf-8", errors="replace")
            local_final.write_text(
                _ensure_nearest_neighbor_resample(raw_tp),
                encoding="utf-8",
            )

            mask_in = mask_stage / "habit_moving_mask.nii.gz"
            sitk.WriteImage(
                sitk.Cast(moving_mask_sitk, sitk.sitkUInt8),
                str(mask_in),
            )

            tx_out = mask_stage / "transformix_out"
            tx_out.mkdir(parents=True, exist_ok=True)
            self._runner.run_transformix(
                input_image=mask_in,
                out_dir=tx_out,
                transform_parameters=local_final,
                threads=self._threads,
            )

            warped_path = _find_elastix_result_image(tx_out)
            warped = sitk.ReadImage(str(warped_path))
            return sitk.Cast(warped, sitk.sitkUInt8)
        finally:
            shutil.rmtree(mask_stage, ignore_errors=True)
