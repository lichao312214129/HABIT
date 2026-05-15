#!/usr/bin/env python3
"""
Print SimpleITK volume geometry for modalities referenced in a preprocessing YAML.

Uses the same path discovery as the batch pipeline: ``PreprocessingConfig`` plus
``habit.utils.io_utils.get_image_and_mask_paths``. Intended for debugging registration
issues (spacing, dimension, direction) on disk **before** any resampling step.

Usage (from repository root):

    python scripts/print_preprocessing_volume_geometry.py
    python scripts/print_preprocessing_volume_geometry.py -c config/preprocessing/config_preprocessing_demo_elastix.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import SimpleITK as sitk

# Repository root: scripts/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from habit.core.preprocessing.config_schemas import PreprocessingConfig  # noqa: E402
from habit.utils.io_utils import get_image_and_mask_paths  # noqa: E402


def _spacing_issues(spacing: Sequence[float]) -> List[str]:
    """Return human-readable issues for ITK spacing (must be positive, finite)."""
    issues: List[str] = []
    for i, s in enumerate(spacing):
        if math.isnan(s) or math.isinf(s):
            issues.append(f"axis_{i}={s} (non-finite)")
        elif s <= 0:
            issues.append(f"axis_{i}={s} (non-positive)")
    return issues


def _format_direction(direction: Sequence[float], dimension: int) -> str:
    """Format direction cosines as a compact matrix string."""
    n = dimension * dimension
    if len(direction) < n:
        return repr(tuple(direction))
    rows: List[str] = []
    for r in range(dimension):
        row = [direction[r * dimension + c] for c in range(dimension)]
        rows.append("[" + ", ".join(f"{v:.6g}" for v in row) + "]")
    return "\n    " + "\n    ".join(rows)


def _print_volume(path: str, logical_key: str) -> None:
    """Load one file and print geometry; no plots (English labels only)."""
    if not path or not Path(path).exists():
        print(f"  [{logical_key}] MISSING PATH: {path}")
        return
    try:
        img: sitk.Image = sitk.ReadImage(path)
    except Exception as exc:
        print(f"  [{logical_key}] FAILED TO READ {path}: {exc}")
        return

    dim = img.GetDimension()
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    pixel_id = img.GetPixelIDValue()

    spacing_problems = _spacing_issues(spacing)
    print(f"  [{logical_key}]")
    print(f"    path:     {path}")
    print(f"    dim:      {dim}")
    print(f"    size:     {size}")
    print(f"    spacing:  {spacing}")
    print(f"    origin:   {origin}")
    print(f"    direction:{_format_direction(direction, dim)}")
    print(f"    pixel_id: {pixel_id}")
    if spacing_problems:
        print(f"    WARNING:  spacing issues: {', '.join(spacing_problems)}")
    print()


def _modalities_for_print(config: PreprocessingConfig) -> Tuple[List[str], Optional[str]]:
    """
    Prefer modalities from the registration step (matches fixed/moving setup);
    otherwise use the first available step with a non-empty ``images`` list.
    """
    pp = config.Preprocessing
    reg = pp.get("registration")
    if reg is not None and reg.images:
        fixed = getattr(reg, "fixed_image", None)
        return list(reg.images), fixed if isinstance(fixed, str) else None

    for step_name in ("resample", "reorientation", "zscore_normalization"):
        step = pp.get(step_name)
        if step is not None and getattr(step, "images", None):
            return list(step.images), None

    modalities: List[str] = []
    for step in pp.values():
        if step.images:
            modalities.extend(step.images)
    return sorted(set(modalities)), None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print SimpleITK geometry for volumes listed in a preprocessing config.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(
            _PROJECT_ROOT / "config" / "preprocessing" / "config_preprocessing_demo_elastix.yaml"
        ),
        help="Path to preprocessing YAML (default: config/preprocessing/config_preprocessing_demo_elastix.yaml).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg: PreprocessingConfig = PreprocessingConfig.from_file(str(config_path))

    modalities, fixed_image = _modalities_for_print(cfg)
    if not modalities:
        print("No modalities found under Preprocessing.*.images in the config.")
        sys.exit(1)

    # Optional: show reorientation target from config (no file I/O)
    reo = cfg.Preprocessing.get("reorientation")
    target_ori = getattr(reo, "target_orientation", None) if reo is not None else None
    reo_mode = getattr(reo, "mode", None) if reo is not None else None

    print("=== Preprocessing volume geometry ===")
    print(f"config:     {config_path}")
    print(f"data_dir:   {cfg.data_dir}")
    print(f"modalities: {modalities}")
    if fixed_image:
        print(f"fixed_image (registration): {fixed_image}")
    if target_ori is not None:
        print(f"reorientation target: {target_ori}  mode: {reo_mode}")
    print()

    images_paths: Dict[str, Dict[str, str]]
    mask_paths: Dict[str, Dict[str, str]]
    images_paths, mask_paths = get_image_and_mask_paths(
        cfg.data_dir,
        auto_select_first_file=cfg.auto_select_first_file,
    )

    if not images_paths:
        print("No subjects discovered under data_dir (get_image_and_mask_paths returned empty).")
        sys.exit(1)

    for subject_id in sorted(images_paths.keys()):
        print(f"--- subject: {subject_id} ---")
        subj_img = images_paths[subject_id]
        subj_msk = mask_paths.get(subject_id, {})

        for mod in modalities:
            p = subj_img.get(mod)
            _print_volume(p if p else "", mod)

            mk = f"mask_{mod}"
            # Same layout as pipeline: mask path is keyed by modality folder name under masks/
            pm = subj_msk.get(mod)
            if pm:
                _print_volume(pm, mk)

        print()


if __name__ == "__main__":
    main()
