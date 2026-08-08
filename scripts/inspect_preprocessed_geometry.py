#!/usr/bin/env python
"""Print geometry for every image/mask under demo_data/preprocessed.

Compares each modality image to its ROI mask with HABIT's
``Geometry.is_compatible_with`` (same check that raises GeometryError in
``raw`` feature extraction). Also prints SimpleITK Size / Spacing / Origin /
Direction and array shape so you can cross-check against ITK-SNAP.

Run from the repository root::

    python scripts/inspect_preprocessed_geometry.py
    python scripts/inspect_preprocessed_geometry.py --root demo_data/preprocessed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

# Allow ``python scripts/...`` without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from habit.contracts.geometry import Geometry  # noqa: E402


def _fmt_tuple(values: Sequence[float], ndigits: int = 6) -> str:
    """Format a float sequence for one-line logging."""
    return "(" + ", ".join(f"{float(v):.{ndigits}g}" for v in values) + ")"


def _direction_matrix(direction: Sequence[float], dim: int) -> np.ndarray:
    """Reshape a flat SimpleITK direction vector into a dim x dim matrix."""
    return np.asarray(direction, dtype=np.float64).reshape(dim, dim)


def _geometry_from_sitk(image: sitk.Image) -> Geometry:
    """Build a HABIT Geometry from a SimpleITK image (matches adapters)."""
    # SimpleITK size is (x, y, z); HABIT Geometry.shape is NumPy (z, y, x).
    size_xyz = tuple(int(v) for v in image.GetSize())
    shape = tuple(reversed(size_xyz))
    spacing = tuple(float(v) for v in image.GetSpacing())
    origin = tuple(float(v) for v in image.GetOrigin())
    direction = tuple(float(v) for v in image.GetDirection())
    return Geometry(
        shape=shape,
        spacing=spacing,
        origin=origin,
        direction=direction,
    )


def _describe_nrrd(path: Path) -> Dict[str, Any]:
    """Load one NRRD and return a dict of geometry fields for printing."""
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)  # z, y, x
    dim = int(image.GetDimension())
    direction = tuple(float(v) for v in image.GetDirection())
    geom = _geometry_from_sitk(image)
    return {
        "path": path,
        "size_sitk": tuple(int(v) for v in image.GetSize()),  # x, y, z
        "array_shape": tuple(int(v) for v in arr.shape),  # z, y, x
        "spacing": tuple(float(v) for v in image.GetSpacing()),
        "origin": tuple(float(v) for v in image.GetOrigin()),
        "direction": direction,
        "direction_matrix": _direction_matrix(direction, dim),
        "geometry": geom,
        "pixel_id": image.GetPixelIDTypeAsString(),
        "dtype": str(arr.dtype),
    }


def _diff_origin(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    """Component-wise origin difference a - b."""
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _print_block(kind: str, modality: str, info: Dict[str, Any]) -> None:
    """Print one image/mask block."""
    rel = info["path"]
    try:
        rel = info["path"].relative_to(_REPO_ROOT)
    except ValueError:
        pass
    print(f"  [{kind:5}] {modality}")
    print(f"         file: {rel}")
    size = info["size_sitk"]
    print(f"         Size (ITK x,y,z): ({size[0]}, {size[1]}, {size[2]})")
    print(f"         Array shape (z,y,x): {info['array_shape']}  dtype={info['dtype']}")
    print(f"         Spacing: {_fmt_tuple(info['spacing'])}")
    print(f"         Origin:  {_fmt_tuple(info['origin'])}")
    print(f"         Direction (flat): {_fmt_tuple(info['direction'])}")
    mat = info["direction_matrix"]
    print("         Direction matrix:")
    for row in mat:
        print("           " + _fmt_tuple(row))


def _find_nrrd(folder: Path) -> Optional[Path]:
    """Return the first .nrrd under folder, or None."""
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.nrrd"))
    return files[0] if files else None


def inspect_subject(subject_dir_images: Path, subject_dir_masks: Path) -> None:
    """Print geometry for one subject and HABIT compatibility per modality."""
    subject_id = subject_dir_images.name
    print("=" * 78)
    print(f"SUBJECT: {subject_id}")
    print("=" * 78)

    modalities = sorted(
        {
            p.name
            for p in list(subject_dir_images.iterdir()) + list(subject_dir_masks.iterdir())
            if p.is_dir()
        }
    )
    if not modalities:
        print("  (no modality folders)")
        return

    image_infos: Dict[str, Dict[str, Any]] = {}
    mask_infos: Dict[str, Dict[str, Any]] = {}

    for modality in modalities:
        img_path = _find_nrrd(subject_dir_images / modality)
        msk_path = _find_nrrd(subject_dir_masks / modality)
        if img_path is not None:
            image_infos[modality] = _describe_nrrd(img_path)
            _print_block("IMAGE", modality, image_infos[modality])
        else:
            print(f"  [IMAGE] {modality}: MISSING")
        if msk_path is not None:
            mask_infos[modality] = _describe_nrrd(msk_path)
            _print_block("MASK", modality, mask_infos[modality])
        else:
            print(f"  [MASK ] {modality}: MISSING")

        if modality in image_infos and modality in mask_infos:
            img_g = image_infos[modality]["geometry"]
            msk_g = mask_infos[modality]["geometry"]
            ok = img_g.is_compatible_with(msk_g)
            d_origin = _diff_origin(
                image_infos[modality]["origin"], mask_infos[modality]["origin"]
            )
            d_dir = (
                image_infos[modality]["direction_matrix"]
                - mask_infos[modality]["direction_matrix"]
            )
            print(f"  [CHECK] {modality}: HABIT is_compatible_with = {ok}")
            print(f"         origin(image) - origin(mask) = {_fmt_tuple(d_origin)}")
            print(
                "         max |direction diff| = "
                f"{float(np.max(np.abs(d_dir))):.6g}"
            )
        print()

    # Cross-check: are all images on one grid? all masks on one grid?
    if len(image_infos) >= 2:
        mods = list(image_infos.keys())
        ref = image_infos[mods[0]]["geometry"]
        same = all(
            image_infos[m]["geometry"].is_compatible_with(ref) for m in mods[1:]
        )
        print(f"  [SUMMARY] all images share one grid: {same}")
    if len(mask_infos) >= 2:
        mods = list(mask_infos.keys())
        ref = mask_infos[mods[0]]["geometry"]
        same = all(
            mask_infos[m]["geometry"].is_compatible_with(ref) for m in mods[1:]
        )
        print(f"  [SUMMARY] all masks share one grid: {same}")
    if image_infos and mask_infos:
        img_ref = next(iter(image_infos.values()))["geometry"]
        msk_ref = next(iter(mask_infos.values()))["geometry"]
        print(
            "  [SUMMARY] any image vs any mask compatible: "
            f"{img_ref.is_compatible_with(msk_ref)}"
        )
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: inspect demo_data/preprocessed geometry."""
    parser = argparse.ArgumentParser(
        description="Inspect image/mask geometry under demo_data/preprocessed."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "demo_data" / "preprocessed",
        help="Root with images/ and masks/ (default: demo_data/preprocessed)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject ids (default: all under images/)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.root.resolve()
    images_root = root / "images"
    masks_root = root / "masks"
    if not images_root.is_dir():
        print(f"ERROR: missing {images_root}", file=sys.stderr)
        return 1

    subjects = (
        list(args.subjects)
        if args.subjects
        else sorted(p.name for p in images_root.iterdir() if p.is_dir())
    )
    print(f"Root: {root}")
    print(
        "HABIT Geometry.is_compatible_with tolerances: "
        "spacing/origin atol=1e-5, direction atol=1e-4"
    )
    print()

    for subject_id in subjects:
        inspect_subject(images_root / subject_id, masks_root / subject_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
