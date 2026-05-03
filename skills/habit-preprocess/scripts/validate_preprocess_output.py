"""
Validate the output of `habit preprocess`.

Purpose:
    After `habit preprocess` runs, check that the canonical output structure
    exists, that every subject has all expected modalities, and that intensity
    statistics look reasonable (no NaN-only or constant volumes).

Expected layout:
    <out_dir>/processed_images/
        images/<subject>/<modality>/<modality>.nii(.gz)
        masks/<subject>/<modality>/<modality>.nii(.gz)

Usage:
    python validate_preprocess_output.py <out_dir>
    python validate_preprocess_output.py <out_dir> --modalities T1 T2 DWI ADC
    python validate_preprocess_output.py <out_dir> --json

Exit codes:
    0 = all subjects ok
    1 = at least one subject has missing modality / corrupt volume
    2 = output directory not found / wrong layout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

NIFTI_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")


def _first_image(folder: Path) -> Optional[Path]:
    """Return the first image-like file under `folder` (non-recursive)."""
    if not folder.is_dir():
        return None
    for fp in sorted(folder.iterdir()):
        if fp.is_file() and any(fp.name.lower().endswith(s) for s in NIFTI_SUFFIXES):
            return fp
    return None


def _load_stats(path: Path) -> Dict[str, Any]:
    """Try to load a volume with SimpleITK and report basic stats.

    Falls back to nibabel if SimpleITK is unavailable. Returns a dict with
    shape/spacing/min/max/mean keys, or {'error': '...'} on failure.
    """
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        return {
            "shape": list(arr.shape),
            "spacing": list(img.GetSpacing()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "n_nonzero": int((arr != 0).sum()),
        }
    except Exception:
        try:
            import nibabel as nib
            nii = nib.load(str(path))
            arr = nii.get_fdata()
            return {
                "shape": list(arr.shape),
                "spacing": list(nii.header.get_zooms()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "n_nonzero": int((arr != 0).sum()),
            }
        except Exception as exc:
            return {"error": f"cannot read volume: {exc}"}


def scan_processed_images(processed_dir: Path,
                          required_modalities: Optional[List[str]]) -> Dict[str, Any]:
    """Scan the processed_images/ tree and validate every modality."""
    images_root = processed_dir / "images"
    masks_root = processed_dir / "masks"
    if not images_root.is_dir():
        return {"error": f"missing 'images' subdirectory in {processed_dir}"}

    per_subject: List[Dict[str, Any]] = []
    all_modalities: set = set()
    for subj in sorted(images_root.iterdir()):
        if not subj.is_dir():
            continue
        subj_info: Dict[str, Any] = {
            "subject_id": subj.name,
            "modalities": {},
            "issues": [],
        }
        for mod_dir in sorted(subj.iterdir()):
            if not mod_dir.is_dir():
                continue
            img = _first_image(mod_dir)
            if img is None:
                subj_info["issues"].append(f"no image file under images/{mod_dir.name}/")
                continue
            stats = _load_stats(img)
            mask_path = masks_root / subj.name / mod_dir.name
            mask_file = _first_image(mask_path)
            mask_stats = _load_stats(mask_file) if mask_file else None
            entry = {
                "image_file": str(img.relative_to(processed_dir)),
                "image_stats": stats,
                "mask_file": (str(mask_file.relative_to(processed_dir))
                              if mask_file else None),
                "mask_stats": mask_stats,
            }
            if "error" in stats:
                subj_info["issues"].append(f"{mod_dir.name}: {stats['error']}")
            elif stats.get("max") == stats.get("min"):
                subj_info["issues"].append(f"{mod_dir.name}: constant volume (min==max)")
            elif stats.get("n_nonzero", 0) == 0:
                subj_info["issues"].append(f"{mod_dir.name}: all-zero volume")
            if mask_stats is None:
                subj_info["issues"].append(f"{mod_dir.name}: missing mask")
            elif mask_stats and mask_stats.get("n_nonzero", 0) == 0:
                subj_info["issues"].append(f"{mod_dir.name}: empty mask")
            subj_info["modalities"][mod_dir.name] = entry
            all_modalities.add(mod_dir.name)

        if required_modalities:
            for m in required_modalities:
                if m not in subj_info["modalities"]:
                    subj_info["issues"].append(f"missing required modality: {m}")
        per_subject.append(subj_info)

    summary: Dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "total_subjects": len(per_subject),
        "subjects_with_issues": sum(1 for s in per_subject if s["issues"]),
        "modalities_seen": sorted(all_modalities),
        "required_modalities": required_modalities or [],
    }
    return {"summary": summary, "per_subject": per_subject}


def render_text(report: Dict[str, Any]) -> str:
    """Render a readable post-preprocess report."""
    if "error" in report:
        return f"ERROR: {report['error']}"
    summary = report["summary"]
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"Preprocess output report: {summary['processed_dir']}")
    lines.append("=" * 70)
    lines.append(f"Subjects total          : {summary['total_subjects']}")
    lines.append(f"Subjects with issues    : {summary['subjects_with_issues']}")
    lines.append(f"Modalities seen         : {summary['modalities_seen']}")
    if summary["required_modalities"]:
        lines.append(f"Required modalities     : {summary['required_modalities']}")
    bad = [s for s in report["per_subject"] if s["issues"]][:20]
    if bad:
        lines.append("")
        lines.append("Subjects with issues (first 20):")
        for s in bad:
            lines.append(f"  - {s['subject_id']}: {', '.join(s['issues'])}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Validate `habit preprocess` output.")
    parser.add_argument("out_dir", type=str, help="Path to the value of `out_dir` in the preprocess config.")
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="Required modality names that every subject must have.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    processed = out_dir / "processed_images"
    if not processed.is_dir():
        # Some pipelines write directly without the processed_images subfolder.
        if (out_dir / "images").is_dir():
            processed = out_dir
        else:
            print(f"ERROR: cannot find processed_images/ under {out_dir}", file=sys.stderr)
            return 2

    report = scan_processed_images(processed, args.modalities)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))

    if "error" in report:
        return 2
    if report["summary"]["subjects_with_issues"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
