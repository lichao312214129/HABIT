"""
Data layout validator for HABIT input directories.

Purpose:
    HABIT expects a fairly specific folder layout:

        data_dir/
            <subject_id>/
                images/
                    <modality_name>/<modality_name>.nii(.gz)
                masks/
                    <modality_name>/<modality_name>.nii(.gz)

    This script scans `data_dir`, reports how many subjects look healthy,
    which ones are missing images or masks, and which modalities are
    available across the cohort. The agent should call this BEFORE writing
    a config so it can pick the correct `images:` / `fixed_image:` etc.

Usage:
    python check_data_layout.py <data_dir>
    python check_data_layout.py <data_dir> --json
    python check_data_layout.py <data_dir> --modalities T1 T2 DWI ADC
                                            # require specific modalities

Exit codes:
    0 = all subjects healthy and (if requested) all required modalities present
    1 = at least one subject has issues
    2 = data_dir does not exist or has no subjects
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


NIFTI_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")


def find_image_file(modality_dir: Path) -> Optional[Path]:
    """Return the first NIfTI/NRRD/MHA file inside `modality_dir`, if any."""
    if not modality_dir.is_dir():
        return None
    for fp in sorted(modality_dir.iterdir()):
        if fp.is_file() and any(fp.name.lower().endswith(s) for s in NIFTI_SUFFIXES):
            return fp
    return None


def scan_subject(subject_dir: Path) -> Dict[str, Any]:
    """Scan a single subject directory and return its modality inventory."""
    images_dir = subject_dir / "images"
    masks_dir = subject_dir / "masks"
    info: Dict[str, Any] = {
        "subject_id": subject_dir.name,
        "has_images_dir": images_dir.is_dir(),
        "has_masks_dir": masks_dir.is_dir(),
        "image_modalities": [],
        "mask_modalities": [],
        "issues": [],
    }
    if images_dir.is_dir():
        for sub in sorted(images_dir.iterdir()):
            if sub.is_dir() and find_image_file(sub) is not None:
                info["image_modalities"].append(sub.name)
            elif sub.is_file() and any(sub.name.lower().endswith(s) for s in NIFTI_SUFFIXES):
                # Some users keep flat layout: images/T1.nii.gz
                info["image_modalities"].append(sub.stem.replace(".nii", ""))
    else:
        info["issues"].append("missing 'images/' directory")

    if masks_dir.is_dir():
        for sub in sorted(masks_dir.iterdir()):
            if sub.is_dir() and find_image_file(sub) is not None:
                info["mask_modalities"].append(sub.name)
            elif sub.is_file() and any(sub.name.lower().endswith(s) for s in NIFTI_SUFFIXES):
                info["mask_modalities"].append(sub.stem.replace(".nii", ""))
    else:
        info["issues"].append("missing 'masks/' directory")

    if not info["image_modalities"]:
        info["issues"].append("no image modalities found")
    if info["has_masks_dir"] and not info["mask_modalities"]:
        info["issues"].append("masks/ exists but no mask files found")

    return info


def aggregate(per_subject: List[Dict[str, Any]],
              required_modalities: Optional[List[str]]) -> Dict[str, Any]:
    """Aggregate per-subject info into cohort-level summary."""
    healthy: List[str] = []
    issues: Dict[str, List[str]] = {}
    image_modality_counts: Dict[str, int] = {}
    mask_modality_counts: Dict[str, int] = {}

    for s in per_subject:
        if s["issues"]:
            issues[s["subject_id"]] = s["issues"]
        else:
            healthy.append(s["subject_id"])
        for m in s["image_modalities"]:
            image_modality_counts[m] = image_modality_counts.get(m, 0) + 1
        for m in s["mask_modalities"]:
            mask_modality_counts[m] = mask_modality_counts.get(m, 0) + 1

    summary: Dict[str, Any] = {
        "total_subjects": len(per_subject),
        "healthy_subjects": len(healthy),
        "subjects_with_issues": len(issues),
        "image_modalities_present": image_modality_counts,
        "mask_modalities_present": mask_modality_counts,
        "issues_by_subject": issues,
    }

    if required_modalities:
        n = len(per_subject)
        missing_per_modality: Dict[str, List[str]] = {}
        for mod in required_modalities:
            missing = [s["subject_id"] for s in per_subject if mod not in s["image_modalities"]]
            if missing:
                missing_per_modality[mod] = missing
        summary["required_modalities"] = required_modalities
        summary["missing_required_modality"] = missing_per_modality
        summary["all_required_present"] = (
            all(image_modality_counts.get(m, 0) == n for m in required_modalities)
        )
    return summary


def render_text(summary: Dict[str, Any], data_dir: Path) -> str:
    """Render the summary into a readable cohort QC report."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"HABIT data layout report for: {data_dir}")
    lines.append("=" * 70)
    lines.append(f"Total subject folders found : {summary['total_subjects']}")
    lines.append(f"Healthy (images + masks ok) : {summary['healthy_subjects']}")
    lines.append(f"Subjects with issues        : {summary['subjects_with_issues']}")
    lines.append("")
    lines.append("Image modalities present (subject count):")
    for mod, n in sorted(summary["image_modalities_present"].items()):
        lines.append(f"  - {mod:<20} {n}")
    if summary["mask_modalities_present"]:
        lines.append("")
        lines.append("Mask modalities present (subject count):")
        for mod, n in sorted(summary["mask_modalities_present"].items()):
            lines.append(f"  - {mod:<20} {n}")
    if "required_modalities" in summary:
        lines.append("")
        lines.append(f"Required modalities check: {summary['required_modalities']}")
        if summary["all_required_present"]:
            lines.append("  -> ALL subjects have ALL required modalities.")
        else:
            for mod, missing in summary["missing_required_modality"].items():
                lines.append(f"  - {mod}: missing in {len(missing)} subjects (e.g. {missing[:5]})")
    if summary["issues_by_subject"]:
        lines.append("")
        lines.append("Per-subject issues (first 20 shown):")
        for sid, problems in list(summary["issues_by_subject"].items())[:20]:
            lines.append(f"  - {sid}: {', '.join(problems)}")
        if len(summary["issues_by_subject"]) > 20:
            lines.append(f"  ... and {len(summary['issues_by_subject']) - 20} more")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    """Parse args, scan, and emit report. Returns exit code."""
    parser = argparse.ArgumentParser(description="HABIT data layout check")
    parser.add_argument("data_dir", type=str, help="Root directory containing per-subject folders")
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="Required image modalities (e.g. T1 T2 DWI ADC)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data_dir does not exist or is not a directory: {data_dir}", file=sys.stderr)
        return 2

    subject_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not subject_dirs:
        print(f"ERROR: no subject subdirectories under {data_dir}", file=sys.stderr)
        return 2

    per_subject = [scan_subject(p) for p in subject_dirs]
    summary = aggregate(per_subject, args.modalities)

    if args.json:
        print(json.dumps({"per_subject": per_subject, "summary": summary}, indent=2))
    else:
        print(render_text(summary, data_dir))

    if summary["subjects_with_issues"] > 0:
        return 1
    if "all_required_present" in summary and not summary["all_required_present"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
