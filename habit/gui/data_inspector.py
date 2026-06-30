# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Dataset inspection helpers for the habitat wizard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

NIFTI_SUFFIXES: tuple[str, ...] = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")


@dataclass
class HabitatDatasetReport:
    """Summary of a habitat input directory scan."""

    data_dir: str
    n_subjects: int = 0
    n_ok: int = 0
    modalities: List[str] = field(default_factory=list)
    issues_by_subject: Dict[str, List[str]] = field(default_factory=dict)
    status_message: str = ""

    @property
    def healthy_subjects(self) -> int:
        """Return number of subjects without layout issues."""
        return self.n_ok


def _find_image_file(modality_dir: Path) -> Optional[Path]:
    """
    Return the first supported image file inside a modality folder.

    Args:
        modality_dir: Candidate modality directory.

    Returns:
        Optional[Path]: First matching image path or None.
    """
    if not modality_dir.is_dir():
        return None
    for fp in sorted(modality_dir.iterdir()):
        if fp.is_file() and any(fp.name.lower().endswith(suffix) for suffix in NIFTI_SUFFIXES):
            return fp
    return None


def _scan_subject(subject_dir: Path) -> Dict[str, Any]:
    """
    Scan one subject folder for images/masks layout.

    Args:
        subject_dir: Subject directory under the data root.

    Returns:
        Dict[str, Any]: Per-subject scan record with modalities and issues.
    """
    images_dir = subject_dir / "images"
    masks_dir = subject_dir / "masks"
    info: Dict[str, Any] = {
        "subject_id": subject_dir.name,
        "image_modalities": [],
        "issues": [],
    }
    if images_dir.is_dir():
        for sub in sorted(images_dir.iterdir()):
            if sub.is_dir() and _find_image_file(sub) is not None:
                info["image_modalities"].append(sub.name)
            elif sub.is_file() and any(sub.name.lower().endswith(s) for s in NIFTI_SUFFIXES):
                info["image_modalities"].append(sub.stem.replace(".nii", ""))
    else:
        info["issues"].append("missing images/ directory")

    if masks_dir.is_dir():
        has_mask = any(
            (sub.is_dir() and _find_image_file(sub) is not None)
            or (sub.is_file() and any(sub.name.lower().endswith(s) for s in NIFTI_SUFFIXES))
            for sub in masks_dir.iterdir()
        )
        if not has_mask:
            info["issues"].append("masks/ exists but no mask files found")
    else:
        info["issues"].append("missing masks/ directory")

    if not info["image_modalities"]:
        info["issues"].append("no image modalities found")
    return info


def inspect_habitat_dataset(data_dir: str) -> HabitatDatasetReport:
    """
    Inspect a habitat ``data_dir`` and summarize cohort readiness.

    Args:
        data_dir: Root directory containing one folder per subject.

    Returns:
        HabitatDatasetReport: Scan summary for GUI display.
    """
    root = Path(str(data_dir or "").strip())
    if not root.is_dir():
        return HabitatDatasetReport(
            data_dir=str(data_dir),
            status_message="Data directory not found or not accessible.",
        )

    subject_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not subject_dirs:
        return HabitatDatasetReport(
            data_dir=str(root),
            status_message="No subject folders found under data_dir.",
        )

    per_subject = [_scan_subject(subject_dir) for subject_dir in subject_dirs]
    issues = {s["subject_id"]: s["issues"] for s in per_subject if s["issues"]}
    healthy = [s["subject_id"] for s in per_subject if not s["issues"]]
    modality_counts: Dict[str, int] = {}
    for subject in per_subject:
        for modality in subject["image_modalities"]:
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
    modalities = sorted(modality_counts.keys())

    if issues:
        status = (
            f"Found {len(healthy)}/{len(subject_dirs)} healthy subjects. "
            f"{len(issues)} subject(s) have layout issues."
        )
    else:
        status = f"All {len(subject_dirs)} subject(s) passed the layout check."

    return HabitatDatasetReport(
        data_dir=str(root),
        n_subjects=len(subject_dirs),
        n_ok=len(healthy),
        modalities=modalities,
        issues_by_subject=issues,
        status_message=status,
    )


def default_modality_selection(
    report: HabitatDatasetReport,
    expected_modalities: Sequence[str],
) -> List[str]:
    """
    Choose default modality checkbox values after scanning data.

    Args:
        report: Dataset scan report.
        expected_modalities: Modalities suggested by the selected template.

    Returns:
        List[str]: Modalities to pre-select in the wizard.
    """
    available: Set[str] = set(report.modalities)
    expected = [str(m) for m in expected_modalities if str(m) in available]
    if expected:
        return expected
    if report.modalities:
        return list(report.modalities)
    return list(expected_modalities) if expected_modalities else ["T2"]


def format_inspection_table(report: HabitatDatasetReport) -> str:
    """
    Render a plain-text inspection table for ``gr.Textbox`` display.

    Args:
        report: Dataset scan report.

    Returns:
        str: Multi-line table text.
    """
    lines = [
        f"Directory: {report.data_dir}",
        f"Subjects: {report.n_ok}/{report.n_subjects} healthy",
        f"Modalities: {', '.join(report.modalities) if report.modalities else '—'}",
        "",
        "Issues:",
    ]
    if not report.issues_by_subject:
        lines.append("  (none)")
    else:
        for subject_id, subject_issues in sorted(report.issues_by_subject.items()):
            joined = "; ".join(subject_issues)
            lines.append(f"  - {subject_id}: {joined}")
    return "\n".join(lines)


__all__ = [
    "HabitatDatasetReport",
    "inspect_habitat_dataset",
    "default_modality_selection",
    "format_inspection_table",
]
