"""
Validate the output of `habit get-habitat`.

Purpose:
    After habitat clustering finishes, verify that:
      1. Every subject has a `*_habitats*.nrrd` file.
      2. (two_step) every subject has a `*_supervoxel.nrrd` file.
      3. `habitats.csv` exists and the per-subject habitat fractions sum to ~1.
      4. The habitat label maps contain >= 2 distinct labels (i.e. real
         clustering happened, not a degenerate 1-cluster collapse).

Usage:
    python validate_habitat_output.py <out_dir>
    python validate_habitat_output.py <out_dir> --two-step
    python validate_habitat_output.py <out_dir> --json

Exit codes:
    0 = all subjects ok
    1 = at least one subject missing a file or has a degenerate map
    2 = output directory does not exist or no subjects
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _find_one(folder: Path, pattern: str) -> Path | None:
    """Return the first file matching `pattern` inside `folder`, or None."""
    if not folder.is_dir():
        return None
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def _read_label_stats(path: Path) -> Dict[str, Any]:
    """Read a label volume and report unique labels + voxel counts."""
    try:
        import SimpleITK as sitk
        import numpy as np
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        uniq, counts = np.unique(arr, return_counts=True)
        # Drop background 0 from the report's "habitat label set" but keep
        # it in the raw stats for transparency.
        habitat_labels = [int(u) for u in uniq if u != 0]
        return {
            "shape": list(arr.shape),
            "unique_labels": [int(u) for u in uniq],
            "label_counts": {int(u): int(c) for u, c in zip(uniq, counts)},
            "n_habitat_labels": len(habitat_labels),
        }
    except Exception as exc:
        return {"error": f"cannot read label volume: {exc}"}


def scan_subjects(out_dir: Path, two_step: bool) -> Dict[str, Any]:
    """Walk every subject folder under `out_dir` and validate files."""
    per_subject: List[Dict[str, Any]] = []
    for subj in sorted(out_dir.iterdir()):
        if not subj.is_dir():
            continue
        # Skip well-known non-subject dirs created by HABIT
        if subj.name in {"visualizations", "logs"}:
            continue
        info: Dict[str, Any] = {
            "subject_id": subj.name,
            "habitat_file": None,
            "supervoxel_file": None,
            "habitat_stats": None,
            "issues": [],
        }
        habitat = (
            _find_one(subj, "*_habitats_remapped.nrrd")
            or _find_one(subj, "*_habitats.nrrd")
        )
        if habitat is None:
            info["issues"].append("no *_habitats*.nrrd found")
        else:
            info["habitat_file"] = habitat.name
            stats = _read_label_stats(habitat)
            info["habitat_stats"] = stats
            if "error" in stats:
                info["issues"].append(stats["error"])
            elif stats["n_habitat_labels"] < 2:
                info["issues"].append(
                    f"degenerate habitat map: only {stats['n_habitat_labels']} non-zero label(s)"
                )

        if two_step:
            sv = _find_one(subj, "*_supervoxel.nrrd")
            if sv is None:
                info["issues"].append("two_step mode but no *_supervoxel.nrrd")
            else:
                info["supervoxel_file"] = sv.name

        per_subject.append(info)
    return {"per_subject": per_subject}


def check_habitats_csv(out_dir: Path, tolerance: float = 0.02) -> Dict[str, Any]:
    """Sanity-check the cohort-level `habitats.csv`."""
    csv_path = out_dir / "habitats.csv"
    if not csv_path.is_file():
        return {"present": False, "issue": f"{csv_path.name} not found"}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return {"present": True, "issue": f"cannot read csv: {exc}"}
    info: Dict[str, Any] = {
        "present": True,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
    }
    # Heuristic: any column named like 'habitat_*_fraction' should sum to ~1 per row
    fraction_cols = [c for c in df.columns if "fraction" in c.lower()]
    if fraction_cols:
        sums = df[fraction_cols].sum(axis=1)
        bad = ((sums - 1.0).abs() > tolerance).sum()
        info["fraction_columns"] = fraction_cols
        info["rows_with_bad_fraction_sum"] = int(bad)
        if bad > 0:
            info["issue"] = (
                f"{bad} rows have fraction sums outside 1±{tolerance}; "
                "check habitat extraction."
            )
    return info


def render_text(report: Dict[str, Any], out_dir: Path) -> str:
    """Pretty-print the report to the terminal."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"Habitat output report: {out_dir}")
    lines.append("=" * 70)
    subs = report["subjects"]["per_subject"]
    bad = [s for s in subs if s["issues"]]
    lines.append(f"Subjects total          : {len(subs)}")
    lines.append(f"Subjects with issues    : {len(bad)}")
    csv = report.get("habitats_csv", {})
    lines.append("")
    lines.append("habitats.csv:")
    if not csv.get("present"):
        lines.append(f"  - {csv.get('issue', 'missing')}")
    else:
        lines.append(f"  - rows           : {csv.get('n_rows')}")
        if csv.get("fraction_columns"):
            lines.append(f"  - fraction cols  : {csv.get('fraction_columns')}")
            lines.append(f"  - bad fraction sums : {csv.get('rows_with_bad_fraction_sum', 0)}")
        if csv.get("issue"):
            lines.append(f"  - WARNING        : {csv['issue']}")
    if bad:
        lines.append("")
        lines.append("Subjects with issues (first 20):")
        for s in bad[:20]:
            lines.append(f"  - {s['subject_id']}: {', '.join(s['issues'])}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Validate habit get-habitat output.")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--two-step", action="store_true",
                        help="Also require *_supervoxel.nrrd per subject (two_step mode).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        print(f"ERROR: out_dir does not exist: {out_dir}", file=sys.stderr)
        return 2

    report = {
        "subjects": scan_subjects(out_dir, two_step=args.two_step),
        "habitats_csv": check_habitats_csv(out_dir),
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report, out_dir))

    bad = sum(1 for s in report["subjects"]["per_subject"] if s["issues"])
    if bad > 0 or report["habitats_csv"].get("issue"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
