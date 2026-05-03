"""
Inspect a feature CSV produced by `habit extract` or `habit radiomics`.

Purpose:
    Before feeding a feature CSV into `habit model`, the agent should verify:
      - subject_id column exists and has no duplicates
      - label column (if any) is binary 0/1
      - no entirely-NaN columns
      - feature count is reasonable
      - identifies constant / near-constant columns that variance threshold
        will drop (gives the agent a heads-up about feature attrition)

Usage:
    python inspect_feature_csv.py <csv_path>
    python inspect_feature_csv.py <csv_path> --subject-id-col subjID --label-col label
    python inspect_feature_csv.py <csv_path> --json

Exit codes:
    0 = csv looks fine
    1 = at least one warning/issue
    2 = file not found / unreadable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def inspect(path: Path,
            subject_id_col: Optional[str],
            label_col: Optional[str]) -> Dict[str, Any]:
    """Compute basic QC stats on a feature CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as exc:
        return {"error": f"cannot read csv: {exc}"}

    info: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns_first_5": list(df.columns[:5]),
        "issues": [],
        "warnings": [],
    }

    # Subject ID column
    sid_col = subject_id_col or df.columns[0]
    info["subject_id_col"] = sid_col
    if sid_col not in df.columns:
        info["issues"].append(f"subject_id_col '{sid_col}' not in CSV columns")
    else:
        n_dup = int(df[sid_col].duplicated().sum())
        info["n_duplicate_subject_ids"] = n_dup
        if n_dup > 0:
            info["issues"].append(f"{n_dup} duplicated subject IDs")

    # Label column
    if label_col is not None:
        info["label_col"] = label_col
        if label_col not in df.columns:
            info["issues"].append(f"label_col '{label_col}' not in CSV columns")
        else:
            uniq = sorted(df[label_col].dropna().unique().tolist())
            info["label_unique_values"] = uniq
            if not set(uniq).issubset({0, 1, 0.0, 1.0, "0", "1"}):
                info["issues"].append(
                    f"label column has non-binary values: {uniq}"
                )

    # Feature columns = everything except sid + label
    skip = {sid_col}
    if label_col:
        skip.add(label_col)
    feat_df = df.drop(columns=[c for c in skip if c in df.columns], errors="ignore")
    # Restrict to numeric features
    numeric = feat_df.select_dtypes(include="number")
    info["n_numeric_features"] = int(numeric.shape[1])
    info["n_non_numeric_features"] = int(feat_df.shape[1] - numeric.shape[1])

    if numeric.shape[1] == 0:
        info["issues"].append("no numeric feature columns found")
        return info

    nan_per_col = numeric.isna().sum()
    all_nan = int((nan_per_col == len(numeric)).sum())
    any_nan = int((nan_per_col > 0).sum())
    info["columns_all_nan"] = all_nan
    info["columns_with_some_nan"] = any_nan
    if all_nan > 0:
        info["issues"].append(f"{all_nan} feature columns are entirely NaN")
    if any_nan > 0:
        info["warnings"].append(f"{any_nan} feature columns contain at least one NaN")

    variances = numeric.var(numeric_only=True)
    n_constant = int((variances == 0).sum())
    n_near_constant = int(((variances > 0) & (variances < 1e-8)).sum())
    info["columns_constant"] = n_constant
    info["columns_near_constant"] = n_near_constant
    if n_constant > 0:
        info["warnings"].append(
            f"{n_constant} constant feature columns will be dropped by variance selector"
        )

    # Heuristic: warn if features >> rows (typical radiomics dim curse)
    if numeric.shape[1] > 5 * info["n_rows"]:
        info["warnings"].append(
            f"feature count ({numeric.shape[1]}) >> sample count ({info['n_rows']}); "
            "use aggressive feature selection (variance + correlation + lasso)."
        )
    return info


def render_text(info: Dict[str, Any], path: Path) -> str:
    """Render a readable summary."""
    if "error" in info:
        return f"ERROR: {info['error']}"
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"Feature CSV inspection: {path}")
    lines.append("=" * 70)
    lines.append(f"rows                       : {info['n_rows']}")
    lines.append(f"total columns              : {info['n_cols']}")
    lines.append(f"subject_id_col             : {info.get('subject_id_col')}")
    if "n_duplicate_subject_ids" in info:
        lines.append(f"duplicate subject IDs      : {info['n_duplicate_subject_ids']}")
    if "label_col" in info:
        lines.append(f"label_col                  : {info['label_col']}")
        lines.append(f"label unique values        : {info.get('label_unique_values')}")
    lines.append(f"numeric features           : {info['n_numeric_features']}")
    lines.append(f"non-numeric features       : {info['n_non_numeric_features']}")
    lines.append(f"all-NaN columns            : {info.get('columns_all_nan', 0)}")
    lines.append(f"columns with any NaN       : {info.get('columns_with_some_nan', 0)}")
    lines.append(f"constant columns           : {info.get('columns_constant', 0)}")
    lines.append(f"near-constant columns      : {info.get('columns_near_constant', 0)}")
    if info["issues"]:
        lines.append("")
        lines.append("Issues:")
        for w in info["issues"]:
            lines.append(f"  - {w}")
    if info["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for w in info["warnings"]:
            lines.append(f"  - {w}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Inspect HABIT feature CSV.")
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--subject-id-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    path = Path(args.csv_path).expanduser().resolve()
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    info = inspect(path, args.subject_id_col, args.label_col)

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(render_text(info, path))

    if "error" in info:
        return 2
    if info["issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
