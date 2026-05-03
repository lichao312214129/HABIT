"""
Generate train_ids.txt / test_ids.txt for `habit model --split_method custom`.

Purpose:
    Many users want a fixed (reproducible) train/test split rather than a
    random one each run. This script reads a feature CSV, optionally a
    label column, and writes two text files (one subject ID per line) that
    can be referenced from a `habit model` config.

Usage:
    python prepare_split_files.py <csv_path> --subject-id-col subjID \
        --label-col label --test-size 0.3 --output-dir ./splits

    python prepare_split_files.py <csv_path> --subject-id-col subjID \
        --custom-test-ids id1 id2 id3 --output-dir ./splits

Outputs:
    <output_dir>/train_ids.txt
    <output_dir>/test_ids.txt
    <output_dir>/split_summary.json   (counts + class balance)

Exit codes:
    0 = success
    1 = bad input
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_split(train_ids: List[str], test_ids: List[str],
                output_dir: Path) -> Dict[str, Path]:
    """Write the two ID files and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_ids.txt"
    test_path = output_dir / "test_ids.txt"
    train_path.write_text("\n".join(str(x) for x in train_ids), encoding="utf-8")
    test_path.write_text("\n".join(str(x) for x in test_ids), encoding="utf-8")
    return {"train": train_path, "test": test_path}


def class_balance(df, ids: List[str], sid_col: str, label_col: Optional[str]) -> Dict[str, Any]:
    """Compute class counts inside a given ID subset."""
    if label_col is None:
        return {"n": len(ids)}
    sub = df[df[sid_col].astype(str).isin([str(i) for i in ids])]
    counts = sub[label_col].value_counts().to_dict()
    return {"n": len(ids), "label_counts": {str(k): int(v) for k, v in counts.items()}}


def main() -> int:
    """Build the splits and write files."""
    parser = argparse.ArgumentParser(description="Generate train/test ID files for HABIT ML.")
    parser.add_argument("csv_path", type=str, help="Path to feature CSV")
    parser.add_argument("--subject-id-col", required=True)
    parser.add_argument("--label-col", default=None,
                        help="If given, stratify on this column (binary 0/1).")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Test fraction (0-1). Ignored if --custom-test-ids is set.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./splits")
    parser.add_argument("--custom-test-ids", nargs="*", default=None,
                        help="Force these IDs into the test set.")
    args = parser.parse_args()

    path = Path(args.csv_path).expanduser().resolve()
    if not path.is_file():
        print(f"ERROR: csv not found: {path}", file=sys.stderr)
        return 1

    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR: cannot read csv: {exc}", file=sys.stderr)
        return 1

    if args.subject_id_col not in df.columns:
        print(f"ERROR: subject_id_col '{args.subject_id_col}' not in CSV columns: {list(df.columns)[:10]}",
              file=sys.stderr)
        return 1
    if args.label_col is not None and args.label_col not in df.columns:
        print(f"ERROR: label_col '{args.label_col}' not in CSV", file=sys.stderr)
        return 1

    all_ids = df[args.subject_id_col].astype(str).tolist()

    if args.custom_test_ids:
        test_ids = [str(i) for i in args.custom_test_ids]
        unknown = [i for i in test_ids if i not in all_ids]
        if unknown:
            print(f"WARNING: {len(unknown)} of the custom test IDs are not in the CSV: {unknown[:10]}",
                  file=sys.stderr)
        train_ids = [i for i in all_ids if i not in set(test_ids)]
    else:
        try:
            from sklearn.model_selection import train_test_split
        except Exception:
            print("ERROR: scikit-learn is required for stratified split", file=sys.stderr)
            return 1
        if args.label_col:
            stratify = df[args.label_col]
        else:
            stratify = None
        train_ids, test_ids = train_test_split(
            all_ids,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    paths = write_split(train_ids, test_ids, out_dir)

    summary = {
        "train_path": str(paths["train"]),
        "test_path": str(paths["test"]),
        "train": class_balance(df, train_ids, args.subject_id_col, args.label_col),
        "test": class_balance(df, test_ids, args.subject_id_col, args.label_col),
    }

    summary_path = out_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {paths['train']}\n       {paths['test']}\n       {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
