"""工具 — 合并 CSV

Run: python tests/utils/util_merge_csv.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))

    from habit.cli import cli
    from habit.utils.log_utils import stop_queue_listener

    out_path = root / "demo_data" / "results" / "merge_csv_demo.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        file1 = tmp / "part_a.csv"
        file2 = tmp / "part_b.csv"
        pd.DataFrame({"subjID": ["sub1", "sub2"], "feature_a": [1.0, 2.0]}).to_csv(
            file1, index=False
        )
        pd.DataFrame({"subjID": ["sub1", "sub2"], "feature_b": [10.0, 20.0]}).to_csv(
            file2, index=False
        )
        try:
            sys.argv = [
                "habit",
                "merge-csv",
                str(file1),
                str(file2),
                "-o",
                str(out_path),
                "--index-col",
                "subjID",
                *sys.argv[1:],
            ]
            cli()
        finally:
            stop_queue_listener()


if __name__ == "__main__":
    main()
