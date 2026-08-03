# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
        pd.DataFrame({"subject_id": ["sub1", "sub2"], "feature_a": [1.0, 2.0]}).to_csv(
            file1, index=False
        )
        pd.DataFrame({"subject_id": ["sub1", "sub2"], "feature_b": [10.0, 20.0]}).to_csv(
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
                "subject_id",
                *sys.argv[1:],
            ]
            cli()
        finally:
            stop_queue_listener()


if __name__ == "__main__":
    main()
