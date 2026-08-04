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
"""
Freeze the synthetic fast golden gate used by CI.

Unlike ``scripts/make_golden_baseline.py``, these cases never touch
``demo_data/``. They exercise the v1 recipes and a minimal ML workflow on
in-memory synthetic inputs and write their fingerprints to
``tests/golden/baseline/fast/``.

Usage
-----
    python scripts/make_fast_golden_baseline.py
    python scripts/make_fast_golden_baseline.py --case habitat_two_step
    python scripts/make_fast_golden_baseline.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_golden_baseline import environment_fingerprint  # noqa: E402
from tests.golden.fast._runner import (  # noqa: E402
    FAST_GOLDEN_CASES,
    baseline_dir,
    compare_fast_records,
    run_case,
)


def write_baseline(record: dict) -> Path:
    """
    Persist one fast baseline record.

    Args:
        record: Case fingerprint document.

    Returns:
        Path of the written JSON file.
    """
    destination = baseline_dir() / f"{record['case']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Generate or verify the synthetic fast golden baselines.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in FAST_GOLDEN_CASES],
        help="Restrict to one case; repeatable. Defaults to all cases.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "demo_data" / "results" / "_golden_fast",
        help="Scratch directory for baseline generation runs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-run and compare against stored baselines without writing.",
    )
    args = parser.parse_args(argv)

    selected = [
        case
        for case in FAST_GOLDEN_CASES
        if args.case is None or case.name in args.case
    ]
    environment = environment_fingerprint()
    problems = []
    for case in selected:
        current = run_case(case, (args.out_root / case.name).resolve())
        current["environment"] = environment
        if args.verify:
            baseline_path = baseline_dir() / f"{case.name}.json"
            if not baseline_path.is_file():
                print(f"missing baseline: {baseline_path}")
                problems.append(case.name)
                continue
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            drift = compare_fast_records(baseline, current)
            if drift:
                print(f"DRIFT {case.name}:")
                for line in drift[:20]:
                    print(f"  {line}")
                problems.append(case.name)
            else:
                print(f"OK {case.name}")
        else:
            path = write_baseline(current)
            print(f"wrote {path}")

    if problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
