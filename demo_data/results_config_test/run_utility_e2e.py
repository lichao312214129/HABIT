#!/usr/bin/env python3
"""E2E checks for HABIT utility CLI commands (no YAML config file).

Commands covered:
  habit merge-csv
  habit dice
  habit dicom-info
  habit --help / subcommand --help smoke tests

Usage::

    python demo_data/results_config_test/scripts/run_utility_e2e.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Cannot find repo root from {start}")


def _find_demo_root(repo_root: Path) -> Path:
    """Return demo_data directory under the repository root."""
    demo_root = repo_root / "demo_data"
    if demo_root.is_dir():
        return demo_root.resolve()
    raise RuntimeError(f"demo_data directory not found under {repo_root}")
    repo_root: Path,
    argv: List[str],
    log_path: Path,
    timeout: int = 600,
) -> Tuple[int, float]:
    cmd = [sys.executable, "-m", "habit.cli", *argv]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Command: {' '.join(cmd)}\n\n")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write("\nTIMEOUT\n")
            rc = 124
    return rc, time.monotonic() - t0


def _record_check(
    summary: Dict[str, Any],
    failed_counter: List[int],
    name: str,
    argv: List[str],
    rc: int,
    elapsed: float,
    log_path: Path,
    expected_output: Path | None,
) -> None:
    ok = rc == 0
    if expected_output is not None and not expected_output.is_file():
        ok = False
    summary["checks"].append(
        {
            "name": name,
            "argv": argv,
            "exit_code": rc,
            "elapsed_sec": round(elapsed, 1),
            "expected_output": str(expected_output) if expected_output else None,
            "status": "passed" if ok else "failed",
            "log": str(log_path),
        }
    )
    if ok:
        print(f"[PASS] {name} ({elapsed:.1f}s)")
    else:
        failed_counter[0] += 1
        print(f"[FAIL] {name} rc={rc} ({elapsed:.1f}s) log={log_path}")


def main() -> int:
    script_path = Path(__file__).resolve()
    results_root = script_path.parents[1]
    repo_root = _find_repo_root(script_path)
    demo_root = _find_demo_root(repo_root)
    logs_dir = results_root / "logs" / "utility"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "checks": [],
    }
    failed_counter = [0]

    try:
        import pydicom  # noqa: F401
        pydicom_ok = True
    except ImportError:
        pydicom_ok = False

    print("Utility E2E checks")
    print("-" * 60)

    for sub in (
        "preprocess", "sort-dicom", "get-habitat", "extract", "model", "cv",
        "compare", "icc", "radiomics", "retest", "dice", "dicom-info",
        "merge-csv", "gui",
    ):
        name = f"help_{sub}"
        log_path = logs_dir / f"{name}_{stamp}.log"
        rc, elapsed = _run_habit(repo_root, [sub, "--help"], log_path)
        _record_check(summary, failed_counter, name, [sub, "--help"], rc, elapsed, log_path, None)

    merge_out = results_root / "utility_outputs" / "merge_demo.csv"
    merge_out.parent.mkdir(parents=True, exist_ok=True)
    dice_out = results_root / "utility_outputs" / "dice_demo.csv"
    resample = repo_root / ".cursor" / "test" / "resample_02"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        f1 = tmp / "a.csv"
        f2 = tmp / "b.csv"
        pd.DataFrame({"subjID": ["s1", "s2"], "a": [1.0, 2.0]}).to_csv(f1, index=False)
        pd.DataFrame({"subjID": ["s1", "s2"], "b": [3.0, 4.0]}).to_csv(f2, index=False)

        merge_argv = [
            "merge-csv", str(f1), str(f2), "-o", str(merge_out), "--index-col", "subjID",
        ]
        log_path = logs_dir / f"merge_csv_{stamp}.log"
        rc, elapsed = _run_habit(repo_root, merge_argv, log_path)
        _record_check(summary, failed_counter, "merge_csv", merge_argv, rc, elapsed, log_path, merge_out)

        if resample.is_dir():
            dice_argv = [
                "dice",
                "--input1", str(resample),
                "--input2", str(resample),
                "--output", str(dice_out),
                "--mask-keyword", "masks",
            ]
            log_path = logs_dir / f"dice_resample02_{stamp}.log"
            rc, elapsed = _run_habit(repo_root, dice_argv, log_path)
            _record_check(
                summary, failed_counter, "dice_resample02", dice_argv, rc, elapsed, log_path, dice_out
            )

    dicom_dir = demo_root / "dicom"
    dicom_out = results_root / "utility_outputs" / "dicom_info.csv"
    if pydicom_ok and dicom_dir.is_dir() and any(dicom_dir.iterdir()):
        dicom_argv = [
            "dicom-info",
            "-i", str(dicom_dir),
            "-o", str(dicom_out),
            "-f", "csv",
            "--group-by-series",
            "-j", "2",
        ]
        log_path = logs_dir / f"dicom_info_{stamp}.log"
        rc, elapsed = _run_habit(repo_root, dicom_argv, log_path)
        _record_check(summary, failed_counter, "dicom_info", dicom_argv, rc, elapsed, log_path, dicom_out)
    else:
        reason = "pydicom not installed" if not pydicom_ok else "DICOM demo folder missing"
        print(f"[SKIP] dicom_info: {reason}")
        summary["checks"].append({"name": "dicom_info", "status": "skipped", "reason": reason})

    failed = failed_counter[0]
    passed = sum(1 for c in summary["checks"] if c.get("status") == "passed")
    summary["failed"] = failed
    summary["passed"] = passed
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_json = results_root / f"utility_summary_{stamp}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 60)
    print(f"Utility done: passed={passed} failed={failed}")
    print(f"Summary: {out_json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
