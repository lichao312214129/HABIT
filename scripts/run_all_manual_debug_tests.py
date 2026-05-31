"""Run every one-script-one-function test under tests/ (excluding pytest test_*.py)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_NAMES = {"__init__.py", "conftest.py"}
SKIP_PREFIXES = ("test_",)
SKIP_SCRIPTS = frozenset(
    {
        "workflow_preprocess_to_compare.py",
        "workflow_resample_to_compare.py",
    }
)

TIMEOUT_DEFAULT = 120
TIMEOUT_HEAVY = 600
HEAVY_KEYWORDS = ("_train.py", "preprocess_n4", "extract_features", "ml_kfold", "registration_elastix")


def discover_scripts() -> list[Path]:
    scripts: list[Path] = []
    for sub in (
        "preprocessing",
        "habitat",
        "feature_extraction",
        "machine_learning",
        "model_comparison",
        "integration",
        "utils",
    ):
        folder = ROOT / "tests" / sub
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.py")):
            name = path.name
            if name in SKIP_NAMES or name in SKIP_SCRIPTS or any(name.startswith(p) for p in SKIP_PREFIXES):
                continue
            text = path.read_text(encoding="utf-8")
            if "from habit.cli import cli" not in text or "def main()" not in text:
                continue
            scripts.append(path)

    def _sort_key(path: Path) -> tuple[int, str]:
        # Run *_train.py before *_predict.py so shared pipeline outputs stay consistent.
        name = path.name
        if name.endswith("_train.py"):
            return (0, name)
        if name.endswith("_predict.py"):
            return (1, name)
        return (0, name)

    return sorted(scripts, key=_sort_key)


def timeout_for(path: Path) -> int:
    rel = path.as_posix()
    if any(k in rel for k in HEAVY_KEYWORDS):
        return TIMEOUT_HEAVY
    return TIMEOUT_DEFAULT


def run_script(path: Path) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    timeout = timeout_for(path)
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - started
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = out[-4000:] if len(out) > 4000 else out
        return {
            "script": rel,
            "exit_code": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
            "timed_out": False,
            "output_tail": tail,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        out = ""
        if exc.stdout:
            out += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="replace")
        if exc.stderr:
            out += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", errors="replace")
        tail = out[-4000:] if len(out) > 4000 else out
        return {
            "script": rel,
            "exit_code": -999,
            "elapsed_sec": round(elapsed, 1),
            "timed_out": True,
            "output_tail": tail,
        }


def main() -> int:
    scripts = discover_scripts()
    print(f"Found {len(scripts)} test script(s)\n")
    results: list[dict] = []
    for path in scripts:
        rel = path.relative_to(ROOT).as_posix()
        print(f"Running {rel} ...", flush=True)
        result = run_script(path)
        status = "TIMEOUT" if result["timed_out"] else f"exit={result['exit_code']}"
        print(f"  -> {status} ({result['elapsed_sec']}s)\n", flush=True)
        results.append(result)

    ok = [r for r in results if r["exit_code"] == 0]
    fail = [r for r in results if r["exit_code"] != 0 and not r["timed_out"]]
    timeout = [r for r in results if r["timed_out"]]

    print("=" * 72)
    print(f"PASS: {len(ok)}  FAIL: {len(fail)}  TIMEOUT: {len(timeout)}")
    for r in ok:
        print(f"  OK   {r['script']}")
    for r in fail:
        print(f"  FAIL {r['script']} (exit={r['exit_code']})")
    for r in timeout:
        print(f"  TOUT {r['script']} ({r['elapsed_sec']}s)")

    report_path = ROOT / "tests" / "_manual_debug_run_report.txt"
    lines = ["One-script test run report", "=" * 72, ""]
    for r in results:
        lines.append(f"SCRIPT: {r['script']}")
        lines.append(f"EXIT: {r['exit_code']}  ELAPSED: {r['elapsed_sec']}s  TIMEOUT: {r['timed_out']}")
        lines.append("-" * 72)
        lines.append(r["output_tail"])
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFull log: {report_path}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
