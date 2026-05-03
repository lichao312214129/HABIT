"""
Environment check script for the HABIT package.

Purpose:
    Quickly verify that the user's environment is ready to run any HABIT CLI
    command. The agent (Claude / Cursor / etc.) should call this BEFORE the
    first `habit ...` invocation in a session.

Usage:
    python check_environment.py
    python check_environment.py --json   # machine-readable output

Checks performed:
    1. Python version (>= 3.8 required, >= 3.10 needed for AutoGluon).
    2. Conda environment name (warns if not in a dedicated env).
    3. Whether the `habit` CLI is on PATH and importable.
    4. Whether key dependencies (SimpleITK, pandas, numpy, sklearn,
       PyYAML, click, ANTsPy, pyradiomics) are importable.
    5. Whether `dcm2niix` is on PATH (only warning if missing — only
       needed for DICOM conversion).

Exit codes:
    0 = all critical checks passed
    1 = critical failure (HABIT cannot run)
    2 = warnings only (HABIT can run but some optional deps missing)
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Tuple


# Critical packages — HABIT will not work without these
CRITICAL_PACKAGES: List[Tuple[str, str]] = [
    ("habit", "HABIT package itself"),
    ("click", "CLI framework"),
    ("yaml", "YAML config parser (PyYAML)"),
    ("numpy", "numerical arrays"),
    ("pandas", "data tables"),
    ("SimpleITK", "medical image I/O"),
    ("sklearn", "scikit-learn (clustering / ML)"),
]

# Optional packages — only needed for specific commands
OPTIONAL_PACKAGES: List[Tuple[str, str]] = [
    ("ants", "ANTsPy (image registration; needed for `habit preprocess` registration step)"),
    ("radiomics", "PyRadiomics (needed for radiomics feature extraction)"),
    ("xgboost", "XGBoost classifier (needed only if used in ML config)"),
    ("autogluon", "AutoGluon AutoML (needed only if used; requires Python 3.10)"),
    ("matplotlib", "plotting (needed for visualizations)"),
    ("seaborn", "statistical plotting"),
    ("scipy", "statistical tests (DeLong, ICC)"),
]


def check_python_version() -> Dict[str, Any]:
    """Check Python interpreter version against HABIT requirements."""
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major < 3 or (major == 3 and minor < 8):
        return {
            "name": "python_version",
            "status": "FAIL",
            "value": version_str,
            "message": "Python >= 3.8 required.",
        }
    if major == 3 and minor < 10:
        return {
            "name": "python_version",
            "status": "WARN",
            "value": version_str,
            "message": "Python 3.8/3.9 works for most things, but AutoGluon requires 3.10.",
        }
    return {
        "name": "python_version",
        "status": "OK",
        "value": version_str,
        "message": "",
    }


def check_conda_env() -> Dict[str, Any]:
    """Detect the active conda env (best-effort; warns if 'base' or none)."""
    env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if not env_name:
        return {
            "name": "conda_env",
            "status": "WARN",
            "value": "(none detected)",
            "message": "Not in a dedicated conda/virtualenv. Recommended: `conda create -n habit python=3.8`.",
        }
    if env_name == "base":
        return {
            "name": "conda_env",
            "status": "WARN",
            "value": env_name,
            "message": "Running in the conda 'base' env is discouraged. Create a dedicated 'habit' env.",
        }
    return {
        "name": "conda_env",
        "status": "OK",
        "value": env_name,
        "message": "",
    }


def check_habit_cli() -> Dict[str, Any]:
    """Try `habit --version` to confirm the CLI entrypoint is installed."""
    cli = shutil.which("habit")
    if cli is None:
        return {
            "name": "habit_cli",
            "status": "FAIL",
            "value": "(not found)",
            "message": "`habit` CLI not on PATH. Run `pip install -e .` from the HABIT repo root.",
        }
    try:
        out = subprocess.run(
            [cli, "--version"], capture_output=True, text=True, timeout=15
        )
        version_text = (out.stdout or out.stderr or "").strip().splitlines()[0] if (out.stdout or out.stderr) else ""
        return {
            "name": "habit_cli",
            "status": "OK",
            "value": version_text or cli,
            "message": "",
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "name": "habit_cli",
            "status": "FAIL",
            "value": cli,
            "message": f"`habit --version` failed: {exc}",
        }


def check_package(import_name: str, description: str, critical: bool) -> Dict[str, Any]:
    """Try to import a Python package and report version if available."""
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "(unknown version)")
        return {
            "name": f"package:{import_name}",
            "status": "OK",
            "value": str(version),
            "message": "",
        }
    except Exception as exc:
        status = "FAIL" if critical else "WARN"
        return {
            "name": f"package:{import_name}",
            "status": status,
            "value": "(missing)",
            "message": f"{description} — import failed: {exc.__class__.__name__}",
        }


def check_dcm2niix() -> Dict[str, Any]:
    """Check whether dcm2niix executable is on PATH (only needed for DICOM)."""
    path = shutil.which("dcm2niix")
    if path:
        return {
            "name": "dcm2niix",
            "status": "OK",
            "value": path,
            "message": "",
        }
    return {
        "name": "dcm2niix",
        "status": "WARN",
        "value": "(not on PATH)",
        "message": "Only needed for DICOM->NIfTI conversion. Pass full path via dcm2nii config if needed.",
    }


def render_text(results: List[Dict[str, Any]]) -> str:
    """Render results as a human-readable terminal report."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("HABIT environment check")
    lines.append("=" * 70)
    width = max(len(r["name"]) for r in results)
    for r in results:
        marker = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}.get(r["status"], "[ ?  ]")
        line = f"{marker}  {r['name']:<{width}}  {r['value']}"
        lines.append(line)
        if r["message"]:
            lines.append(f"        -> {r['message']}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    """Run all checks and print a summary; return process exit code."""
    parser = argparse.ArgumentParser(description="HABIT environment check")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    results.append(check_python_version())
    results.append(check_conda_env())
    results.append(check_habit_cli())
    for name, desc in CRITICAL_PACKAGES:
        results.append(check_package(name, desc, critical=True))
    for name, desc in OPTIONAL_PACKAGES:
        results.append(check_package(name, desc, critical=False))
    results.append(check_dcm2niix())

    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    warn_count = sum(1 for r in results if r["status"] == "WARN")

    if args.json:
        print(json.dumps({
            "results": results,
            "summary": {"fail": fail_count, "warn": warn_count, "ok": len(results) - fail_count - warn_count},
        }, indent=2))
    else:
        print(render_text(results))
        print(f"\nSummary: {fail_count} failures, {warn_count} warnings, {len(results) - fail_count - warn_count} ok")

    if fail_count > 0:
        return 1
    if warn_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
