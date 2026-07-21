"""Verify a Windows HABIT installation with real native capability probes."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


Result = Dict[str, Any]


def _result(name: str, ok: bool, value: str, detail: str = "") -> Result:
    """Create one stable verification result for text and JSON renderers."""
    return {"name": name, "ok": ok, "value": value, "detail": detail}


def check_python() -> Result:
    """Require the CPython 3.10 ABI used by both bundled native wheels."""
    version = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info[:2] == (3, 10) and sys.maxsize > 2**32
    return _result("python", ok, version, "64-bit CPython 3.10 is required.")


def check_distribution(name: str, expected: str | None = None) -> Result:
    """Import package metadata and optionally compare its exact release."""
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        return _result(name, False, "missing", str(exc))
    ok = expected is None or version == expected
    detail = "" if ok else f"Expected {expected}, found {version}."
    return _result(name, ok, version, detail)


def check_import(module_name: str) -> Result:
    """Import a runtime module so missing DLLs are detected, not only metadata."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # Native DLL failures are regular exceptions on Windows.
        return _result(module_name, False, "import failed", repr(exc))
    return _result(module_name, True, str(getattr(module, "__version__", "ok")))


def check_habit_cli() -> Result:
    """Execute the installed console entry point through the active environment."""
    executable = shutil.which("habit")
    if not executable:
        return _result("habit-cli", False, "missing", "habit.exe is not on process PATH.")
    completed = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    return _result(
        "habit-cli",
        completed.returncode == 0,
        output or executable,
        "" if completed.returncode == 0 else f"Exit code {completed.returncode}.",
    )


def check_native_extension() -> Result:
    """Require HABIT's compiled supervoxel extension in distributable installs."""
    try:
        from habit.core.habitat_analysis.clustering_features.supervoxel_cext import (
            is_cext_available,
        )

        available = bool(is_cext_available())
    except Exception as exc:
        return _result("habit-cext", False, "import failed", repr(exc))
    return _result(
        "habit-cext",
        available,
        "native" if available else "fallback",
        "The release wheel must contain _sv_cmatrices.pyd.",
    )


def check_radiomics_preset() -> Result:
    """Confirm that non-editable wheels include the built-in voxel preset."""
    try:
        from habit.utils.radiomics_preset_utils import get_preset_path

        path = Path(get_preset_path("voxel")).resolve()
    except Exception as exc:
        return _result("radiomics-preset", False, "lookup failed", repr(exc))
    return _result("radiomics-preset", path.is_file(), str(path))


def check_pyradiomics_execution() -> Result:
    """Run a tiny first-order extraction across SimpleITK and PyRadiomics."""
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor

        image_array = np.arange(7 * 7 * 7, dtype=np.float32).reshape((7, 7, 7))
        mask_array = np.zeros((7, 7, 7), dtype=np.uint8)
        mask_array[2:5, 2:5, 2:5] = 1
        image = sitk.GetImageFromArray(image_array)
        mask = sitk.GetImageFromArray(mask_array)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("firstorder", ["Mean"])
        values = extractor.execute(image, mask)
        mean_value = float(values["original_firstorder_Mean"])
    except Exception as exc:
        return _result("pyradiomics-execution", False, "failed", repr(exc))
    return _result("pyradiomics-execution", np.isfinite(mean_value), str(mean_value))


def check_external_tool(name: str) -> Result:
    """Locate one staged native tool through the process-local PATH."""
    path = shutil.which(name)
    return _result(name, bool(path), path or "missing")


def check_gpu() -> Result:
    """Initialize CUDA and execute an actual tensor operation."""
    try:
        import torch

        if not torch.cuda.is_available():
            return _result("cuda", False, "unavailable", "torch.cuda.is_available() is false.")
        tensor = torch.tensor([1.0, 2.0], device="cuda")
        value = float((tensor * 2).sum().cpu().item())
        device = torch.cuda.get_device_name(0)
    except Exception as exc:
        return _result("cuda", False, "initialization failed", repr(exc))
    return _result("cuda", value == 6.0, device, f"Tensor result: {value}")


def check_optional_profile(profile: str) -> List[Result]:
    """Import every public dependency promised by an optional profile."""
    modules = {
        "automl": ["autogluon.tabular"],
        "analysis": ["krippendorff", "shap", "plotly", "lifelines"],
    }
    return [check_import(module_name) for module_name in modules[profile]]


def run_checks(require_gpu: bool, profile: str | None = None) -> List[Result]:
    """Run deterministic installation checks in dependency order."""
    results = [
        check_python(),
        check_distribution("HABIT"),
        check_distribution("pyradiomics", "3.0.1"),
        check_distribution("numpy", "1.26.1"),
        check_distribution("SimpleITK", "2.2.1"),
        check_distribution("pyarrow", "20.0.0"),
        check_import("habit"),
        check_import("radiomics"),
        check_import("ants"),
        check_import("pyarrow"),
        check_habit_cli(),
        check_native_extension(),
        check_radiomics_preset(),
        check_pyradiomics_execution(),
        check_external_tool("dcm2niix"),
        check_external_tool("elastix"),
        check_external_tool("transformix"),
    ]
    if require_gpu:
        results.append(check_gpu())
    if profile is not None:
        results.extend(check_optional_profile(profile))
    return results


def main() -> int:
    """Render verification results and return a process-friendly exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Require a working CUDA backend.")
    parser.add_argument(
        "--profile",
        choices=("automl", "analysis"),
        help="Require one optional feature profile.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()
    results = run_checks(require_gpu=args.gpu, profile=args.profile)
    failed = [result for result in results if not result["ok"]]
    if args.json:
        print(json.dumps({"results": results, "failed": len(failed)}, indent=2))
    else:
        for result in results:
            marker = "OK" if result["ok"] else "FAIL"
            print(f"[{marker:4}] {result['name']}: {result['value']}")
            if result["detail"]:
                print(f"       {result['detail']}")
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
