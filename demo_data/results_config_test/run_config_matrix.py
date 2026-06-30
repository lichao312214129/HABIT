#!/usr/bin/env python3
"""Run HABIT CLI pipelines from matrix_manifest.yaml; each case writes to its own folder.

Usage (from repository root, conda env ``habit`` active)::

    python demo_data/results_config_test/scripts/run_config_matrix.py
    python demo_data/results_config_test/scripts/run_config_matrix.py --include-slow
    python demo_data/results_config_test/scripts/run_config_matrix.py --cases ml_radiomics,extract_graph
    python demo_data/results_config_test/scripts/run_config_matrix.py --dry-run

All user-visible log messages are in English.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    """Walk upward until pyproject.toml or setup.py is found."""
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "setup.py").exists():
            return candidate
    raise RuntimeError(f"Cannot locate HABIT repository root from {start}")


def _find_demo_root(repo_root: Path) -> Path:
    """Prefer F: drive demo_data (WSL mount) when present."""
    wsl_demo = Path("/mnt/f/work/habit_project_v1/demo_data")
    if wsl_demo.is_dir() and (wsl_demo / "ml_data").is_dir():
        return wsl_demo
    local_demo = repo_root / "demo_data"
    if local_demo.is_dir():
        return local_demo.resolve()
    raise RuntimeError("demo_data directory not found (checked /mnt/f/... and repo/demo_data)")


def _to_native_path(path: Path) -> str:
    """Return a string path suitable for YAML configs on the current OS."""
    return str(path.resolve())


def _manifest_path(script_path: Path) -> Path:
    """Return matrix_manifest.yaml next to the results_config_test folder."""
    return script_path.parents[1] / "matrix_manifest.yaml"


# ---------------------------------------------------------------------------
# Test case definitions (loaded from manifest)
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """One config-variant pipeline run."""

    case_id: str
    name: str
    command: str
    base_config: str
    output_keys: Tuple[str, ...]
    recipe_id: str = ""
    category: str = ""
    extra_args: Tuple[str, ...] = ()
    depends_on: Tuple[str, ...] = ()
    slow: bool = False
    skip_if: Optional[str] = None
    patch: Optional[str] = None
    overlay: Dict[str, Any] = field(default_factory=dict)


def _patch_output_path(cfg: Dict[str, Any], key_path: str, out_dir: Path) -> None:
    """Set a nested YAML key (dot-separated) to *out_dir*."""
    parts = key_path.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = _to_native_path(out_dir)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overlay* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# Skip predicates
# ---------------------------------------------------------------------------

def _skip_if_no_dicom(demo_root: Path, _: Path) -> Optional[str]:
    dicom_dir = demo_root / "dicom"
    if not dicom_dir.is_dir() or not any(dicom_dir.iterdir()):
        return "DICOM input directory missing or empty"
    return None


def _skip_if_no_preprocessed(demo_root: Path, _: Path) -> Optional[str]:
    images = demo_root / "preprocessed" / "processed_images" / "images"
    if not images.is_dir() or not any(images.iterdir()):
        return "Preprocessed images not found"
    return None


def _skip_if_no_resample02(_: Path, repo_root: Path) -> Optional[str]:
    resample = repo_root / ".cursor" / "test" / "resample_02"
    if not resample.is_dir():
        return "resample_02 test NIfTI folder missing"
    return None


def _skip_if_retest_nrrd_missing(demo_root: Path, _: Path) -> Optional[str]:
    retest_dir = demo_root / "habitat_maps" / "retest"
    if not retest_dir.is_dir() or not any(retest_dir.iterdir()):
        return "demo_data/habitat_maps/retest missing or empty"
    test_csv = demo_root / "ml_data" / "habitats_test.csv"
    if not test_csv.is_file():
        return "demo_data/ml_data/habitats_test.csv missing"
    return None


SKIP_REGISTRY: Dict[str, Callable[[Path, Path], Optional[str]]] = {
    "no_dicom": _skip_if_no_dicom,
    "no_preprocessed": _skip_if_no_preprocessed,
    "no_resample02": _skip_if_no_resample02,
    "no_retest_nrrd": _skip_if_retest_nrrd_missing,
}


# ---------------------------------------------------------------------------
# Patch functions (referenced by name in matrix_manifest.yaml)
# ---------------------------------------------------------------------------

def _patch_compare_predictions(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path]
) -> None:
    radiomics_csv = case_outputs["ml_radiomics"] / "all_prediction_results.csv"
    clinical_csv = case_outputs["ml_clinical"] / "all_prediction_results.csv"
    if not radiomics_csv.is_file() or not clinical_csv.is_file():
        return
    for entry in cfg.get("files_config", []):
        model = entry.get("model_name", "")
        if model == "radiomics":
            entry["path"] = _to_native_path(radiomics_csv)
        elif model == "clinical":
            entry["path"] = _to_native_path(clinical_csv)


def _patch_compare_single_radiomics(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path]
) -> None:
    radiomics_csv = case_outputs["ml_radiomics"] / "all_prediction_results.csv"
    if not radiomics_csv.is_file():
        return
    for entry in cfg.get("files_config", []):
        if entry.get("model_name") == "radiomics":
            entry["path"] = _to_native_path(radiomics_csv)


def _patch_extract_habitat_folder(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path]
) -> None:
    habitat_out = case_outputs.get("habitat_two_step_kmeans")
    if habitat_out is not None:
        cfg["habitats_map_folder"] = _to_native_path(habitat_out)
    cfg["raw_img_folder"] = _to_native_path(
        demo_root / "preprocessed" / "processed_images"
    )


def _patch_dicom_sort_dcm2niix(cfg: Dict[str, Any], demo_root: Path, _: Path) -> None:
    linux_bin = Path("/usr/bin/dcm2niix")
    if linux_bin.is_file():
        cfg["dcm2niix_path"] = _to_native_path(linux_bin)
    cfg["data_dir"] = _to_native_path(demo_root / "dicom")
    cfg["f"] = "subj_%n_%g_%x/%s_%d/%r_%o.dcm"


def _patch_ml_predict_pipeline(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path]
) -> None:
    clinical_out = case_outputs.get("ml_clinical")
    if clinical_out is None:
        return
    for name in ("LogisticRegression_final_pipeline.pkl", "SVM_final_pipeline.pkl"):
        pipeline = clinical_out / "models" / name
        if pipeline.is_file():
            cfg["pipeline_path"] = _to_native_path(pipeline)
            break
    cfg["output"] = _to_native_path(clinical_out / "predictions_cli")
    cfg["run_mode"] = "predict"


def _patch_habitat_predict_pipeline(
    cfg: Dict[str, Any],
    demo_root: Path,
    case_outputs: Dict[str, Path],
    train_case_id: str,
) -> None:
    train_out = case_outputs.get(train_case_id)
    if train_out is not None:
        pipeline = train_out / "habitat_pipeline.pkl"
        if pipeline.is_file():
            cfg["pipeline_path"] = _to_native_path(pipeline)
    cfg["data_dir"] = _to_native_path(demo_root / "preprocessed" / "processed_images")
    cfg["run_mode"] = "predict"


def _patch_habitat_predict_two_step(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path]
) -> None:
    _patch_habitat_predict_pipeline(cfg, demo_root, case_outputs, "habitat_two_step_kmeans")


def _patch_habitat_one_step_predict(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path]
) -> None:
    _patch_habitat_predict_pipeline(cfg, demo_root, case_outputs, "habitat_one_step_raw_concat")


def _patch_habitat_direct_pooling_predict(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path]
) -> None:
    _patch_habitat_predict_pipeline(cfg, demo_root, case_outputs, "habitat_direct_pooling")


def _patch_habitat_voxel_radiomics_predict(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path], repo_root: Path
) -> None:
    _patch_habitat_voxel_radiomics_demo(cfg, demo_root, repo_root)
    _patch_habitat_predict_pipeline(
        cfg, demo_root, case_outputs, "habitat_two_step_voxel_radiomics"
    )


def _patch_habitat_pooling_voxel_radiomics_predict_combined(
    cfg: Dict[str, Any], demo_root: Path, case_outputs: Dict[str, Path], repo_root: Path
) -> None:
    _patch_habitat_voxel_radiomics_demo(cfg, demo_root, repo_root)
    _patch_habitat_predict_pipeline(
        cfg, demo_root, case_outputs, "habitat_pooling_voxel_radiomics"
    )


def _patch_retest_demo_tables(
    cfg: Dict[str, Any], demo_root: Path, _: Path
) -> None:
    cfg["test_habitat_table"] = _to_native_path(demo_root / "ml_data" / "habitats_test.csv")
    cfg["retest_habitat_table"] = _to_native_path(
        demo_root / "ml_data" / "habitats_retest.csv"
    )


def _patch_radiomics_resample02(
    cfg: Dict[str, Any], demo_root: Path, repo_root: Path
) -> None:
    resample = repo_root / ".cursor" / "test" / "resample_02"
    if "paths" not in cfg:
        cfg["paths"] = {}
    cfg["paths"]["images_folder"] = _to_native_path(resample)
    params = repo_root / "config" / "radiomics" / "parameter.yaml"
    if params.is_file():
        cfg["paths"]["params_file"] = _to_native_path(params)


def _patch_icc_demo_metrics(cfg: Dict[str, Any], _: Path, __: Path) -> None:
    cfg["metrics"] = ["icc2", "icc3", "fleiss"]


def _patch_habitat_voxel_radiomics_demo(
    cfg: Dict[str, Any], demo_root: Path, repo_root: Path
) -> None:
    """Fix modality names and absolute radiomics params paths for demo data."""
    fc = cfg.get("FeatureConstruction") or {}
    for level_key in ("voxel_level", "supervoxel_level"):
        node = fc.get(level_key)
        if not node:
            continue
        method = node.get("method", "")
        if "T2" in method or "T1" in method:
            method = method.replace("T1", "delay2").replace("T2", "delay3")
            node["method"] = method
        params = node.setdefault("params", {})
        pf_hint = str(params.get("params_file", ""))
        if "supervoxel" in pf_hint or "supervoxel_radiomics" in method:
            params_file = repo_root / "config" / "radiomics" / "params_supervoxel_radiomics.yaml"
        elif params or "voxel_radiomics" in method or "radiomics" in pf_hint:
            params_file = repo_root / "config" / "radiomics" / "params_voxel_radiomics.yaml"
        else:
            continue
        if params_file.is_file():
            params["params_file"] = _to_native_path(params_file)


PATCH_REGISTRY: Dict[str, Callable[..., None]] = {
    "compare_predictions": _patch_compare_predictions,
    "compare_single_radiomics": _patch_compare_single_radiomics,
    "extract_habitat_folder": _patch_extract_habitat_folder,
    "dicom_sort_dcm2niix": _patch_dicom_sort_dcm2niix,
    "ml_predict_pipeline": _patch_ml_predict_pipeline,
    "habitat_predict_pipeline": _patch_habitat_predict_two_step,
    "habitat_one_step_predict_pipeline": _patch_habitat_one_step_predict,
    "habitat_direct_pooling_predict_pipeline": _patch_habitat_direct_pooling_predict,
    "habitat_voxel_radiomics_predict_pipeline": _patch_habitat_voxel_radiomics_predict,
    "habitat_pooling_voxel_radiomics_predict_pipeline": (
        _patch_habitat_pooling_voxel_radiomics_predict_combined
    ),
    "retest_demo_tables": _patch_retest_demo_tables,
    "radiomics_resample02": _patch_radiomics_resample02,
    "icc_demo_metrics": _patch_icc_demo_metrics,
    "habitat_voxel_radiomics_demo": _patch_habitat_voxel_radiomics_demo,
    "habitat_slic_demo_modalities": _patch_habitat_voxel_radiomics_demo,
}


def load_test_cases_from_manifest(manifest_file: Path) -> List[TestCase]:
    """Parse matrix_manifest.yaml into TestCase objects."""
    raw = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
    cases: List[TestCase] = []
    for entry in raw.get("cases", []):
        cases.append(
            TestCase(
                case_id=entry["case_id"],
                name=entry.get("name", entry["case_id"]),
                command=entry["command"],
                base_config=entry["base_config"],
                output_keys=tuple(entry.get("output_keys", ["output"])),
                recipe_id=entry.get("recipe_id", ""),
                category=entry.get("category", ""),
                extra_args=tuple(entry.get("extra_args", [])),
                depends_on=tuple(entry.get("depends_on", [])),
                slow=bool(entry.get("slow", False)),
                skip_if=entry.get("skip_if"),
                patch=entry.get("patch"),
                overlay=entry.get("overlay") or {},
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Config generation & execution
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML with encoding fallbacks (some habitat configs use legacy encodings)."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(encoding)
            return yaml.safe_load(text) or {}
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("yaml", b"", 0, 1, f"Cannot decode {path}")


def _absolutize_demo_paths(node: Any, demo_root: Path, repo_root: Path) -> Any:
    """Rewrite demo_data-relative strings to absolute paths for generated case configs."""
    if isinstance(node, dict):
        return {k: _absolutize_demo_paths(v, demo_root, repo_root) for k, v in node.items()}
    if isinstance(node, list):
        return [_absolutize_demo_paths(v, demo_root, repo_root) for v in node]
    if not isinstance(node, str):
        return node

    text = node.strip()
    demo_marker = "demo_data/"
    if demo_marker in text.replace("\\", "/"):
        normalized = text.replace("\\", "/")
        if normalized.startswith("../../demo_data/"):
            rel = normalized.split(demo_marker, 1)[1]
            return _to_native_path(demo_root / rel)
        if "F:/work/habit_project_v1/demo_data/" in normalized:
            rel = normalized.split(demo_marker, 1)[1]
            return _to_native_path(demo_root / rel)
        if normalized.startswith("demo_data/"):
            return _to_native_path(repo_root / normalized)
    if text.startswith("~/"):
        return _to_native_path(Path(text).expanduser())
    return node


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def prepare_case_config(
    case: TestCase,
    repo_root: Path,
    demo_root: Path,
    case_dir: Path,
    case_outputs: Dict[str, Path],
) -> Path:
    """Copy base config, patch output paths, write to case_dir/config.yaml."""
    base_path = repo_root / case.base_config
    cfg = load_yaml(base_path)
    if case.overlay:
        cfg = _deep_merge(cfg, case.overlay)

    out_dir = case_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in case.output_keys:
        if key in ("data_dir", "raw_img_folder"):
            if key == "data_dir":
                _patch_output_path(
                    cfg, key, demo_root / "preprocessed" / "processed_images"
                )
            continue
        if key == "output.path":
            _patch_output_path(cfg, key, out_dir / "icc_results.json")
            continue
        _patch_output_path(cfg, key, out_dir)

    if "data_dir" in cfg and case.command == "get-habitat":
        cfg["data_dir"] = _to_native_path(
            demo_root / "preprocessed" / "processed_images"
        )

    if case.patch:
        patch_fn = PATCH_REGISTRY.get(case.patch)
        if patch_fn is None:
            raise KeyError(f"Unknown patch '{case.patch}' for case {case.case_id}")
        if case.patch == "radiomics_resample02":
            patch_fn(cfg, demo_root, repo_root)
        elif case.patch in (
            "habitat_voxel_radiomics_demo",
            "habitat_slic_demo_modalities",
        ):
            patch_fn(cfg, demo_root, repo_root)
        elif case.patch in (
            "habitat_voxel_radiomics_predict_pipeline",
            "habitat_pooling_voxel_radiomics_predict_pipeline",
        ):
            patch_fn(cfg, demo_root, case_outputs, repo_root)
        else:
            patch_fn(cfg, demo_root, case_outputs)

    cfg = _absolutize_demo_paths(cfg, demo_root, repo_root)

    config_path = case_dir / "config.yaml"
    save_yaml(cfg, config_path)
    return config_path


def count_output_files(out_dir: Path) -> int:
    if not out_dir.is_dir():
        return 0
    return sum(1 for p in out_dir.rglob("*") if p.is_file())


def list_key_outputs(out_dir: Path, limit: int = 8) -> List[str]:
    if not out_dir.is_dir():
        return []
    files = sorted(out_dir.rglob("*"))
    files = [p for p in files if p.is_file()]
    rel = [str(p.relative_to(out_dir)) for p in files[:limit]]
    if len(files) > limit:
        rel.append(f"... (+{len(files) - limit} more files)")
    return rel


def run_habit_cli(
    repo_root: Path,
    command: str,
    config_path: Path,
    extra_args: Sequence[str],
    log_path: Path,
    timeout_sec: Optional[int],
) -> Tuple[int, float]:
    """Invoke ``habit <command> -c <config>`` and capture stdout/stderr."""
    cmd = [
        sys.executable,
        "-m",
        "habit.cli",
        command,
        "-c",
        str(config_path),
        *extra_args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    t0 = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"Command: {' '.join(cmd)}\n")
        log_fh.write(f"Started: {datetime.now(timezone.utc).isoformat()}\n\n")
        log_fh.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            log_fh.write(f"\nTIMEOUT after {timeout_sec}s\n")
            rc = 124
    elapsed = time.monotonic() - t0
    return rc, elapsed


def resolve_case_order(cases: List[TestCase]) -> List[TestCase]:
    """Topological sort by depends_on."""
    by_id = {c.case_id: c for c in cases}
    ordered: List[TestCase] = []
    seen: set[str] = set()

    def visit(cid: str) -> None:
        if cid in seen:
            return
        case = by_id[cid]
        for dep in case.depends_on:
            if dep in by_id:
                visit(dep)
        seen.add(cid)
        ordered.append(case)

    for c in cases:
        visit(c.case_id)
    return ordered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run HABIT config-variant matrix; each case uses a separate output folder."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Comma-separated case_id list (default: all eligible cases)",
    )
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow cases (habitat, preprocess, dicom_sort)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs only; do not invoke habit CLI",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-case timeout in seconds (default 1800)",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    results_root = script_path.parents[1]
    repo_root = _find_repo_root(script_path)
    demo_root = _find_demo_root(repo_root)
    cases_dir = results_root / "cases"
    logs_dir = results_root / "logs"
    cases_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = _manifest_path(script_path)
    all_cases = load_test_cases_from_manifest(manifest_file)

    if args.cases.strip():
        wanted = {x.strip() for x in args.cases.split(",") if x.strip()}
        all_cases = [c for c in all_cases if c.case_id in wanted]
        unknown = wanted - {c.case_id for c in all_cases}
        if unknown:
            print(f"WARNING: unknown case ids ignored: {sorted(unknown)}")

    if not args.include_slow:
        all_cases = [c for c in all_cases if not c.slow]

    all_cases = resolve_case_order(all_cases)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = logs_dir / f"matrix_run_{stamp}.log"
    summary: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_file),
        "repo_root": str(repo_root),
        "demo_root": str(demo_root),
        "results_root": str(results_root),
        "cases": [],
    }

    case_outputs: Dict[str, Path] = {}
    passed = failed = skipped = 0

    print(f"Manifest:     {manifest_file}")
    print(f"Results root: {results_root}")
    print(f"Demo data:    {demo_root}")
    print(f"Cases to run: {len(all_cases)}")
    print("-" * 60)

    with master_log.open("w", encoding="utf-8") as master_fh:
        master_fh.write(f"Matrix run {stamp}\n")
        master_fh.write(f"Manifest: {manifest_file}\n\n")

        for idx, case in enumerate(all_cases, 1):
            case_dir = cases_dir / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            out_dir = case_dir / "outputs"
            case_outputs[case.case_id] = out_dir

            record: Dict[str, Any] = {
                "case_id": case.case_id,
                "recipe_id": case.recipe_id,
                "category": case.category,
                "name": case.name,
                "command": case.command,
                "base_config": case.base_config,
                "case_dir": str(case_dir),
                "output_dir": str(out_dir),
                "status": "pending",
            }

            missing_deps = [
                d for d in case.depends_on
                if d not in case_outputs
                or not (case_outputs[d] / ".dep_ok").is_file()
            ]
            if missing_deps:
                record["status"] = "skipped"
                record["reason"] = f"missing failed/skipped dependencies: {missing_deps}"
                skipped += 1
                print(f"[SKIP] {case.case_id}: {record['reason']}")
                master_fh.write(f"SKIP {case.case_id}: {record['reason']}\n")
                summary["cases"].append(record)
                continue

            skip_reason: Optional[str] = None
            if case.skip_if:
                skip_fn = SKIP_REGISTRY.get(case.skip_if)
                if skip_fn is None:
                    raise KeyError(f"Unknown skip_if '{case.skip_if}' for {case.case_id}")
                skip_reason = skip_fn(demo_root, repo_root)

            if skip_reason:
                record["status"] = "skipped"
                record["reason"] = skip_reason
                skipped += 1
                print(f"[SKIP] {case.case_id}: {skip_reason}")
                master_fh.write(f"SKIP {case.case_id}: {skip_reason}\n")
                summary["cases"].append(record)
                continue

            config_path = prepare_case_config(
                case, repo_root, demo_root, case_dir, case_outputs
            )
            record["config"] = str(config_path)
            print(f"[RUN ] ({idx}/{len(all_cases)}) {case.case_id} -> {out_dir}")

            if args.dry_run:
                record["status"] = "dry_run"
                summary["cases"].append(record)
                continue

            run_log = case_dir / "run.log"
            rc, elapsed = run_habit_cli(
                repo_root,
                case.command,
                config_path,
                case.extra_args,
                run_log,
                timeout_sec=args.timeout,
            )
            n_files = count_output_files(out_dir)
            key_files = list_key_outputs(out_dir)

            record["exit_code"] = rc
            record["elapsed_sec"] = round(elapsed, 1)
            record["output_file_count"] = n_files
            record["key_outputs"] = key_files
            record["run_log"] = str(run_log)

            if rc == 0 and n_files > 0:
                record["status"] = "passed"
                passed += 1
                (out_dir / ".dep_ok").write_text("ok\n", encoding="utf-8")
                print(f"       PASS ({elapsed:.1f}s, {n_files} files)")
            elif rc == 0:
                record["status"] = "passed_empty"
                passed += 1
                (out_dir / ".dep_ok").write_text("ok\n", encoding="utf-8")
                print(f"       PASS with 0 output files ({elapsed:.1f}s)")
            else:
                record["status"] = "failed"
                failed += 1
                print(f"       FAIL exit={rc} ({elapsed:.1f}s, see {run_log})")

            master_fh.write(
                f"{record['status'].upper()} {case.case_id} rc={rc} files={n_files} "
                f"elapsed={elapsed:.1f}s\n"
            )
            summary["cases"].append(record)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["passed"] = passed
    summary["failed"] = failed
    summary["skipped"] = skipped

    summary_path = results_root / f"summary_{stamp}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    latest_path = results_root / "matrix_summary.json"
    with latest_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("-" * 60)
    print(f"Done: passed={passed} failed={failed} skipped={skipped}")
    print(f"Summary: {summary_path}")
    print(f"Latest:  {latest_path}")
    print(f"Master log: {master_log}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
