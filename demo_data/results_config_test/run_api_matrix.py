#!/usr/bin/env python3
"""Run the same config matrix through HABIT Python API entry points (not CLI).

Usage (repository root, conda env ``habit`` active)::

    python demo_data/results_config_test/scripts/run_api_matrix.py
    python demo_data/results_config_test/scripts/run_api_matrix.py --cases ml_radiomics,icc_demo
    python demo_data/results_config_test/scripts/run_api_matrix.py --include-slow

Reuses case definitions and generated configs from ``run_config_matrix.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import shared matrix helpers from sibling script.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import run_config_matrix as matrix  # noqa: E402


def _load_config(command: str, config_path: Path) -> Any:
    """Load typed config for a matrix case command."""
    if command in ("model", "cv"):
        from habit.core.machine_learning.config_schemas import MLConfig
        return MLConfig.from_file(str(config_path))
    if command == "compare":
        from habit.core.machine_learning.config_schemas import ModelComparisonConfig
        return ModelComparisonConfig.from_file(str(config_path))
    if command == "get-habitat":
        from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
        return HabitatAnalysisConfig.from_file(str(config_path))
    if command == "extract":
        from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig
        return FeatureExtractionConfig.from_file(str(config_path))
    if command == "preprocess":
        from habit.core.preprocessing.config_schemas import PreprocessingConfig
        return PreprocessingConfig.from_file(str(config_path))
    if command == "sort-dicom":
        from habit.core.dicom_sort import DicomSortConfig
        return DicomSortConfig.from_file(str(config_path))
    if command == "icc":
        from habit.core.machine_learning.feature_selectors.icc.config import ICCConfig
        return ICCConfig.from_file(str(config_path))
    if command == "radiomics":
        from habit.core.habitat_analysis.config_schemas import RadiomicsConfig
        return RadiomicsConfig.from_file(str(config_path))
    if command == "retest":
        from habit.core.machine_learning.config_schemas import TestRetestConfig
        return TestRetestConfig.from_file(str(config_path))
    raise ValueError(f"No API loader for command: {command}")


def _run_api_case(
    case: matrix.TestCase,
    config_path: Path,
    out_dir: Path,
    extra_args: Tuple[str, ...],
) -> None:
    """Dispatch one matrix case through the programmatic API."""
    from habit.utils.log_utils import setup_logger, stop_queue_listener

    command = case.command
    config = _load_config(command, config_path)
    log_name = f"api.{command.replace('-', '_')}"
    logger = setup_logger(
        name=log_name,
        output_dir=out_dir,
        log_filename="api_run.log",
        level=logging.INFO,
    )

    try:
        if command == "preprocess":
            from habit.core.preprocessing.run import run_preprocess_from_config
            run_preprocess_from_config(config, logger=logger)
        elif command == "sort-dicom":
            from habit.core.dicom_sort import run_dicom_sort
            run_dicom_sort(config, logger=logger)
        elif command == "get-habitat":
            from habit.core.habitat_analysis.run import run_habitat_analysis_from_config
            run_habitat_analysis_from_config(config, logger=logger, output_dir=str(out_dir))
        elif command == "extract":
            from habit.core.habitat_analysis.run import run_feature_extraction_from_config
            run_feature_extraction_from_config(config, logger=logger)
        elif command == "model":
            from habit.core.machine_learning.run import (
                apply_ml_mode_override,
                run_ml_from_config,
            )
            mode = "train"
            if "-m" in extra_args:
                idx = extra_args.index("-m")
                if idx + 1 < len(extra_args):
                    mode = extra_args[idx + 1]
            config = apply_ml_mode_override(config, mode)
            run_ml_from_config(config, logger=logger, output_dir=str(out_dir))
        elif command == "cv":
            from habit.core.machine_learning.run import run_kfold_from_config
            run_kfold_from_config(config, logger=logger, output_dir=str(out_dir))
        elif command == "compare":
            from habit.core.machine_learning.run import run_model_comparison_from_config
            run_model_comparison_from_config(config, logger=logger, output_dir=str(out_dir))
        elif command == "icc":
            from habit.core.machine_learning.feature_selectors.icc.icc import (
                run_icc_analysis_from_config,
            )
            run_icc_analysis_from_config(config)
        elif command == "radiomics":
            from habit.core.habitat_analysis.run import run_radiomics_from_config
            run_radiomics_from_config(config, logger=logger, output_dir=str(out_dir))
        elif command == "retest":
            from habit.core.habitat_analysis.configurator import HabitatConfigurator
            from habit.core.machine_learning.feature_selectors.icc.habitat_test_retest_mapper import (
                batch_process_files,
                configure_logging,
                find_habitat_mapping,
            )
            configurator = HabitatConfigurator(
                config=config, logger=logger, output_dir=str(out_dir)
            )
            cfg = configurator.create_test_retest_analyzer()
            configure_logging(out_dir, cfg.debug)
            habitat_mapping = find_habitat_mapping(
                cfg.test_habitat_table,
                cfg.retest_habitat_table,
                cfg.features,
                cfg.similarity_method,
            )
            batch_process_files(cfg.input_dir, habitat_mapping, cfg.out_dir, cfg.processes)
        else:
            raise ValueError(f"No API runner for command: {command}")
    finally:
        stop_queue_listener()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run HABIT config matrix via Python API entry points."
    )
    parser.add_argument("--cases", type=str, default="", help="Comma-separated case ids")
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--reuse-cli-configs",
        action="store_true",
        help="Use configs already generated under cases/<id>/config.yaml",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    results_root = script_path.parents[1]
    repo_root = matrix._find_repo_root(script_path)
    demo_root = matrix._find_demo_root(repo_root)
    cases_dir = results_root / "cases"
    api_cases_dir = results_root / "api_cases"
    logs_dir = results_root / "logs"
    api_cases_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_cases = matrix.build_test_cases()
    if args.cases.strip():
        wanted = {x.strip() for x in args.cases.split(",") if x.strip()}
        all_cases = [c for c in all_cases if c.case_id in wanted]
    if not args.include_slow:
        all_cases = [c for c in all_cases if not c.slow]
    all_cases = matrix.resolve_case_order(all_cases)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "python_api",
        "repo_root": str(repo_root),
        "demo_root": str(demo_root),
        "cases": [],
    }

    case_outputs: Dict[str, Path] = {}
    passed = failed = skipped = 0

    print(f"API matrix: {len(all_cases)} cases")
    print(f"Demo data:  {demo_root}")
    print("-" * 60)

    for idx, case in enumerate(all_cases, 1):
        case_dir = api_cases_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        out_dir = case_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        case_outputs[case.case_id] = out_dir

        record: Dict[str, Any] = {"case_id": case.case_id, "command": case.command}

        missing_deps = [
            d for d in case.depends_on
            if d not in case_outputs or not (case_outputs[d] / ".dep_ok").is_file()
        ]
        if missing_deps:
            record["status"] = "skipped"
            record["reason"] = f"missing dependencies: {missing_deps}"
            skipped += 1
            print(f"[SKIP] {case.case_id}: {record['reason']}")
            summary["cases"].append(record)
            continue

        skip_reason = case.skip_if(demo_root, repo_root) if case.skip_if else None
        if skip_reason:
            record["status"] = "skipped"
            record["reason"] = skip_reason
            skipped += 1
            print(f"[SKIP] {case.case_id}: {skip_reason}")
            summary["cases"].append(record)
            continue

        cli_case_dir = cases_dir / case.case_id
        if args.reuse_cli_configs and (cli_case_dir / "config.yaml").is_file():
            config_path = cli_case_dir / "config.yaml"
        else:
            config_path = matrix.prepare_case_config(
                case, repo_root, demo_root, case_dir, case_outputs
            )

        run_log = case_dir / "api_run.log"
        print(f"[API ] ({idx}/{len(all_cases)}) {case.case_id}")
        t0 = time.monotonic()
        rc = 0
        err_text = ""
        try:
            _run_api_case(case, config_path, out_dir, case.extra_args)
        except Exception as exc:  # noqa: BLE001
            rc = 1
            err_text = traceback.format_exc()
            run_log.write_text(err_text, encoding="utf-8")
        elapsed = time.monotonic() - t0

        n_files = matrix.count_output_files(out_dir)
        record.update(
            {
                "config": str(config_path),
                "exit_code": rc,
                "elapsed_sec": round(elapsed, 1),
                "output_file_count": n_files,
                "run_log": str(run_log),
            }
        )

        if rc == 0:
            record["status"] = "passed"
            passed += 1
            (out_dir / ".dep_ok").write_text("ok\n", encoding="utf-8")
            print(f"       PASS ({elapsed:.1f}s, {n_files} files)")
        else:
            record["status"] = "failed"
            failed += 1
            print(f"       FAIL ({elapsed:.1f}s) see {run_log}")

        summary["cases"].append(record)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["passed"] = passed
    summary["failed"] = failed
    summary["skipped"] = skipped
    summary_path = results_root / f"api_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 60)
    print(f"API done: passed={passed} failed={failed} skipped={skipped}")
    print(f"Summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    repo = matrix._find_repo_root(Path(__file__).resolve())
    sys.path.insert(0, str(repo))
    raise SystemExit(main())
