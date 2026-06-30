#!/usr/bin/env python3
"""Full HABIT V1 E2E validation: CLI matrix + Python API + utility commands + pytest.

Uses demo data under F:\\work\\habit_project_v1 (WSL: /mnt/f/work/habit_project_v1).

Usage (repository root)::

    python scripts/run_full_e2e_validation.py
    python scripts/run_full_e2e_validation.py --skip-docker
    python scripts/run_full_e2e_validation.py --fast   # exclude slow imaging cases
    python scripts/run_full_e2e_validation.py --docker-only

All user-visible log messages are in English.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").exists():
        return root
    raise RuntimeError("Cannot locate repository root")


def _run_step(name: str, cmd: List[str], cwd: Path) -> int:
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print(f"CMD:  {' '.join(cmd)}")
    print("=" * 70)
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return proc.returncode


def _run_pytest(repo: Path, patterns: List[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", *patterns, "-q", "--tb=short"]
    return _run_step("pytest unit/integration", cmd, repo)


def _run_docker_validation(repo: Path) -> int:
    dockerfile = repo / "docker" / "Dockerfile"
    if not dockerfile.is_file():
        print("SKIP docker: docker/Dockerfile not found")
        return 0

    image = "habit:0.1.0-cpu"
    build_rc = _run_step(
        "docker build",
        [
            "docker", "build",
            "-f", str(dockerfile),
            "-t", image,
            str(repo),
        ],
        repo,
    )
    if build_rc != 0:
        return build_rc

    checks = [
        ("docker habit --version", ["docker", "run", "--rm", image, "--version"]),
        ("docker habit --help", ["docker", "run", "--rm", image, "--help"]),
        ("docker habit preprocess --help", ["docker", "run", "--rm", image, "preprocess", "--help"]),
        (
            "docker habit icc (mounted demo_data + config, writable output)",
            [
                "docker", "run", "--rm",
                "-v", f"{repo / 'demo_data'}:/app/demo_data",
                "-v", f"{repo / 'config'}:/app/config:ro",
                image,
                "icc", "-c", "/app/config/auxiliary/config_icc_demo.yaml",
            ],
        ),
    ]
    failed = 0
    for label, cmd in checks:
        rc = _run_step(label, cmd, repo)
        if rc != 0:
            failed += 1
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full HABIT V1 E2E validation.")
    parser.add_argument("--fast", action="store_true", help="Exclude slow imaging cases")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--docker-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    repo = _repo_root()
    matrix_script = repo / "demo_data" / "results_config_test" / "scripts" / "run_config_matrix.py"
    api_script = repo / "demo_data" / "results_config_test" / "scripts" / "run_api_matrix.py"
    utility_script = repo / "demo_data" / "results_config_test" / "scripts" / "run_utility_e2e.py"

    summary: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "steps": [],
    }
    overall_rc = 0

    def record(name: str, rc: int) -> None:
        nonlocal overall_rc
        summary["steps"].append({"name": name, "exit_code": rc})
        if rc != 0:
            overall_rc = 1

    if args.docker_only:
        record("docker", _run_docker_validation(repo))
    else:
        matrix_cmd = [sys.executable, str(matrix_script), f"--timeout={args.timeout}"]
        if not args.fast:
            matrix_cmd.append("--include-slow")
        record("cli_matrix", _run_step("CLI config matrix", matrix_cmd, repo))

        if not args.skip_api:
            api_cmd = [sys.executable, str(api_script), f"--timeout={args.timeout}"]
            if not args.fast:
                api_cmd.append("--include-slow")
            record("api_matrix", _run_step("Python API matrix", api_cmd, repo))

        record("utility_e2e", _run_step("Utility CLI", [sys.executable, str(utility_script)], repo))

        if not args.skip_pytest:
            pytest_targets = [
                "tests/preprocessing/test_cli_preprocess.py",
                "tests/integration/test_python_api.py",
                "tests/utils/test_cli_merge_csv.py",
                "tests/model_comparison/test_cli_compare.py",
            ]
            record("pytest", _run_pytest(repo, pytest_targets))

        if not args.skip_docker:
            record("docker", _run_docker_validation(repo))

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["overall_exit_code"] = overall_rc
    out = repo / "demo_data" / "results_config_test" / f"full_e2e_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"FULL E2E finished exit={overall_rc}")
    print(f"Summary written to: {out}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
