"""
Manual-style runner for individual CLI workflow steps (print-heavy diagnostics).

When imported by pytest, only ``run_cli_workflow_step`` is used by ``main()`` —
avoid naming helpers ``test_*`` so pytest does not collect them as tests.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from click.testing import CliRunner

from habit.cli import cli


def run_cli_workflow_step(
    step_name: str,
    command: list[str],
    config_path: Path,
    extra_args: list[str] | None = None,
) -> bool:
    """Run one CLI step; print diagnostics; return True if exit code is 0."""
    print(f"\n{'=' * 80}")
    print(f"Testing Step: {step_name}")
    print(f"Command: {' '.join(command)}")
    print(f"Config: {config_path}")
    print("=" * 80)

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False

    runner = CliRunner()
    cmd = [*command, "-c", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)

    result = runner.invoke(cli, cmd)

    print(f"Exit Code: {result.exit_code}")
    if result.exit_code == 0:
        print(f"{step_name} completed successfully")
        return True

    print(f"{step_name} failed")
    print("\nError Output:")
    print(result.output)
    if result.exception:
        print("\nException:")
        traceback.print_exception(
            type(result.exception),
            result.exception,
            result.exception.__traceback__,
        )
    return False


def main() -> int:
    """Run all workflow steps sequentially from demo YAML paths."""
    demo_data_dir = Path(__file__).resolve().parents[2] / "demo_data"
    results: dict[str, bool] = {}

    results["preprocess"] = run_cli_workflow_step(
        "Preprocess",
        ["preprocess"],
        demo_data_dir / "config_preprocessing.yaml",
    )

    results["get_habitat"] = run_cli_workflow_step(
        "Get Habitat",
        ["get-habitat"],
        demo_data_dir / "config_habitat.yaml",
    )

    results["extract"] = run_cli_workflow_step(
        "Extract Features",
        ["extract"],
        demo_data_dir / "config_extract_features.yaml",
    )

    results["model_train"] = run_cli_workflow_step(
        "Model Train",
        ["model"],
        demo_data_dir / "config_machine_learning_clinical.yaml",
        extra_args=["-m", "train"],
    )

    results["compare"] = run_cli_workflow_step(
        "Compare",
        ["compare"],
        demo_data_dir / "config_model_comparison.yaml",
    )

    print(f"\n{'=' * 80}")
    print("WORKFLOW TEST SUMMARY")
    print("=" * 80)
    for step, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"{step:20s}: {status}")
    print("=" * 80)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
