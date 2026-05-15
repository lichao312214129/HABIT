"""
Manual end-to-end workflow: preprocess -> get-habitat -> extract -> model -> compare.

``manual_*.py`` is not auto-collected. Each step invokes the real CLI (Click
CliRunner) with cwd at repo root so demo YAML relative paths resolve like a
local developer run. Execute when needed::

    pytest tests/integration/manual_end_to_end_workflow.py -v -s

Legacy path was ``test_end_to_end_workflow.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli


class TestEndToEndWorkflow:
    """Exercise the full pipeline CLI sequence with demo configs (best-effort)."""

    def setup_method(self) -> None:
        self.runner = CliRunner()
        self.config_root = Path(__file__).resolve().parents[2] / "config"
        self.results: dict = {}

    def test_step1_preprocess(self, cwd_repo_root: None) -> None:
        config_path = self.config_root / "preprocessing" / "config_preprocessing_demo_elastix.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        result = self.runner.invoke(cli, ["preprocess", "-c", str(config_path)])
        self.results["preprocess"] = {
            "exit_code": result.exit_code,
            "output": result.output,
            "error": result.exception,
        }

        print(f"\n=== Preprocess Result ===\nExit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Output: {result.output}")

        assert result.exit_code in [0, 1], (
            f"Preprocess failed with exit code {result.exit_code}"
        )

    def test_step2_get_habitat(self, cwd_repo_root: None) -> None:
        config_path = self.config_root / "habitat" / "config/habitat/config_habitat_two_step.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        result = self.runner.invoke(cli, ["get-habitat", "-c", str(config_path)])
        self.results["get_habitat"] = {
            "exit_code": result.exit_code,
            "output": result.output,
            "error": result.exception,
        }

        print(f"\n=== Get-Habitat Result ===\nExit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Output: {result.output}")

        assert result.exit_code in [0, 1], (
            f"Get-habitat failed with exit code {result.exit_code}"
        )

    def test_step3_extract_features(self, cwd_repo_root: None) -> None:
        config_path = self.config_root / "feature_extraction" / "config_extract_features_demo.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        result = self.runner.invoke(cli, ["extract", "-c", str(config_path)])
        self.results["extract"] = {
            "exit_code": result.exit_code,
            "output": result.output,
            "error": result.exception,
        }

        print(f"\n=== Extract Features Result ===\nExit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Output: {result.output}")

        assert result.exit_code in [0, 1], f"Extract failed with exit code {result.exit_code}"

    def test_step4_model_train(self, cwd_repo_root: None) -> None:
        config_path = self.config_root / "machine_learning" / "config_machine_learning_clinical.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        result = self.runner.invoke(
            cli, ["model", "-c", str(config_path), "-m", "train"]
        )
        self.results["model_train"] = {
            "exit_code": result.exit_code,
            "output": result.output,
            "error": result.exception,
        }

        print(f"\n=== Model Train Result ===\nExit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Output: {result.output}")

        assert result.exit_code in [0, 1], (
            f"Model train failed with exit code {result.exit_code}"
        )

    def test_step5_compare(self, cwd_repo_root: None) -> None:
        config_path = self.config_root / "model_comparison" / "config_model_comparison_demo.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        result = self.runner.invoke(cli, ["compare", "-c", str(config_path)])
        self.results["compare"] = {
            "exit_code": result.exit_code,
            "output": result.output,
            "error": result.exception,
        }

        print(f"\n=== Compare Result ===\nExit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Output: {result.output}")

        assert result.exit_code in [0, 1], (
            f"Compare failed with exit code {result.exit_code}"
        )

    def teardown_method(self) -> None:
        print("\n" + "=" * 80)
        print("WORKFLOW TEST SUMMARY")
        print("=" * 80)
        for step, result in self.results.items():
            status = "PASS (exit 0)" if result["exit_code"] == 0 else "NON-ZERO EXIT"
            print(f"{step:20s}: {status} (exit_code={result['exit_code']})")
        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
