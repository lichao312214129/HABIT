"""
End-to-end workflow integration tests: lightweight CLI smoke only.

Heavy sequential demo steps: ``manual_integration_sequential_cli.py``.
Full class-based workflow with printed summary: ``manual_end_to_end_workflow.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest
from click.testing import CliRunner

from habit.cli import cli


def _invoke(
    args: List[str], config: Optional[Path] = None, extra: Optional[List[str]] = None
) -> int:
    """Run CLI command and return exit code."""
    cmd = args[:]
    if config is not None:
        cmd += ["-c", str(config)]
    if extra:
        cmd += extra
    runner = CliRunner()
    result = runner.invoke(cli, cmd)
    return result.exit_code


# ---------------------------------------------------------------------------
# Individual CLI step smoke tests
# ---------------------------------------------------------------------------


class TestCLISmoke:
    """Smoke tests: each command must handle --help without error."""

    COMMANDS = [
        "preprocess",
        "get-habitat",
        "extract",
        "model",
        "cv",
        "compare",
        "dicom-info",
        "merge-csv",
    ]

    @pytest.mark.parametrize("cmd", COMMANDS)
    def test_help_exits_zero(self, cmd: str) -> None:
        exit_code = _invoke([cmd, "--help"])
        assert exit_code == 0
