# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
CLI-level tests for the `habit preprocess` command.

Tests verify argument parsing, help output, and error handling for missing
config. Heavy demo-data runs: ``tests/preprocessing/preprocess_registration_elastix.py`` (not auto-collected).
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from habit.cli import cli


class TestPreprocessCLI:
    """Tests for `habit preprocess` command."""

    # ------------------------------------------------------------------
    # Help / meta
    # ------------------------------------------------------------------

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_preprocess(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert "preprocess" in result.output.lower()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "-c", "nonexistent_file.yaml"])
        assert result.exit_code != 0

    def test_no_config_arg_fails(self) -> None:
        """Invoking without -c should not silently succeed."""
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess"])
        # Should fail because -c is required, or show help
        assert result.exit_code != 0 or "help" in result.output.lower()
