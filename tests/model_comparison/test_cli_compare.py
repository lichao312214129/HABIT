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
CLI-level tests for `habit compare` command (model comparison).

Heavy demo YAML run: ``tests/model_comparison/compare_models_multi.py``.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from habit.cli import cli


class TestCompareCLI:
    # ------------------------------------------------------------------
    # Help / meta
    # ------------------------------------------------------------------

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_compare(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert "compare" in result.output.lower() or "model" in result.output.lower()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0
