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
CLI-level tests for ``habit radiomics`` command (standalone radiomics pipeline entry).

Heavy demo invocation: ``tests/machine_learning/ml_radiomics_standalone.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "machine_learning"
    / "config_machine_learning_radiomics.yaml"
)


class TestRadiomicsCLI:
    """Smoke tests for ``habit radiomics``."""

    def test_radiomics_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["radiomics", "--help"])
        assert result.exit_code == 0
        assert "radiomics" in result.output.lower()


if __name__ == "__main__":
    sys.argv = ["habit", "radiomics", "-c", str(DEMO_CONFIG)]
    cli()
