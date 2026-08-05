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
"""CLI commands must not import v0.1 engine config schema modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_COMMANDS_ROOT = Path(__file__).resolve().parents[2] / "habit" / "commands"

_FORBIDDEN_PREFIXES = (
    "habit.compat.engines.habitat_analysis.config_schemas",
    "habit.compat.engines.machine_learning.config_schemas",
)


def _module_level_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


@pytest.mark.unit
def test_command_modules_avoid_legacy_config_schema_imports() -> None:
    """The five migrated commands load schemas from ``habit.schemas`` only."""
    offenders: list[str] = []
    for path in sorted(_COMMANDS_ROOT.glob("cmd_*.py")):
        for imported in _module_level_imports(path):
            if any(imported == prefix or imported.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES):
                offenders.append(f"{path.name} imports {imported}")
    assert not offenders, offenders
