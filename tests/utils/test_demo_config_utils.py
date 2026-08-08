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
"""Tests for bundled demo-config packaging and ``copy_demo_config``."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli
from habit.utils.demo_config_utils import copy_demo_config, demo_config_root


_KEY_YAML = Path("habitat") / "config_habitat_two_step.yaml"


@pytest.mark.unit
def test_demo_config_root_contains_key_yaml() -> None:
    """Bundled resource tree must include the primary habitat demo YAML."""
    root: Path = demo_config_root()
    assert root.is_dir()
    assert (root / _KEY_YAML).is_file()


@pytest.mark.unit
def test_copy_demo_config_materializes_tree(tmp_path: Path) -> None:
    """``copy_demo_config`` writes ``<dest>/config/...`` without demo_data."""
    work: Path = tmp_path / "work"
    config_dir: Path = copy_demo_config(work, show_progress=False)
    assert config_dir == work / "config"
    assert (config_dir / _KEY_YAML).is_file()
    assert not (work / "demo_data").exists()


@pytest.mark.unit
def test_copy_demo_config_refuses_overwrite_by_default(tmp_path: Path) -> None:
    """Existing ``config/`` is protected unless ``overwrite=True``."""
    work: Path = tmp_path / "work"
    copy_demo_config(work, show_progress=False)
    with pytest.raises(FileExistsError):
        copy_demo_config(work, show_progress=False)
    again: Path = copy_demo_config(work, overwrite=True, show_progress=False)
    assert (again / _KEY_YAML).is_file()


@pytest.mark.unit
def test_copy_demo_config_cli(tmp_path: Path) -> None:
    """``habit copy-demo-config --dest`` creates the expected YAML path."""
    work: Path = tmp_path / "cli_work"
    runner = CliRunner()
    result = runner.invoke(cli, ["copy-demo-config", "--dest", str(work)])
    assert result.exit_code == 0, result.output
    assert (work / "config" / _KEY_YAML).is_file()


@pytest.mark.unit
def test_public_copy_demo_config_symbol() -> None:
    """``copy_demo_config`` is importable from the top-level ``habit`` package."""
    import habit

    assert callable(habit.copy_demo_config)
