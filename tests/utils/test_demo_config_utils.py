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
"""Tests for demo-config single-source packaging and ``copy_demo_config``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli
from habit.utils.demo_config_utils import (
    copy_demo_config,
    demo_config_root,
    iter_demo_config_files,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPO_CONFIG = _PROJECT_ROOT / "config"
_KEY_YAML = Path("habitat") / "config_habitat_two_step.yaml"


@pytest.mark.unit
def test_demo_config_root_uses_repo_config_in_checkout() -> None:
    """Editable/source installs must read repo-root ``config/`` live."""
    root: Path = demo_config_root()
    assert root.resolve() == _REPO_CONFIG.resolve()
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


@pytest.mark.unit
def test_sync_script_mirrors_repo_config(tmp_path: Path) -> None:
    """Build helper must copy the same relative YAML set as repo ``config/``."""
    sync_script = _PROJECT_ROOT / "scripts" / "sync_demo_config.py"
    spec = importlib.util.spec_from_file_location("sync_demo_config", sync_script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dst = tmp_path / "demo_config"
    copied = module.sync_demo_config(src=_REPO_CONFIG, dst=dst)
    assert copied
    assert (dst / _KEY_YAML).is_file()

    # Relative paths from the sync helper match live iter_demo_config_files.
    live_rels = {
        path.relative_to(_REPO_CONFIG).as_posix()
        for path in iter_demo_config_files()
    }
    synced_rels = {rel.as_posix() for rel in copied}
    assert synced_rels == live_rels
