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
"""CLI smoke tests for ``habit view``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from click.testing import CliRunner

from habit.cli import cli
from habit.exceptions import OptionalDependencyError

pytestmark = pytest.mark.unit


def _write_nrrd(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    sitk.WriteImage(image, str(path))


def _tiny_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Write a tiny image + habitat NRRD pair under ``tmp_path``."""
    image = np.ones((6, 8, 8), dtype=np.float32)
    labels = np.zeros((6, 8, 8), dtype=np.int32)
    labels[2:4, 2:6, 2:6] = 1
    habitat_path = tmp_path / "subj001_habitats.nrrd"
    image_path = tmp_path / "subj001_T1.nrrd"
    _write_nrrd(habitat_path, labels)
    _write_nrrd(image_path, image)
    return image_path, habitat_path


def test_habit_view_positional_paths(tmp_path: Path, monkeypatch) -> None:
    """``habit view --backend matplotlib IMAGE HABITAT --no-open`` writes a PNG."""
    image_path, habitat_path = _tiny_pair(tmp_path)
    png_path = tmp_path / "overlay.png"

    monkeypatch.setattr(
        "habit.commands.cmd_view._open_with_system_viewer", lambda path: None
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "view",
            str(image_path),
            str(habitat_path),
            "--backend",
            "matplotlib",
            "--output",
            str(png_path),
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert png_path.is_file()
    assert "Habitat overlay preview" in result.output
    assert "LabelOverlay" in result.output
    assert "Backend" in result.output and "matplotlib" in result.output


def test_habit_view_flag_paths(tmp_path: Path, monkeypatch) -> None:
    """``--image`` / ``--habitat`` flags are accepted with matplotlib backend."""
    image = np.ones((4, 4, 4), dtype=np.float32)
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 2
    habitat_path = tmp_path / "map.nrrd"
    image_path = tmp_path / "img.nrrd"
    _write_nrrd(habitat_path, labels)
    _write_nrrd(image_path, image)

    monkeypatch.setattr(
        "habit.commands.cmd_view._open_with_system_viewer", lambda path: None
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "view",
            "--backend",
            "matplotlib",
            "--image",
            str(image_path),
            "--habitat",
            str(habitat_path),
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "map_overlay.png").is_file()


def test_habit_view_requires_both_paths() -> None:
    """Missing IMAGE/HABITAT exits with a usage error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["view"])
    assert result.exit_code != 0
    assert "IMAGE" in result.output or "image" in result.output.lower()


def test_habit_view_backend_help_prefers_napari() -> None:
    """``habit view --help`` documents auto default and napari preference."""
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "--help"])
    assert result.exit_code == 0, result.output
    assert "--backend" in result.output
    assert "auto" in result.output
    assert "napari" in result.output
    assert "matplotlib" in result.output
    assert "default: auto" in result.output.lower() or "default: auto" in result.output
    assert "--image" in result.output
    assert "multi" in result.output.lower() or "repeatable" in result.output.lower()
    assert "fall back" in result.output.lower() or "fallback" in result.output.lower()


def test_habit_view_backend_napari_no_open(tmp_path: Path, monkeypatch) -> None:
    """``--backend napari --no-open`` builds layers via habit.viz then exits."""
    image_path, habitat_path = _tiny_pair(tmp_path)

    calls: list[dict] = []

    class _Viewer:
        def close(self) -> None:
            calls.append({"closed": True})

    def _fake_view(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return _Viewer()

    # ``cmd_view`` does ``from habit.viz import view_habitat_napari`` at call time.
    monkeypatch.setattr("habit.viz.view_habitat_napari", _fake_view)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "view",
            str(image_path),
            str(habitat_path),
            "--backend",
            "napari",
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Backend" in result.output and "napari" in result.output
    assert any("kwargs" in c for c in calls)
    napari_call = next(c for c in calls if "kwargs" in c)
    assert napari_call["kwargs"].get("show") is False
    assert any(c.get("closed") for c in calls)


def test_habit_view_default_auto_uses_napari(tmp_path: Path, monkeypatch) -> None:
    """Default ``--backend auto`` prefers napari when the viewer is available."""
    image_path, habitat_path = _tiny_pair(tmp_path)

    calls: list[dict] = []

    class _Viewer:
        def close(self) -> None:
            calls.append({"closed": True})

    def _fake_view(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return _Viewer()

    monkeypatch.setattr("habit.viz.view_habitat_napari", _fake_view)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "view",
            str(image_path),
            str(habitat_path),
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Backend" in result.output and "napari" in result.output
    assert "fallback" not in result.output.lower()
    assert any("kwargs" in c for c in calls)
    assert any(c.get("closed") for c in calls)


def test_habit_view_auto_falls_back_when_napari_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """Missing napari prints install hint and writes a matplotlib PNG."""
    image_path, habitat_path = _tiny_pair(tmp_path)
    png_path = tmp_path / "fallback.png"

    def _boom(*args, **kwargs):
        raise OptionalDependencyError(
            'Missing optional dependency napari for interactive habitat viewing. '
            'Install with: pip install "habitat-analysis[view]"'
        )

    monkeypatch.setattr("habit.viz.view_habitat_napari", _boom)
    monkeypatch.setattr(
        "habit.commands.cmd_view._open_with_system_viewer", lambda path: None
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "view",
            str(image_path),
            str(habitat_path),
            "--backend",
            "auto",
            "--output",
            str(png_path),
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert png_path.is_file()
    assert "habitat-analysis[view]" in result.output
    assert "Falling back" in result.output or "fallback" in result.output.lower()
    assert "matplotlib" in result.output


def test_habit_view_napari_backend_also_falls_back(
    tmp_path: Path, monkeypatch
) -> None:
    """Explicit ``--backend napari`` still falls back when napari is missing."""
    image_path, habitat_path = _tiny_pair(tmp_path)

    def _boom(*args, **kwargs):
        raise OptionalDependencyError(
            'Missing optional dependency napari. '
            'Install with: pip install "habitat-analysis[view]"'
        )

    monkeypatch.setattr("habit.viz.view_habitat_napari", _boom)
    monkeypatch.setattr(
        "habit.commands.cmd_view._open_with_system_viewer", lambda path: None
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "view",
            str(image_path),
            str(habitat_path),
            "--backend",
            "napari",
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "subj001_habitats_overlay.png").is_file()
    assert "habitat-analysis[view]" in result.output
    assert "fallback" in result.output.lower()


def test_habit_view_napari_multi_image_flags(tmp_path: Path, monkeypatch) -> None:
    """Repeated ``--image`` + ``--habitat`` passes all arrays to napari."""
    shape = (4, 5, 5)
    labels = np.zeros(shape, dtype=np.int32)
    labels[1:3, 1:4, 1:4] = 1
    habitat_path = tmp_path / "map_habitats.nrrd"
    t1_path = tmp_path / "T1.nrrd"
    t2_path = tmp_path / "T2.nrrd"
    _write_nrrd(habitat_path, labels)
    _write_nrrd(t1_path, np.ones(shape, dtype=np.float32))
    _write_nrrd(t2_path, np.full(shape, 2.0, dtype=np.float32))

    calls: list[dict] = []

    class _Viewer:
        def close(self) -> None:
            calls.append({"closed": True})

    def _fake_view(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return _Viewer()

    monkeypatch.setattr("habit.viz.view_habitat_napari", _fake_view)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "view",
            "--backend",
            "napari",
            "--image",
            str(t1_path),
            "--image",
            str(t2_path),
            "--habitat",
            str(habitat_path),
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    napari_call = next(c for c in calls if "kwargs" in c)
    images_arg = napari_call["args"][0]
    assert isinstance(images_arg, list) and len(images_arg) == 2
    assert napari_call["kwargs"].get("image_names") == ["T1", "T2"]
    assert napari_call["kwargs"].get("show") is False


def test_habit_view_matplotlib_multi_image_warns(
    tmp_path: Path, monkeypatch
) -> None:
    """Matplotlib with multiple images uses the first and prints a note."""
    shape = (4, 4, 4)
    labels = np.zeros(shape, dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 1
    habitat_path = tmp_path / "map.nrrd"
    t1_path = tmp_path / "T1.nrrd"
    t2_path = tmp_path / "T2.nrrd"
    _write_nrrd(habitat_path, labels)
    _write_nrrd(t1_path, np.ones(shape, dtype=np.float32))
    _write_nrrd(t2_path, np.full(shape, 3.0, dtype=np.float32))
    png_path = tmp_path / "out.png"

    monkeypatch.setattr(
        "habit.commands.cmd_view._open_with_system_viewer", lambda path: None
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "view",
            "--backend",
            "matplotlib",
            str(t1_path),
            str(t2_path),
            str(habitat_path),
            "--output",
            str(png_path),
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert png_path.is_file()
    assert "first source image only" in result.output
    assert "napari" in result.output.lower()
