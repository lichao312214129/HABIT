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
"""Unit tests for :func:`habit.datasets.fetch_demo` (local zip, no GitHub)."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli
from habit.datasets import fetch_demo, inspect_preprocessed_root
from habit.datasets.demo import format_preprocessed_inventory
from habit.exceptions import DataFormatError


def _write_tiny_demo_zip(path: Path) -> str:
    """Write a minimal official-shaped zip and return its SHA-256 hex."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "preprocessed/images/subj001/LAP/demo_lap.nrrd",
            b"fake-nrrd",
        )
        archive.writestr(
            "preprocessed/masks/subj001/LAP/demo_roi.nrrd",
            b"fake-mask",
        )
        archive.writestr(
            "preprocessed/images/subj002/LAP/demo_lap.nrrd",
            b"fake-nrrd-2",
        )
        archive.writestr(
            "preprocessed/masks/subj002/LAP/demo_roi.nrrd",
            b"fake-mask-2",
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


@pytest.mark.unit
def test_inspect_preprocessed_root_lists_subjects(tmp_path: Path) -> None:
    """Directory listing reports subjects, series, and one example file."""
    image = tmp_path / "images" / "subj001" / "LAP" / "img.nrrd"
    mask = tmp_path / "masks" / "subj001" / "LAP" / "roi.nrrd"
    image.parent.mkdir(parents=True)
    mask.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    mask.write_bytes(b"y")
    info = inspect_preprocessed_root(tmp_path)
    assert info.subjects == ("subj001",)
    assert info.image_keys == ("LAP",)
    assert info.mask_keys == ("LAP",)
    assert info.example_image == "images/subj001/LAP/img.nrrd"
    report = format_preprocessed_inventory(info)
    assert "Your own data must use the same folder tree" in report
    assert "cohort_from_directory" in report
    assert str(tmp_path.resolve()) in report


@pytest.mark.unit
def test_inspect_rejects_missing_images(tmp_path: Path) -> None:
    """A random folder is not a HABIT preprocessed root."""
    with pytest.raises(DataFormatError, match="missing images/"):
        inspect_preprocessed_root(tmp_path)


@pytest.mark.unit
def test_fetch_demo_downloads_once_and_prints_layout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """First call extracts the zip; second call is a cache hit."""
    zip_path = tmp_path / "preprocessed.zip"
    digest = _write_tiny_demo_zip(zip_path)
    data_home = tmp_path / "habit_data"
    root = fetch_demo(
        data_home=data_home,
        url=zip_path.as_uri(),
        sha256=digest,
        verbose=True,
    )
    assert root == (data_home / "demo-data-v1" / "preprocessed").resolve()
    assert (root / "images" / "subj001" / "LAP" / "demo_lap.nrrd").is_file()
    printed = capsys.readouterr().out
    assert "HABIT demo data (downloaded)" in printed
    assert "subjects (2): subj001, subj002" in printed
    assert "Your own data must use the same folder tree" in printed
    assert str(root) in printed

    root_again = fetch_demo(
        data_home=data_home,
        url="http://127.0.0.1/should-not-be-fetched",
        sha256=digest,
        verbose=True,
    )
    assert root_again == root
    cached_out = capsys.readouterr().out
    assert "HABIT demo data (cached)" in cached_out


@pytest.mark.unit
def test_fetch_demo_rejects_bad_checksum(tmp_path: Path) -> None:
    """A checksum mismatch deletes the zip and raises."""
    zip_path = tmp_path / "preprocessed.zip"
    _write_tiny_demo_zip(zip_path)
    with pytest.raises(DataFormatError, match="SHA-256 mismatch"):
        fetch_demo(
            data_home=tmp_path / "habit_data",
            url=zip_path.as_uri(),
            sha256="0" * 64,
            verbose=False,
        )


@pytest.mark.unit
def test_cli_fetch_demo_help() -> None:
    """``habit fetch-demo --help`` describes the download-once flow."""
    result = CliRunner().invoke(cli, ["fetch-demo", "--help"])
    assert result.exit_code == 0
    assert "Download the official preprocessed demo pack" in result.output
