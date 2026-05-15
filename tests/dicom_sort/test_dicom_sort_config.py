"""
Unit tests for ``DicomSortConfig`` (standalone DICOM sort YAML).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from habit.core.common.configs.base import ConfigValidationError
from habit.core.dicom_sort.config_schema import DicomSortConfig


class TestDicomSortConfig:
    """Validation rules for ``DicomSortConfig``."""

    def _minimal(self, **extra: object) -> dict:
        base = {
            "data_dir": "/in",
            "out_dir": "/out",
            "f": "%n.dcm",
        }
        base.update(extra)
        return base

    def test_minimal_with_f(self) -> None:
        cfg = DicomSortConfig(**self._minimal())
        assert cfg.data_dir == "/in"
        assert cfg.out_dir == "/out"
        assert cfg.f == "%n.dcm"
        assert cfg.extra_args == []

    def test_deprecated_filename_format(self) -> None:
        cfg = DicomSortConfig(
            data_dir="/in",
            out_dir="/out",
            f=None,
            filename_format="%s_%d.dcm",
        )
        assert cfg.filename_format == "%s_%d.dcm"

    def test_missing_f_and_filename_format_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="dicom_sort: set `f`"):
            DicomSortConfig(data_dir="/in", out_dir="/out")

    def test_empty_data_dir_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            DicomSortConfig(data_dir="", out_dir="/out", f="%n.dcm")

    def test_from_file_resolves_only_path_keys_preserves_f_and_extra_args(
        self, tmp_path: Path
    ) -> None:
        """YAML directory must not prefix the dcm2niix ``-f`` string or ``extra_args`` tokens."""
        pattern = r"%n_%g_%x/%s_%d/%r_%o.dcm"
        yaml_path = tmp_path / "sort.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    "data_dir: ./indata",
                    "out_dir: ./outdata",
                    f'f: "{pattern}"',
                    'extra_args: ["-z", "i"]',
                    'dcm2niix_path: ./tool/dcm2niix.exe',
                ]
            ),
            encoding="utf-8",
        )
        (tmp_path / "indata").mkdir()
        (tmp_path / "tool").mkdir()
        exe = tmp_path / "tool" / "dcm2niix.exe"
        exe.write_bytes(b"")

        cfg = DicomSortConfig.from_file(yaml_path)
        assert cfg.f == pattern
        assert cfg.extra_args == ["-z", "i"]
        assert os.path.isabs(cfg.data_dir)
        assert os.path.isabs(cfg.out_dir)
        assert cfg.dcm2niix_path is not None
        assert os.path.isabs(cfg.dcm2niix_path)
