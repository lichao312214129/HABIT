# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Pydantic schema for standalone DICOM sort (dcm2niix rename/reorganize).

This config is separate from ``PreprocessingConfig``; use ``habit sort-dicom``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator

from habit.core.common.configs.base import BaseConfig


# YAML fields resolved relative to the config file directory in ``from_file`` only.
# Defined at module scope so Pydantic does not treat it as a private model attribute.
DICOM_SORT_PATH_KEYS_FROM_YAML: tuple[str, ...] = (
    "data_dir",
    "out_dir",
    "output_dir",
    "dcm2niix_path",
)


class DicomSortConfig(BaseConfig):
    """
    One-shot dcm2niix sort: ``data_dir`` (input tree) and ``out_dir`` (output root).

    Key ``f`` is passed verbatim to dcm2niix ``-f``. Deprecated YAML alias: ``filename_format``.
    Optional ``output_dir`` overrides ``out_dir`` for the dcm2niix ``-o`` target (same as
    the legacy nested ``sort_dicom.output_dir``).
    """

    model_config = ConfigDict(extra="forbid")

    data_dir: str
    out_dir: str
    f: Optional[str] = None
    filename_format: Optional[str] = Field(
        default=None,
        description="Deprecated: same as ``f`` (dcm2niix -f pattern).",
    )
    dcm2niix_path: Optional[str] = None
    extra_args: List[str] = Field(default_factory=list)
    output_dir: Optional[str] = Field(
        default=None,
        description="If set, used as dcm2niix -o instead of out_dir.",
    )

    @field_validator("data_dir", "out_dir")
    @classmethod
    def path_required(cls, v: str, info: ValidationInfo) -> str:
        if not v or not str(v).strip():
            raise ValueError(f"{info.field_name} is required and cannot be empty")
        return v

    @staticmethod
    def _non_empty(s: Optional[str]) -> bool:
        return s is not None and str(s).strip() != ""

    @model_validator(mode="after")
    def require_f_or_legacy(self) -> "DicomSortConfig":
        """Ensure ``f`` or deprecated ``filename_format`` is provided."""
        if self._non_empty(self.f):
            return self
        if self._non_empty(self.filename_format):
            return self
        raise ValueError(
            "dicom_sort: set `f` to your dcm2niix -f pattern (deprecated alias: filename_format)."
        )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> DicomSortConfig:
        """
        Load a DICOM-sort YAML without global value-based path detection.

        ``BaseConfig.from_file`` uses ``resolve_config_paths``, which mis-treats dcm2niix ``-f``
        patterns (e.g. ``%n_%g_%x/%s_%d/%r_%o.dcm``) as relative file paths. This override loads
        with ``resolve_paths=False`` and resolves only ``data_dir``, ``out_dir``, ``output_dir``,
        and ``dcm2niix_path`` relative to the config directory.

        Args:
            config_path: Path to the YAML / JSON configuration file.

        Returns:
            DicomSortConfig: Validated configuration instance.
        """
        from habit.core.common.configs.loader import PathResolver, load_config

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        raw: Dict[str, Any] = load_config(str(path), resolve_paths=False)
        resolver = PathResolver(config_path=path)
        for key in DICOM_SORT_PATH_KEYS_FROM_YAML:
            if key not in raw:
                continue
            val = raw[key]
            if isinstance(val, str) and val.strip():
                raw[key] = resolver.resolve_path(val)

        return cls.from_dict(raw, config_path=str(path))
