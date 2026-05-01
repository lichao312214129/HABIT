"""
ICC analysis configuration schema.

V1: this module is now a Pydantic ``BaseConfig`` subclass, in line with the
rest of the HABIT configuration system. The previous dict + ``validate_config``
style has been removed (no backward compatibility — V1 ground truth only).

Public surface:
    - :class:`ICCConfig`            : root config (Pydantic ``BaseConfig``).
    - :class:`ICCInputConfig`       : nested input section.
    - :class:`ICCOutputConfig`      : nested output section.
    - :func:`save_default_icc_config`: helper to dump an example YAML.
"""

import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from habit.core.common.config_base import BaseConfig


class ICCInputConfig(BaseModel):
    """
    Input section of the ICC configuration.

    Two mutually exclusive shapes:
        - ``type='files'`` with ``file_groups`` populated (each group is a
          list of file paths).
        - ``type='directories'`` with ``dir_list`` populated.
    """

    type: Literal['files', 'directories']
    file_groups: Optional[List[List[str]]] = None
    dir_list: Optional[List[str]] = None

    class Config:
        extra = 'forbid'

    @field_validator('file_groups', mode='before')
    @classmethod
    def _coerce_file_groups(cls, value: Any) -> Any:
        # Accept both [[a, b], [c, d]] and [a, b, c, d]; in the latter case
        # treat each scalar as a single-element group. Keeps backward
        # compatibility with older YAMLs at the *parser* level (still V1
        # idiomatic since the resulting Pydantic shape is canonical).
        if value is None:
            return None
        if isinstance(value, list):
            normalised: List[List[str]] = []
            for entry in value:
                if isinstance(entry, list):
                    normalised.append([str(p) for p in entry])
                else:
                    normalised.append([str(entry)])
            return normalised
        return value


class ICCOutputConfig(BaseModel):
    """Output section of the ICC configuration."""

    path: str

    class Config:
        extra = 'forbid'


class ICCConfig(BaseConfig):
    """
    Root configuration for the ICC (reliability) analysis.

    Loaded via :py:meth:`ICCConfig.from_file` from a YAML file. All field
    access is by attribute (``cfg.metrics`` instead of ``cfg["metrics"]``).
    """

    input: ICCInputConfig
    output: ICCOutputConfig

    metrics: Optional[List[str]] = None
    processes: Optional[int] = None
    full_results: bool = False
    selected_features: Optional[List[str]] = None
    debug: bool = False

    @model_validator(mode='after')
    def _validate_input_shape(self) -> 'ICCConfig':
        # Cross-field invariant: type ↔ payload presence.
        if self.input.type == 'files':
            if not self.input.file_groups:
                raise ValueError(
                    "ICCConfig: input.type='files' requires a non-empty "
                    "input.file_groups list."
                )
        else:  # 'directories'
            if not self.input.dir_list:
                raise ValueError(
                    "ICCConfig: input.type='directories' requires a "
                    "non-empty input.dir_list."
                )
        return self

    def parse_file_groups(self) -> List[List[str]]:
        """
        Return canonicalised file groups.

        Returns:
            List[List[str]]: list of file groups, where each group is a list
                of file paths.

        Raises:
            ValueError: if ``input.type`` is not ``'files'``.
        """
        if self.input.type != 'files':
            raise ValueError(
                "Cannot parse file groups when input.type is not 'files'."
            )
        return list(self.input.file_groups or [])

    def parse_directories(self) -> List[str]:
        """
        Return the directory list.

        Returns:
            List[str]: list of directory paths.

        Raises:
            ValueError: if ``input.type`` is not ``'directories'``.
        """
        if self.input.type != 'directories':
            raise ValueError(
                "Cannot parse directory list when input.type is not "
                "'directories'."
            )
        return list(self.input.dir_list or [])


def create_default_icc_config() -> dict:
    """Return a default ICC configuration as a plain dict (for YAML dump)."""
    return {
        "input": {
            "type": "files",
            "file_groups": [
                ["path/to/file1.csv", "path/to/file2.csv", "path/to/file3.csv"],
                ["path/to/file4.csv", "path/to/file5.csv", "path/to/file6.csv"],
            ],
        },
        "output": {
            "path": "icc_results.json",
        },
        # Metrics to calculate. Options:
        # - icc1, icc2, icc3, icc1k, icc2k, icc3k: Individual ICC types
        # - multi_icc: All 6 ICC types at once
        # - cohen_kappa, cohen: Cohen's Kappa (2 raters)
        # - fleiss_kappa, fleiss: Fleiss' Kappa (multiple raters)
        # - krippendorff: Krippendorff's Alpha
        "metrics": ["icc3"],
        "full_results": False,
        "selected_features": None,
        "processes": None,
        "debug": False,
    }


def save_default_icc_config(output_path: Union[str, Path]) -> None:
    """
    Save a default ICC configuration to a YAML file.

    Args:
        output_path: target file path (parent directories created if missing).
    """
    default_config = create_default_icc_config()

    parent = os.path.dirname(os.path.abspath(str(output_path)))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
