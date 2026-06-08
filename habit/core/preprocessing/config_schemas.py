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
Configuration Schemas for Image Preprocessing
Uses Pydantic for robust validation and type safety.
"""

from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

from habit.core.common.configs.base import BaseConfig


class SaveOptionsConfig(BaseModel):
    """Options for saving intermediate preprocessing results."""

    save_intermediate: bool = False
    intermediate_steps: List[str] = Field(default_factory=list)


class PreprocessingStepConfig(BaseModel):
    """Generic preprocessing step config; ``images`` lists modality keys to process."""

    model_config = ConfigDict(extra="allow")

    images: List[str] = Field(default_factory=list)

    @field_validator("images")
    def normalize_images(cls, v: List[str]) -> List[str]:
        return list(v)


class PreprocessingConfig(BaseConfig):
    """Top-level preprocessing configuration."""

    model_config = ConfigDict(extra="allow")

    data_dir: str
    out_dir: str
    Preprocessing: Dict[str, PreprocessingStepConfig] = Field(default_factory=dict)
    save_options: SaveOptionsConfig = Field(default_factory=SaveOptionsConfig)
    processes: int = 1
    random_state: int = 42
    auto_select_first_file: bool = True
    # habit_default: data_dir/images/<subject>/<modality>/ (optional masks/ tree).
    preprocessing_input_layout: Literal["habit_default"] = "habit_default"

    @field_validator("data_dir", "out_dir")
    def path_required(cls, v: str, info) -> str:
        if not v or not str(v).strip():
            raise ValueError(f"{info.field_name} is required and cannot be empty")
        return v

    @field_validator("processes")
    def processes_non_negative(cls, v: Optional[int]) -> int:
        if v is None:
            return 1
        if int(v) < 1:
            raise ValueError("processes must be >= 1")
        return int(v)


def validate_preprocessing_config(config_dict: Dict[str, Any]) -> PreprocessingConfig:
    """Validate raw dictionary against the schema."""
    return PreprocessingConfig(**config_dict)
