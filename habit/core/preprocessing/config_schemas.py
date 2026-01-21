"""
Configuration Schemas for Image Preprocessing
Uses Pydantic for robust validation and type safety.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator, ConfigDict


class SaveOptionsConfig(BaseModel):
    """Options for saving intermediate preprocessing results."""

    save_intermediate: bool = False
    intermediate_steps: List[str] = Field(default_factory=list)


class PreprocessingStepConfig(BaseModel):
    """Generic preprocessing step config with required images list."""

    model_config = ConfigDict(extra="allow")

    images: List[str] = Field(default_factory=list)

    @validator("images")
    def images_required(cls, v):
        if not v:
            raise ValueError("images must not be empty")
        return v


class PreprocessingConfig(BaseModel):
    """Top-level preprocessing configuration."""

    model_config = ConfigDict(extra="allow")

    data_dir: str
    out_dir: str
    Preprocessing: Dict[str, PreprocessingStepConfig] = Field(default_factory=dict)
    save_options: SaveOptionsConfig = Field(default_factory=SaveOptionsConfig)
    processes: int = 1
    random_state: int = 42
    auto_select_first_file: bool = True

    @validator("data_dir", "out_dir")
    def path_required(cls, v, field):
        if not v or not str(v).strip():
            raise ValueError(f"{field.name} is required and cannot be empty")
        return v

    @validator("processes")
    def processes_non_negative(cls, v):
        if v is None:
            return 1
        if int(v) < 1:
            raise ValueError("processes must be >= 1")
        return int(v)


def validate_preprocessing_config(config_dict: Dict[str, Any]) -> PreprocessingConfig:
    """Validate raw dictionary against the schema."""
    return PreprocessingConfig(**config_dict)
