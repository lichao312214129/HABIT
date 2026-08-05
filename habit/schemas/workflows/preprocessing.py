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
"""
Workflow configuration schema for image preprocessing.

Canonical location for preprocessing YAML root models. Step-level parameters
are validated against ``habit.core.schemas.steps.preprocessing`` via
:class:`~habit.core.schemas.validation.validate_step_params`.
"""

from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from habit.schemas.base import BaseConfig
from habit.schemas.validation import validate_step_params


class SaveOptionsConfig(BaseModel):
    """Options for saving intermediate preprocessing results."""

    save_intermediate: bool = Field(
        default=False,
        description="Save intermediate images after each step",
        json_schema_extra={"group": "Save", "order": 1},
    )
    intermediate_steps: List[str] = Field(
        default_factory=list,
        description="Steps to save intermediates for (empty = all enabled steps)",
        json_schema_extra={"group": "Save", "order": 2},
    )


class PreprocessingStepConfig(BaseModel):
    """Generic preprocessing step config; ``images`` lists modality keys to process."""

    model_config = ConfigDict(extra="allow")

    __pydantic_extra__: dict[str, Any]

    images: List[str] = Field(default_factory=list)

    @field_validator("images")
    def normalize_images(cls, v: List[str]) -> List[str]:
        return list(v)


class PreprocessingConfig(BaseConfig):
    """Top-level preprocessing configuration."""

    model_config = ConfigDict(extra="allow")

    __pydantic_extra__: dict[str, Any]

    data_dir: str = Field(
        ...,
        description="Input directory containing subject images",
        json_schema_extra={"group": "Paths", "widget": "path_dir", "order": 1},
    )
    out_dir: str = Field(
        ...,
        description="Output directory for preprocessed images",
        json_schema_extra={"group": "Paths", "widget": "path_dir", "order": 2},
    )
    preprocessing: Dict[str, PreprocessingStepConfig] = Field(
        default_factory=dict,
        description="Preprocessing pipeline steps (keyed by step name)",
        json_schema_extra={"group": "Pipeline", "order": 1},
    )
    save_options: SaveOptionsConfig = Field(
        default_factory=SaveOptionsConfig,
        description="Options for saving intermediate results",
        json_schema_extra={"group": "Advanced", "order": 1},
    )
    processes: int = Field(
        1,
        description="Number of parallel processes",
        json_schema_extra={"group": "Advanced", "order": 2},
    )
    random_state: int = Field(
        42,
        description="Random seed for reproducibility",
        json_schema_extra={"group": "Advanced", "order": 3},
    )
    auto_select_first_file: bool = Field(
        True,
        description="Auto-select first file when multiple files found per modality",
        json_schema_extra={"group": "Advanced", "order": 4},
    )
    # habit_default: data_dir/images/<subject>/<modality>/ (optional masks/ tree).
    preprocessing_input_layout: Literal["habit_default"] = Field(
        "habit_default",
        description="Input directory layout pattern",
        json_schema_extra={"group": "Advanced", "order": 5},
    )

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

    @model_validator(mode="before")
    @classmethod
    def _coerce_preprocessing_step_params(cls, data: Any) -> Any:
        """Validate each ``Preprocessing`` block against registered *Params models."""
        if not isinstance(data, dict):
            return data
        prep = data.get("preprocessing")
        if not isinstance(prep, dict):
            return data
        out = dict(data)
        validated: Dict[str, Any] = {}
        for step_type, block in prep.items():
            if isinstance(block, dict):
                validated[step_type] = validate_step_params(
                    "preprocessing",
                    step_type,
                    block,
                    preserve_keys=frozenset({"images"}),
                )
            else:
                validated[step_type] = block
        out["preprocessing"] = validated
        return out


def validate_preprocessing_config(config_dict: Dict[str, Any]) -> PreprocessingConfig:
    """Validate raw dictionary against the schema."""
    return PreprocessingConfig(**config_dict)
