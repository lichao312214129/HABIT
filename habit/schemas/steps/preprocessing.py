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
Pydantic parameter schemas for preprocessing pipeline steps.

These models are the single source of truth for GUI forms, YAML validation hints,
and future CLI schema export. Pipeline-injected keys (``images``, ``keys``) are
omitted — they are filled automatically at run time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

# Interpolation modes supported by ``ResamplePreprocessor.interp_map``.
RESAMPLE_IMG_MODES: Tuple[str, ...] = (
    "nearest",
    "linear",
    "bilinear",
    "bspline",
    "bicubic",
    "gaussian",
    "lanczos",
    "hamming",
    "cosine",
    "welch",
    "blackman",
)

RESAMPLE_PADDING_MODES: Tuple[str, ...] = ("border", "zeros", "reflect", "symmetric")

REGISTRATION_BACKENDS: Tuple[str, ...] = ("ants", "simpleitk", "elastix")
REGISTRATION_TRANSFORMS: Tuple[str, ...] = (
    "SyN",
    "SyNRA",
    "Rigid",
    "Affine",
    "Translation",
    "TRSAA",
    "BSplineSyN",
)
REGISTRATION_METRICS: Tuple[str, ...] = ("MI", "CC", "MeanSquares", "MattesMutualInformation")

REORIENTATION_MODES: Tuple[str, ...] = ("closest", "strict")
COMMON_ORIENTATIONS: Tuple[str, ...] = ("LPS", "RAS", "LIA", "RPI")

DCM2NIIX_MERGE_SLICES: Tuple[str, ...] = ("y", "1", "2", "n", "0")


class LoadImageParams(BaseModel):
    """``load_image`` has no user-facing parameters (modalities injected as ``images``)."""

    model_config = ConfigDict(extra="forbid")


class N4CorrectionParams(BaseModel):
    """N4 bias field correction parameters."""

    num_fitting_levels: int = Field(
        4,
        ge=1,
        le=6,
        description="Number of fitting levels for bias field correction.",
        json_schema_extra={"group": "Parameters", "order": 1, "widget": "slider"},
    )
    num_iterations: List[int] = Field(
        default_factory=lambda: [50, 50, 50, 50],
        description="Iterations per fitting level (comma-separated list).",
        json_schema_extra={"group": "Parameters", "order": 2},
    )
    convergence_threshold: float = Field(
        0.001,
        gt=0.0,
        description="Convergence threshold for the correction.",
        json_schema_extra={"group": "Parameters", "order": 3},
    )
    shrink_factor: int = Field(
        4,
        ge=1,
        description="Shrink factor to accelerate computation.",
        json_schema_extra={"group": "Parameters", "order": 4},
    )
    mask_name: Optional[str] = Field(
        None,
        description=(
            "Explicit Subject ROI name used to restrict N4 estimation. "
            "When omitted, N4 estimates the bias field from the full image."
        ),
        json_schema_extra={"group": "Parameters", "order": 5},
    )


class ResampleParams(BaseModel):
    """Resample images to a target voxel spacing."""

    target_spacing: Tuple[float, float, float] = Field(
        (1.0, 1.0, 1.0),
        description="Target spacing in mm (x, y, z).",
        json_schema_extra={"group": "Parameters", "order": 1, "widget": "spacing"},
    )
    img_mode: Literal[
        "nearest",
        "linear",
        "bilinear",
        "bspline",
        "bicubic",
        "gaussian",
        "lanczos",
        "hamming",
        "cosine",
        "welch",
        "blackman",
    ] = Field(
        "bilinear",
        description="Interpolation mode for image data.",
        json_schema_extra={"group": "Parameters", "order": 2},
    )
    padding_mode: Literal["border", "zeros", "reflect", "symmetric"] = Field(
        "border",
        description="Padding mode for out-of-bound values.",
        json_schema_extra={"group": "Parameters", "order": 3},
    )
    align_corners: bool = Field(
        False,
        description="Whether to align corners when resampling.",
        json_schema_extra={"group": "Parameters", "order": 4},
    )


class RegistrationParams(BaseModel):
    """Image registration parameters (ANTs, SimpleITK, or elastix)."""

    fixed_image: str = Field(
        ...,
        description="Reference modality key to register moving images to.",
        json_schema_extra={"group": "General", "order": 1},
    )
    backend: Literal["ants", "simpleitk", "elastix"] = Field(
        "ants",
        description="Registration backend.",
        json_schema_extra={"group": "General", "order": 2},
    )
    type_of_transform: Literal[
        "SyN", "SyNRA", "Rigid", "Affine", "Translation", "TRSAA", "BSplineSyN"
    ] = Field(
        "SyN",
        description="Transform type (ANTs/SimpleITK naming).",
        json_schema_extra={"group": "General", "order": 3},
    )
    metric: Literal["MI", "CC", "MeanSquares", "MattesMutualInformation"] = Field(
        "MI",
        description="Similarity metric.",
        json_schema_extra={"group": "General", "order": 4},
    )
    optimizer: Optional[str] = Field(
        None,
        description="Optional optimizer hint (SimpleITK: 'lbfgs' selects LBFGSB).",
        json_schema_extra={
            "group": "General",
            "order": 5,
            "choices": ["gradient_descent", "lbfgs"],
            "visible_if": {"backend": "simpleitk"},
        },
    )
    use_mask: bool = Field(
        False,
        description="Use a mask during registration.",
        json_schema_extra={"group": "Mask", "order": 1},
    )
    replace_by_fixed_image_mask: bool = Field(
        True,
        description="Replace moving mask with fixed-image mask after registration.",
        json_schema_extra={"group": "Mask", "order": 2},
    )
    mask_key: str = Field(
        "",
        description="Mask key when use_mask is enabled.",
        json_schema_extra={"group": "Mask", "order": 3, "visible_if": {"use_mask": True}},
    )
    elastix_parameter_files: Optional[str] = Field(
        None,
        description="Elastix parameter file (.txt). Use Browse to select.",
        json_schema_extra={
            "group": "Elastix",
            "order": 1,
            "widget": "path_file",
            "visible_if": {"backend": "elastix"},
        },
    )
    elastix_path: Optional[str] = Field(
        None,
        description="Path to elastix executable (optional).",
        json_schema_extra={
            "group": "Elastix",
            "order": 2,
            "widget": "path_file",
            "visible_if": {"backend": "elastix"},
        },
    )
    transformix_path: Optional[str] = Field(
        None,
        description="Path to transformix executable (optional).",
        json_schema_extra={
            "group": "Elastix",
            "order": 3,
            "widget": "path_file",
            "visible_if": {"backend": "elastix"},
        },
    )
    elastix_threads: int = Field(
        0,
        ge=0,
        description="Elastix thread count (0 = default).",
        json_schema_extra={"group": "Elastix", "order": 4, "visible_if": {"backend": "elastix"}},
    )
    elastix_parameter_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional elastix parameter overrides (YAML dict).",
        json_schema_extra={"group": "Elastix", "order": 5, "visible_if": {"backend": "elastix"}},
    )
    number_of_histogram_bins: int = Field(
        50,
        ge=1,
        description="Histogram bins for SimpleITK registration.",
        json_schema_extra={"group": "SimpleITK", "order": 1, "visible_if": {"backend": "simpleitk"}},
    )
    metric_sampling_percentage: float = Field(
        0.2,
        gt=0.0,
        le=1.0,
        description="Metric sampling percentage (SimpleITK).",
        json_schema_extra={"group": "SimpleITK", "order": 2, "visible_if": {"backend": "simpleitk"}},
    )
    learning_rate: float = Field(
        1.0,
        gt=0.0,
        description="Optimizer learning rate (SimpleITK).",
        json_schema_extra={"group": "SimpleITK", "order": 3, "visible_if": {"backend": "simpleitk"}},
    )
    number_of_iterations: int = Field(
        200,
        ge=1,
        description="Maximum optimizer iterations (SimpleITK).",
        json_schema_extra={"group": "SimpleITK", "order": 4, "visible_if": {"backend": "simpleitk"}},
    )
    shrink_factors_per_level: Optional[str] = Field(
        "4, 2, 1",
        description="Shrink factors per pyramid level (comma-separated).",
        json_schema_extra={"group": "SimpleITK", "order": 5, "visible_if": {"backend": "simpleitk"}},
    )
    smoothing_sigmas_per_level: Optional[str] = Field(
        "2.1, 1.0, 0.0",
        description="Smoothing sigmas per level (comma-separated).",
        json_schema_extra={"group": "SimpleITK", "order": 6, "visible_if": {"backend": "simpleitk"}},
    )
    bspline_mesh_size: int = Field(
        8,
        ge=1,
        description="BSpline mesh size (SimpleITK deformable registration).",
        json_schema_extra={"group": "SimpleITK", "order": 7, "visible_if": {"backend": "simpleitk"}},
    )
    bspline_order: int = Field(
        3,
        ge=1,
        description="BSpline order (SimpleITK).",
        json_schema_extra={"group": "SimpleITK", "order": 8, "visible_if": {"backend": "simpleitk"}},
    )


class ZScoreNormalizationParams(BaseModel):
    """Z-score intensity normalization."""

    only_inmask: bool = Field(
        False,
        description="Compute statistics only within the mask.",
        json_schema_extra={"group": "Parameters", "order": 1},
    )
    mask_key: Optional[str] = Field(
        None,
        description="Mask key when only_inmask is True.",
        json_schema_extra={"group": "Parameters", "order": 2, "visible_if": {"only_inmask": True}},
    )
    clip_values: Optional[Tuple[float, float]] = Field(
        None,
        description="Optional (min, max) clip range after normalization, e.g. (-3, 3).",
        json_schema_extra={"group": "Parameters", "order": 3},
    )


class HistogramStandardizationParams(BaseModel):
    """Nyúl histogram standardization."""

    percentiles: List[float] = Field(
        default_factory=lambda: [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0],
        description="Percentile landmarks for standardization.",
        json_schema_extra={"group": "Parameters", "order": 1},
    )
    target_min: float = Field(
        0.0,
        description="Target minimum intensity after standardization.",
        json_schema_extra={"group": "Parameters", "order": 2},
    )
    target_max: float = Field(
        100.0,
        description="Target maximum intensity after standardization.",
        json_schema_extra={"group": "Parameters", "order": 3},
    )
    mask_key: Optional[str] = Field(
        None,
        description="Optional mask key for masked standardization.",
        json_schema_extra={"group": "Parameters", "order": 4},
    )


class AdaptiveHistogramEqualizationParams(BaseModel):
    """Adaptive histogram equalization (CLAHE-style)."""

    alpha: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Alpha parameter for contrast limiting.",
        json_schema_extra={"group": "Parameters", "order": 1, "widget": "slider"},
    )
    beta: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Beta parameter for contrast limiting.",
        json_schema_extra={"group": "Parameters", "order": 2, "widget": "slider"},
    )
    radius: Union[int, Tuple[int, int, int]] = Field(
        5,
        description="Region radius (single int or x,y,z tuple).",
        json_schema_extra={"group": "Parameters", "order": 3},
    )


class ReorientationParams(BaseModel):
    """Reorient images to a canonical orientation."""

    target_orientation: Literal["LPS", "RAS", "LIA", "RPI"] = Field(
        "LPS",
        description="Target orientation code (e.g. LPS, RAS).",
        json_schema_extra={"group": "Parameters", "order": 1},
    )
    mode: Literal["closest", "strict"] = Field(
        "closest",
        description="closest = axis flip/permutation only; strict = interpolated resampling.",
        json_schema_extra={"group": "Parameters", "order": 2},
    )


class Dcm2niiParams(BaseModel):
    """DICOM to NIfTI conversion via dcm2niix."""

    dcm2niix_path: Optional[str] = Field(
        None,
        description="Path to dcm2niix executable (optional).",
        json_schema_extra={"group": "Paths", "order": 1, "widget": "path_file"},
    )
    filename_format: Optional[str] = Field(
        None,
        description="dcm2niix -f filename pattern.",
        json_schema_extra={"group": "Naming", "order": 1},
    )
    adjacent_dicoms: bool = Field(True, json_schema_extra={"group": "Options", "order": 1})
    compress: bool = Field(True, json_schema_extra={"group": "Options", "order": 2})
    anonymize: bool = Field(False, json_schema_extra={"group": "Options", "order": 3})
    ignore_derived: bool = Field(False, json_schema_extra={"group": "Options", "order": 4})
    crop_images: bool = Field(False, json_schema_extra={"group": "Options", "order": 5})
    generate_json: bool = Field(
        False,
        description="Generate BIDS JSON sidecar files.",
        json_schema_extra={"group": "Options", "order": 6},
    )
    verbose: bool = Field(False, json_schema_extra={"group": "Options", "order": 7})
    batch_mode: bool = Field(True, json_schema_extra={"group": "Options", "order": 8})
    merge_slices: Optional[Literal["y", "1", "2", "n", "0"]] = Field(
        "2",
        description='Merge mode: "2"=by series (recommended), "n"/"0"=no merge.',
        json_schema_extra={"group": "Options", "order": 9},
    )
    single_file_mode: Optional[bool] = Field(
        None,
        description="True=force single output file; False=allow multiple; None=default.",
        json_schema_extra={"group": "Options", "order": 10},
    )


class CustomPreprocessorParams(BaseModel):
    """Plugin custom preprocessor — arbitrary extra keys allowed."""

    model_config = ConfigDict(extra="allow")


PREPROCESSING_PARAM_MODELS: Dict[str, type[BaseModel]] = {
    "load_image": LoadImageParams,
    "n4_correction": N4CorrectionParams,
    "resample": ResampleParams,
    "registration": RegistrationParams,
    "zscore_normalization": ZScoreNormalizationParams,
    "histogram_standardization": HistogramStandardizationParams,
    "adaptive_histogram_equalization": AdaptiveHistogramEqualizationParams,
    "reorientation": ReorientationParams,
    "dcm2nii": Dcm2niiParams,
    "custom_preprocessor": CustomPreprocessorParams,
}
