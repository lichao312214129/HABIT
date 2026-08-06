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
"""Low-level, geometry-aware radiomics feature extraction API.

This module complements the YAML ``run_radiomics`` workflow with a direct
component API suitable for notebooks and third-party pipelines.  It currently
uses PyRadiomics as the explicit backend; other backends must implement the
same :class:`FeatureResult` contract before they are exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError, OptionalDependencyError, ProcessingError
from habit.api.image import (
    GeometryPolicy,
    GeometryReport,
    ImageInput,
    ImageMaskPair,
    MaskInput,
    MaskVolume,
    _coerce_image,
    _coerce_mask,
    align_image_mask,
)

__all__ = [
    "FeatureResult",
    "FeatureTableResult",
    "extract_features",
    "extract_batch",
]

RadiomicsParams = Union[Mapping[str, Any], str, Path, None]


@dataclass(frozen=True)
class FeatureResult:
    """Structured output from radiomics extraction for one image/mask pair."""

    values: Mapping[str, float]
    label: int
    backend: str
    geometry_report: GeometryReport
    resolved_params: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    failed_features: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze mappings and ensure callers receive numeric feature values."""
        normalized_values = {}
        for name, value in self.values.items():
            scalar = np.asarray(value)
            if scalar.ndim != 0:
                raise HABITAPIError(
                    f"Feature '{name}' is not scalar and cannot enter FeatureResult."
                )
            normalized_values[str(name)] = float(scalar)
        object.__setattr__(self, "values", MappingProxyType(normalized_values))
        object.__setattr__(
            self, "resolved_params", MappingProxyType(dict(self.resolved_params))
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(
            self, "failed_features", MappingProxyType(dict(self.failed_features))
        )
        object.__setattr__(
            self, "warnings", tuple(str(value) for value in self.warnings)
        )


@dataclass(frozen=True)
class FeatureTableResult:
    """Structured batch output with successful feature rows and failures."""

    table: pd.DataFrame
    results: Tuple[FeatureResult, ...]
    failures: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy mutable tabular and mapping values at the public boundary."""
        object.__setattr__(self, "table", self.table.copy())
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))


def _create_pyradiomics_extractor(params: RadiomicsParams) -> Any:
    """Create PyRadiomics lazily so base HABIT imports remain lightweight."""
    try:
        from habit.utils.radiomics_params_utils import (
            create_radiomics_feature_extractor,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "radiomics":
            from habit.utils.optional_deps import pyradiomics_install_hint

            raise OptionalDependencyError(pyradiomics_install_hint()) from exc
        raise
    if isinstance(params, Mapping):
        return create_radiomics_feature_extractor(dict(params))
    return create_radiomics_feature_extractor(params)


def _resolved_params(params: RadiomicsParams) -> Mapping[str, Any]:
    """Return serializable parameter provenance without reading optional modules."""
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return dict(params)
    return {"params_file": str(Path(params))}


def _split_extractor_output(
    output: Mapping[str, Any]
) -> Tuple[dict[str, float], dict[str, Any]]:
    """Separate scalar features from PyRadiomics diagnostic provenance fields."""
    values: dict[str, float] = {}
    provenance: dict[str, Any] = {}
    for name, value in output.items():
        if str(name).startswith("diagnostics_"):
            provenance[str(name)] = str(value)
            continue
        scalar = np.asarray(value)
        if scalar.ndim == 0 and np.issubdtype(scalar.dtype, np.number):
            values[str(name)] = float(scalar)
        else:
            provenance[f"non_scalar_output:{name}"] = str(value)
    return values, provenance


def extract_features(
    image: ImageInput,
    mask: MaskInput,
    params: RadiomicsParams = None,
    *,
    label: int = 1,
    backend: str = "pyradiomics",
    geometry_policy: GeometryPolicy = GeometryPolicy.STRICT,
    geometry_tolerance: float = 1e-6,
) -> FeatureResult:
    """Extract segment-based radiomics features from one image/mask pair.

    Args:
        image: Image data as an :class:`ImageVolume`, NumPy array, SimpleITK image,
            or supported image file path.
        mask: Segmentation data as a :class:`MaskVolume`, NumPy array, SimpleITK
            image, or supported mask file path.
        params: PyRadiomics parameter mapping or YAML parameter-file path. ``None``
            selects PyRadiomics defaults.
        label: Positive mask label identifying the region of interest.
        backend: Explicit extraction backend. Only ``"pyradiomics"`` is currently
            implemented by the stable low-level API.
        geometry_policy: How to handle image/mask geometry mismatch.
        geometry_tolerance: Absolute tolerance used by geometry validation.

    Returns:
        A structured result containing scalar feature values, diagnostics, resolved
        parameters, and the geometry action applied before extraction.

    Raises:
        HABITAPIError: If the inputs or requested label violate the public contract.
        OptionalDependencyError: If PyRadiomics is not installed.
        ProcessingError: If the backend cannot extract the requested region.
    """
    if label <= 0:
        raise HABITAPIError("label must be a positive non-background integer.")
    if backend != "pyradiomics":
        raise HABITAPIError(
            f"Unsupported backend '{backend}'. The stable low-level API currently "
            "supports only 'pyradiomics'."
        )

    image_volume = _coerce_image(image)
    mask_volume = _coerce_mask(mask)
    if label not in mask_volume.labels:
        raise HABITAPIError(
            f"Mask does not contain requested label {label}; available labels: "
            f"{list(mask_volume.labels)}."
        )
    pair = align_image_mask(
        ImageMaskPair(image_volume, mask_volume),
        policy=geometry_policy,
        tolerance=geometry_tolerance,
    )

    try:
        extractor = _create_pyradiomics_extractor(params)
        output = extractor.execute(
            imageFilepath=pair.image.to_sitk(),
            maskFilepath=pair.mask.to_sitk(),
            label=label,
        )
    except OptionalDependencyError:
        raise
    except Exception as exc:
        raise ProcessingError(
            f"PyRadiomics extraction failed for label {label}: {exc}"
        ) from exc

    values, provenance = _split_extractor_output(output)
    if not values:
        raise ProcessingError(
            "PyRadiomics completed without scalar feature values. Inspect "
            "FeatureResult.provenance for diagnostic output."
        )
    return FeatureResult(
        values=values,
        label=label,
        backend=backend,
        geometry_report=pair.geometry_report
        or GeometryReport(compatible=True, tolerance=geometry_tolerance),
        resolved_params=_resolved_params(params),
        provenance=provenance,
    )


def extract_batch(
    pairs: Sequence[ImageMaskPair],
    params: RadiomicsParams = None,
    *,
    label: int = 1,
    backend: str = "pyradiomics",
    geometry_policy: GeometryPolicy = GeometryPolicy.STRICT,
    geometry_tolerance: float = 1e-6,
    fail_fast: bool = True,
) -> FeatureTableResult:
    """Extract radiomics features deterministically for an ordered image/mask batch.

    ``extract_batch`` is intentionally sequential in its first stable release.
    Process isolation, GPU scheduling, and retry semantics belong to the workflow
    layer and will be introduced only with one shared structured failure contract.
    """
    rows: list[dict[str, Any]] = []
    results: list[FeatureResult] = []
    failures: dict[str, str] = {}
    for index, pair in enumerate(pairs):
        subject_id = pair.image.subject_id or pair.mask.subject_id or f"row_{index}"
        if subject_id in failures or any(
            row.get("subject_id") == subject_id for row in rows
        ):
            subject_id = f"{subject_id}_{index}"
        try:
            result = extract_features(
                pair.image,
                pair.mask,
                params,
                label=label,
                backend=backend,
                geometry_policy=geometry_policy,
                geometry_tolerance=geometry_tolerance,
            )
        except Exception as exc:
            if fail_fast:
                raise
            failures[subject_id] = str(exc)
            continue
        rows.append({"subject_id": subject_id, **result.values})
        results.append(result)
    return FeatureTableResult(
        table=pd.DataFrame(rows),
        results=tuple(results),
        failures=failures,
    )
