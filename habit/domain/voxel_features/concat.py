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
"""Concatenation of several voxel feature families into one field."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.domain.voxel_features._base import build_voxel_field, roi_voxels
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = ["ConcatVoxelFeatures", "ConcatVoxelFeaturesParams"]


class ConcatVoxelFeaturesParams(BaseModel):
    """Constructor parameters for :class:`ConcatVoxelFeatures`."""

    model_config = ConfigDict(extra="forbid")
    extractors: Sequence[Dict[str, Any]]
    roi: Optional[str] = None
    modalities: Sequence[str] = ()
    expression: Optional[str] = None


@VoxelFeatureExtractorRegistry.register("concat")
class ConcatVoxelFeatures:
    """
    Join several voxel feature families side by side for the same voxels.

    Needed whenever a study describes voxels by more than one kind of
    evidence -- say raw intensity of one modality next to texture of another.
    Homogeneous compositions do NOT need this operator: a single family
    already accepts a modality list, so ``concat`` exists for the genuinely
    mixed case.

    Every child must describe the same ROI, so all children share this
    operator's ``roi`` and their row order is the ROI's C order by contract.

    Args:
        extractors: Child specifications as ``{"name": ..., "params": {...}}``
            mappings, resolved through the voxel-feature registry in the given
            order. A child's own ``roi`` is overridden by this operator's.
        roi: Mask key defining the region of interest for every child;
            ``None`` uses the subject's single mask.
        modalities: Accepted for configuration compatibility and ignored; each
            child names the modalities it reads.
        expression: The original v0 method expression, carried for provenance
            when this extractor was reached by config translation.

    Raises:
        HABITAPIError: If fewer than two children are given.
    """

    def __init__(
        self,
        extractors: Sequence[Dict[str, Any]],
        roi: Optional[str] = None,
        modalities: Sequence[str] = (),
        expression: Optional[str] = None,
    ) -> None:
        children = [dict(child) for child in extractors]
        if len(children) < 2:
            raise HABITAPIError(
                "concat needs at least two child extractors; a single family "
                "already accepts a modality list, so wrapping one child adds "
                f"nothing. Got {len(children)}."
            )
        for child in children:
            if not child.get("name"):
                raise HABITAPIError(
                    f"concat child specification is missing 'name': {child!r}."
                )
        self.extractors = children
        self.roi = roi
        self.modalities = tuple(modalities)
        self.expression = expression

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="concat",
            params={
                "extractors": [dict(child) for child in self.extractors],
                "roi": self.roi,
                "modalities": list(self.modalities),
                "expression": self.expression,
            },
        )

    def _build_children(self) -> List[Any]:
        """
        Instantiate the child extractors from their specifications.

        Returns:
            The child extractor instances, in declaration order.

        Raises:
            ComponentNotFoundError: If a child names an unregistered family.
        """
        children: List[Any] = []
        for child in self.extractors:
            params = dict(child.get("params") or {})
            # One ROI for the whole concatenation: children describing
            # different ROIs could not be joined row-for-row.
            params["roi"] = self.roi
            children.append(
                VoxelFeatureExtractorRegistry.create(str(child["name"]), **params)
            )
        return children

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute and concatenate every child family for one subject.

        Args:
            subject: Subject providing the images and the ROI mask.

        Returns:
            One row per ROI voxel, with every child's columns side by side.

        Raises:
            HABITAPIError: If two children emit the same column name, or a
                child does not describe every ROI voxel.
        """
        mask, _, voxel_index = roi_voxels(subject, self.roi)
        names: List[str] = []
        blocks: List[np.ndarray] = []
        for child in self._build_children():
            field = child(subject)
            if field.values.shape[0] != voxel_index.shape[0]:
                raise HABITAPIError(
                    f"concat: child {child.spec.name!r} described "
                    f"{field.values.shape[0]} voxels of subject "
                    f"{subject.subject_id!r}, but the ROI has "
                    f"{voxel_index.shape[0]}."
                )
            duplicates = [name for name in field.feature_names if name in names]
            if duplicates:
                raise HABITAPIError(
                    f"concat: child {child.spec.name!r} repeats column(s) "
                    f"{duplicates} already produced by an earlier child; "
                    "give the families distinct modalities or rename them."
                )
            names.extend(field.feature_names)
            blocks.append(np.asarray(field.values))

        values = np.concatenate(blocks, axis=1)
        return build_voxel_field(
            subject, mask, voxel_index, names, values, self.spec
        )


VoxelFeatureExtractorRegistry.register_params_model(
    "concat", ConcatVoxelFeaturesParams
)
