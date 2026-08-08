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
"""Connected-component cleanup for habitat / supervoxel label maps (L3).

Not a registry protocol: there is currently a single strategy (drop tiny
components, refill by nearest seed). The operator is Spec-bearing so results
carry a stable fingerprint in provenance.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import numpy as np

from habit.contracts.habitat import HabitatMap, Supervoxelization, VoxelFeatureField
from habit.domain.supervoxel_features import aggregate_voxel_means
from habit.exceptions import HABITAPIError
from habit.kernels.label_postprocess import remove_small_connected_components
from habit.spec.specs import Spec

__all__ = [
    "ConnectedComponentPostprocess",
    "build_connected_component_postprocess",
]

#: Spec name recorded in provenance / HabitatSpec slots.
_SPEC_NAME = "connected_components"

#: Default parameters mirroring ``ConnectedComponentPostprocessConfig``.
_DEFAULT_MIN_COMPONENT_SIZE = 30
_DEFAULT_CONNECTIVITY = 1
_DEFAULT_REASSIGN_METHOD = "neighbor_vote"
_DEFAULT_MAX_ITERATIONS = 3


class ConnectedComponentPostprocess:
    """
    Clean tiny connected components inside an ROI label map.

    Args:
        min_component_size: Minimum voxels required to keep a component.
        connectivity: Neighborhood connectivity in ``{1, 2, 3}``.
        reassign_method: Strategy name reserved for future variants; only
            ``"neighbor_vote"`` (nearest-seed refill) is implemented.
        max_iterations: Cleanup iteration cap retained for YAML parity; the
            current kernel performs a single remove+refill pass.
    """

    def __init__(
        self,
        *,
        min_component_size: int = _DEFAULT_MIN_COMPONENT_SIZE,
        connectivity: int = _DEFAULT_CONNECTIVITY,
        reassign_method: str = _DEFAULT_REASSIGN_METHOD,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        size = int(min_component_size)
        if size < 1:
            raise HABITAPIError(
                "ConnectedComponentPostprocess.min_component_size must be >= 1."
            )
        conn = int(connectivity)
        if conn not in (1, 2, 3):
            raise HABITAPIError(
                "ConnectedComponentPostprocess.connectivity must be 1, 2, or 3; "
                f"got {connectivity!r}."
            )
        method = str(reassign_method).strip() or _DEFAULT_REASSIGN_METHOD
        if method != "neighbor_vote":
            raise HABITAPIError(
                "ConnectedComponentPostprocess.reassign_method currently only "
                f"supports 'neighbor_vote'; got {reassign_method!r}."
            )
        iterations = int(max_iterations)
        if iterations < 1:
            raise HABITAPIError(
                "ConnectedComponentPostprocess.max_iterations must be >= 1."
            )
        self._min_component_size = size
        self._connectivity = conn
        self._reassign_method = method
        self._max_iterations = iterations
        self.spec = Spec(
            name=_SPEC_NAME,
            params={
                "min_component_size": self._min_component_size,
                "connectivity": self._connectivity,
                "reassign_method": self._reassign_method,
                "max_iterations": self._max_iterations,
            },
        )

    def apply_to_label_array(
        self,
        label_map: np.ndarray,
        roi_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Clean a raw label array inside ``roi_mask``.

        Args:
            label_map: 3D integer label map (0 = background).
            roi_mask: 3D boolean ROI mask.

        Returns:
            Cleaned int32 label map.
        """
        return remove_small_connected_components(
            label_map=label_map,
            roi_mask=roi_mask,
            min_component_size=self._min_component_size,
            connectivity=self._connectivity,
        )

    def apply_to_habitat_map(self, habitat_map: HabitatMap) -> HabitatMap:
        """
        Return a cleaned copy of a habitat label map.

        ROI is ``label_array > 0``. ``habitat_ids`` stay the model's assignable
        set (unchanged), matching the contract that ids are model capability
        rather than presence in one subject.

        Args:
            habitat_map: Habitat labels for one subject.

        Returns:
            New :class:`HabitatMap` with cleaned labels and derived provenance.
        """
        labels = np.asarray(habitat_map.label_array)
        roi_mask = labels > 0
        cleaned = self.apply_to_label_array(labels, roi_mask)
        provenance = habitat_map.provenance.derive(
            produced_by=f"postprocess.{_SPEC_NAME}",
            spec_fingerprint=self.spec.fingerprint(),
        )
        return HabitatMap(
            subject_id=habitat_map.subject_id,
            label_array=cleaned,
            geometry=habitat_map.geometry,
            model_id=habitat_map.model_id,
            habitat_ids=habitat_map.habitat_ids,
            provenance=provenance,
        )

    def apply_to_supervoxelization(
        self,
        units: Supervoxelization,
        field: VoxelFeatureField,
    ) -> Supervoxelization:
        """
        Clean a supervoxel partition and re-aggregate feature means.

        Features must be recomputed after label reassignment so the feature
        matrix stays row-aligned with the surviving supervoxel ids.

        Args:
            units: Supervoxel partition to clean.
            field: Per-voxel features used to rebuild region means.

        Returns:
            New :class:`Supervoxelization` with cleaned labels and means.

        Raises:
            HABITAPIError: If ``field`` and ``units`` disagree on subject or
                geometry shape.
        """
        if field.subject_id != units.subject_id:
            raise HABITAPIError(
                "ConnectedComponentPostprocess.apply_to_supervoxelization "
                f"received field for subject {field.subject_id!r} and units "
                f"for {units.subject_id!r}."
            )
        if tuple(int(v) for v in field.geometry.shape) != tuple(
            int(v) for v in units.geometry.shape
        ):
            raise HABITAPIError(
                "ConnectedComponentPostprocess.apply_to_supervoxelization "
                "requires field and units to share the same voxel grid shape."
            )
        labels = np.asarray(units.label_array)
        roi_mask = labels > 0
        cleaned = self.apply_to_label_array(labels, roi_mask)
        features = aggregate_voxel_means(field, cleaned)
        provenance = units.provenance.derive(
            produced_by=f"postprocess.{_SPEC_NAME}",
            spec_fingerprint=self.spec.fingerprint(),
        )
        return Supervoxelization(
            subject_id=units.subject_id,
            label_array=cleaned,
            features=features,
            geometry=units.geometry,
            provenance=provenance,
        )


def build_connected_component_postprocess(
    spec: Optional[Union[Spec, Mapping[str, Any]]],
) -> Optional[ConnectedComponentPostprocess]:
    """
    Build a postprocess operator from a Spec, or ``None`` when unset.

    Args:
        spec: ``Spec(name="connected_components", params=...)`` or ``None``.

    Returns:
        The operator, or ``None`` when cleanup is not configured.

    Raises:
        HABITAPIError: If ``spec.name`` is not ``connected_components``.
    """
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        name = str(spec.get("name", _SPEC_NAME))
        params = dict(spec.get("params") or {})
    else:
        name = spec.name
        params = dict(spec.params)
    if name != _SPEC_NAME:
        raise HABITAPIError(
            "Connected-component postprocess Spec must be named "
            f"{_SPEC_NAME!r}; got {name!r}."
        )
    return ConnectedComponentPostprocess(**params)
