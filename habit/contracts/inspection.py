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
"""Step-inspection contracts: observe habitat dataflow without changing it.

These types let a caller watch every existing pipeline boundary (voxel
features, supervoxels, cohort prep, habitat maps). Observers are NOT part of
an analysis declaration: they must never enter ``Spec``, fingerprints, or
``RunManifest``. Default ``inspect=None`` keeps behaviour bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Optional, Protocol, Tuple, runtime_checkable

__all__ = [
    "STEP_VOXEL_FEATURES_RAW",
    "STEP_VOXEL_FEATURES_PREPROCESSED",
    "STEP_SUPERVOXELS_PARTITION",
    "STEP_SUPERVOXELS_POSTPROCESSED",
    "STEP_SUPERVOXELS_DESCRIBED",
    "STEP_SUPERVOXELS_PREPROCESSED",
    "STEP_UNITS_COHORT_PREPROCESSED",
    "STEP_HABITAT_MAP",
    "STEP_HABITAT_FEATURES",
    "STEP_NAMES",
    "StepRecord",
    "StepObserver",
]

STEP_VOXEL_FEATURES_RAW: Final[str] = "voxel_features.raw"
STEP_VOXEL_FEATURES_PREPROCESSED: Final[str] = "voxel_features.preprocessed"
STEP_SUPERVOXELS_PARTITION: Final[str] = "supervoxels.partition"
STEP_SUPERVOXELS_POSTPROCESSED: Final[str] = "supervoxels.postprocessed"
STEP_SUPERVOXELS_DESCRIBED: Final[str] = "supervoxels.described"
STEP_SUPERVOXELS_PREPROCESSED: Final[str] = "supervoxels.preprocessed"
STEP_UNITS_COHORT_PREPROCESSED: Final[str] = "units.cohort_preprocessed"
STEP_HABITAT_MAP: Final[str] = "habitat_map"
STEP_HABITAT_FEATURES: Final[str] = "habitat_features"

STEP_NAMES: Final[Tuple[str, ...]] = (
    STEP_VOXEL_FEATURES_RAW,
    STEP_VOXEL_FEATURES_PREPROCESSED,
    STEP_SUPERVOXELS_PARTITION,
    STEP_SUPERVOXELS_POSTPROCESSED,
    STEP_SUPERVOXELS_DESCRIBED,
    STEP_SUPERVOXELS_PREPROCESSED,
    STEP_UNITS_COHORT_PREPROCESSED,
    STEP_HABITAT_MAP,
    STEP_HABITAT_FEATURES,
)


@dataclass(frozen=True, eq=False)
class StepRecord:
    """
    One observed pipeline boundary for one subject.

    Attributes:
        step: Stable step name from :data:`STEP_NAMES`.
        subject_id: Owning subject.
        payload: Domain object at that boundary (or a frame kept by a
            recorder). Contracts never convert payloads to DataFrames.
        produced_by: Short producer tag (component or stage id).
        spec_fingerprint: Optional component fingerprint for provenance.
    """

    step: str
    subject_id: str
    payload: Any
    produced_by: str
    spec_fingerprint: Optional[str] = None


@runtime_checkable
class StepObserver(Protocol):
    """
    Optional sink for :class:`StepRecord` events.

    Implementations must treat ``wants`` as a pure filter: when it returns
    ``False``, pipelines skip calling the observer for that step so unused
    records cost nothing beyond the existing intermediate objects.
    """

    def wants(self, step: str) -> bool:
        """Return whether this observer cares about ``step``."""

    def __call__(self, record: StepRecord) -> None:
        """Receive one step record."""
