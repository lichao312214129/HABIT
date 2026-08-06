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
"""Kinetic voxel features: contrast wash-in and wash-out slopes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.domain.voxel_features._base import (
    aligned_image,
    build_voxel_field,
    roi_voxels,
)
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = ["KineticVoxelFeatures", "KineticVoxelFeaturesParams"]

#: v0.1 phase modality keys, in acquisition order.
DEFAULT_PHASES: Sequence[str] = ("pre_contrast", "LAP", "PVP", "delay_3min")

#: Feature columns produced, in column order.
FEATURE_NAMES: Sequence[str] = (
    "wash_in_slope",
    "wash_out_slope_lap_pvp",
    "wash_out_slope_pvp_dp",
)

#: Seconds the unenhanced phase precedes the arterial phase by. v0.1 derives
#: the pre-contrast acquisition time this way because timestamp tables record
#: the contrast phases only.
PRE_CONTRAST_LEAD_SECONDS = 25.0

#: Guard against a zero time difference, as in v0.1.
_EPSILON = 1e-6


class KineticVoxelFeaturesParams(BaseModel):
    """Constructor parameters for :class:`KineticVoxelFeatures`."""

    model_config = ConfigDict(extra="forbid")
    timestamps: Union[str, Dict[str, Dict[str, str]]]
    phases: Sequence[str] = DEFAULT_PHASES
    roi: Optional[str] = None
    time_format: str = "%H-%M-%S"
    modalities: Sequence[str] = ()
    expression: Optional[str] = None


@VoxelFeatureExtractorRegistry.register("kinetic")
class KineticVoxelFeatures:
    """
    Per-voxel enhancement slopes across a dynamic contrast series.

    A voxel's absolute intensity depends on the scanner; how fast it takes up
    and releases contrast depends on its perfusion, so kinetic slopes separate
    tissue that looks identical on any single phase. Three slopes are
    produced: wash-in (unenhanced to arterial), and two wash-out segments
    (arterial to portal-venous, portal-venous to delayed).

    Slopes are divided by the true inter-phase intervals, which differ between
    subjects, so the extractor needs each subject's acquisition times. The
    unenhanced phase is placed :data:`PRE_CONTRAST_LEAD_SECONDS` before the
    arterial phase, and negative wash-in is clipped to zero -- both v0.1
    conventions, preserved so published values stay reproducible.

    Args:
        timestamps: Acquisition times per subject, either as a mapping
            ``{subject_id: {phase: "HH-MM-SS"}}`` for API callers, or a path to
            the v0.1 timestamp table.
        phases: Modality keys of the series in acquisition order:
            unenhanced, arterial, portal-venous, delayed.
        roi: Mask key defining the region of interest; ``None`` uses the
            subject's single mask.
        time_format: ``strptime`` format of the timestamp values.
        modalities: Accepted for configuration compatibility and ignored;
            ``phases`` defines which images are read, because the four phases
            have fixed roles that a flat list cannot express.
        expression: The original v0 method expression, carried for provenance
            when this extractor was reached by config translation.

    Raises:
        HABITAPIError: If ``phases`` does not name exactly four modalities.
    """

    def __init__(
        self,
        timestamps: Union[str, Mapping[str, Mapping[str, str]]],
        phases: Sequence[str] = DEFAULT_PHASES,
        roi: Optional[str] = None,
        time_format: str = "%H-%M-%S",
        modalities: Sequence[str] = (),
        expression: Optional[str] = None,
    ) -> None:
        resolved_phases = tuple(str(name) for name in phases)
        if len(resolved_phases) != 4:
            raise HABITAPIError(
                "kinetic requires exactly four phases (unenhanced, arterial, "
                f"portal-venous, delayed); got {resolved_phases}."
            )
        self.timestamps = timestamps
        self.phases = resolved_phases
        self.roi = roi
        self.time_format = str(time_format)
        self.modalities = tuple(modalities)
        self.expression = expression
        self._time_table: Optional[Any] = None

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="kinetic",
            params={
                "timestamps": (
                    self.timestamps
                    if isinstance(self.timestamps, str)
                    else {
                        str(subject): dict(phases)
                        for subject, phases in self.timestamps.items()
                    }
                ),
                "phases": list(self.phases),
                "roi": self.roi,
                "time_format": self.time_format,
                "modalities": list(self.modalities),
                "expression": self.expression,
            },
        )

    def _phase_times(self, subject_id: str) -> Dict[str, Any]:
        """
        Resolve one subject's acquisition times as parsed timestamps.

        Args:
            subject_id: Subject to look up.

        Returns:
            Phase name -> parsed timestamp, including the derived unenhanced
            phase.

        Raises:
            HABITAPIError: If the subject or a contrast phase is missing from
                the timestamp table.
        """
        import pandas as pd

        if isinstance(self.timestamps, str):
            if self._time_table is None:
                from habit.utils.io_utils import load_timestamp

                self._time_table = load_timestamp(self.timestamps)
            table = self._time_table
            if subject_id not in table.index:
                raise HABITAPIError(
                    f"kinetic: no acquisition times for subject {subject_id!r} "
                    f"in {self.timestamps!r}."
                )
            row = table.loc[subject_id].to_dict()
        else:
            if subject_id not in self.timestamps:
                raise HABITAPIError(
                    f"kinetic: no acquisition times for subject {subject_id!r}."
                )
            row = dict(self.timestamps[subject_id])

        unenhanced, arterial, portal, delayed = self.phases
        missing = [
            phase for phase in (arterial, portal, delayed) if phase not in row
        ]
        if missing:
            raise HABITAPIError(
                f"kinetic: subject {subject_id!r} is missing acquisition times "
                f"for {missing}."
            )
        times = {
            phase: pd.to_datetime(str(row[phase]), format=self.time_format)
            for phase in (arterial, portal, delayed)
        }
        # v0.1 convention: the unenhanced scan is not timestamped, so it is
        # placed a fixed lead time before the arterial phase.
        times[unenhanced] = times[arterial] - pd.Timedelta(
            seconds=PRE_CONTRAST_LEAD_SECONDS
        )
        return times

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute per-voxel kinetic slopes for one subject.

        Args:
            subject: Subject providing the four phase images and the mask.

        Returns:
            One row per ROI voxel and one column per slope.

        Raises:
            GeometryError: If a phase and the mask are on different grids.
            HABITAPIError: If a phase image or an acquisition time is absent.
        """
        unenhanced, arterial, portal, delayed = self.phases
        missing = [name for name in self.phases if name not in subject.images]
        if missing:
            raise HABITAPIError(
                f"kinetic: subject {subject.subject_id!r} does not provide "
                f"phase images {missing}; available: {sorted(subject.images)}."
            )

        mask, inside, voxel_index = roi_voxels(subject, self.roi)
        intensity = {
            phase: aligned_image(subject, phase, mask, owner="kinetic")[inside]
            for phase in self.phases
        }
        times = self._phase_times(subject.subject_id)

        delta_wash_in = (times[arterial] - times[unenhanced]).total_seconds()
        delta_early = (times[portal] - times[arterial]).total_seconds()
        delta_late = (times[delayed] - times[portal]).total_seconds()

        enhancement = intensity[arterial] - intensity[unenhanced]
        # v0.1 treats a drop after contrast as no enhancement rather than as
        # negative wash-in.
        enhancement = np.clip(enhancement, 0.0, None)

        columns: List[np.ndarray] = [
            enhancement / (delta_wash_in + _EPSILON),
            (intensity[portal] - intensity[arterial]) / (delta_early + _EPSILON),
            (intensity[delayed] - intensity[portal]) / (delta_late + _EPSILON),
        ]
        values = np.stack(columns, axis=1)
        return build_voxel_field(
            subject, mask, voxel_index, FEATURE_NAMES, values, self.spec
        )


VoxelFeatureExtractorRegistry.register_params_model(
    "kinetic", KineticVoxelFeaturesParams
)
