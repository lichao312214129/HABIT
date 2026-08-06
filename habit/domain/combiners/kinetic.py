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
"""Kinetic combiner: contrast wash-in/wash-out slopes from phase blocks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from habit.domain.combiners._base import concat_blocks
from habit.domain.combiners.registry import CombinerRegistry
from habit.domain.voxel_features.kinetic import (
    FEATURE_NAMES,
    kinetic_slopes,
    resolve_phase_times,
)
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = ["KineticCombiner", "KineticCombinerParams"]

#: Number of phase columns the kinetic model consumes, in acquisition
#: order: unenhanced, arterial, portal-venous, delayed.
_N_PHASES = 4


class KineticCombinerParams(BaseModel):
    """Constructor parameters for :class:`KineticCombiner`."""

    model_config = ConfigDict(extra="forbid")
    timestamps: Union[str, Dict[str, Dict[str, str]]]
    phases: Sequence[str] = ()
    time_format: str = "%H-%M-%S"


@CombinerRegistry.register("kinetic")
class KineticCombiner:
    """
    Per-unit enhancement slopes across a dynamic contrast series.

    The block-level counterpart of the ``kinetic`` voxel extractor: the four
    phase intensities arrive as child blocks -- typically ``raw(phase)``
    leaves in acquisition order -- instead of being read from the subject's
    images. Both forms share the same slope math and column names
    (:data:`~habit.domain.voxel_features.kinetic.FEATURE_NAMES`), so the
    combiner is the explicit-tree spelling of the same algorithm.

    Example::

        kinetic(
            raw("pre_contrast"), raw("LAP"), raw("PVP"), raw("delay_3min"),
            timestamps="times.csv",
        )

    Args:
        timestamps: Acquisition times per subject, either as a mapping
            ``{subject_id: {phase: "HH-MM-SS"}}`` for API callers, or a path
            to the v0.1 timestamp table.
        phases: Phase labels in acquisition order (unenhanced, arterial,
            portal-venous, delayed). Empty resolves to the merged child
            column names -- for ``raw`` children, their modality names.
        time_format: ``strptime`` format of the timestamp values.
    """

    def __init__(
        self,
        timestamps: Union[str, Mapping[str, Mapping[str, str]]],
        phases: Sequence[str] = (),
        time_format: str = "%H-%M-%S",
    ) -> None:
        resolved_phases = tuple(str(name) for name in phases)
        if resolved_phases and len(resolved_phases) != _N_PHASES:
            raise HABITAPIError(
                "kinetic requires exactly four phases (unenhanced, arterial, "
                f"portal-venous, delayed); got {resolved_phases}."
            )
        self.timestamps = timestamps
        self.phases = resolved_phases
        self.time_format = str(time_format)
        self._time_cache: Dict[str, Any] = {}

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
                "time_format": self.time_format,
            },
        )

    def __call__(
        self,
        blocks: Sequence[pd.DataFrame],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Compute the three kinetic slopes from the child phase blocks.

        Args:
            blocks: Child blocks whose merged column count is exactly four,
                in acquisition order (unenhanced, arterial, portal-venous,
                delayed).
            context: Must carry ``"subject_id"`` so the subject's
                acquisition times can be resolved.

        Returns:
            One column per slope, named as :data:`FEATURE_NAMES`.

        Raises:
            HABITAPIError: If the merged block does not hold exactly four
                columns, the subject id is missing from the context, or the
                subject has no acquisition times.
        """
        merged = concat_blocks(blocks, owner="kinetic")
        if merged.shape[1] != _N_PHASES:
            raise HABITAPIError(
                f"kinetic: expects exactly {_N_PHASES} phase columns from "
                f"its children (unenhanced, arterial, portal-venous, "
                f"delayed); the merged block has {merged.shape[1]} "
                f"({list(merged.columns)})."
            )
        subject_id = (context or {}).get("subject_id")
        if subject_id is None:
            raise HABITAPIError(
                "kinetic: the tree wrapper must supply "
                "context={'subject_id': ...}; acquisition times are "
                "resolved per subject and cannot be guessed from the "
                "blocks alone."
            )
        phases = (
            self.phases
            if self.phases
            else tuple(str(column) for column in merged.columns)
        )
        times = resolve_phase_times(
            self.timestamps,
            phases,
            self.time_format,
            str(subject_id),
            self._time_cache,
            owner="kinetic",
        )
        intensity = {
            phase: merged[phase].to_numpy(dtype=np.float64) for phase in phases
        }
        slopes = kinetic_slopes(intensity, times, phases)
        return pd.DataFrame({name: slopes[name] for name in FEATURE_NAMES})


CombinerRegistry.register_params_model("kinetic", KineticCombinerParams)
