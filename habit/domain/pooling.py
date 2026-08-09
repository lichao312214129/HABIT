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
"""Cross-subject pooling atoms of the habitat dataflow: fan-in / fan-out.

A cohort-level habitat design (two-step, direct-pooling) has two data
movements that v0.1 left implicit inside its orchestrator:

* **fan-in** -- merge every subject's clustering units into one cohort
  matrix while remembering which rows belong to which subject;
* **fan-out** -- the inverse movement, splitting a cohort-length vector
  (for example pooled cluster labels) back into per-subject pieces.

Naming them as atoms keeps the movement honest: the cohort matrix exists
only together with its subject index, so a pooled quantity can always be
traced back to the subjects it came from. The subject-level design
(one-step) performs neither movement, which is exactly what
``HabitatSpec.pooling="none"`` declares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.contracts.habitat import Supervoxelization
from habit.exceptions import CompatibilityError, HABITAPIError

__all__ = ["PooledUnits", "fan_in"]


@dataclass(frozen=True)
class PooledUnits:
    """
    The fan-in product: one cohort matrix plus its subject index.

    Attributes:
        frame: Pooled unit-by-feature matrix with a positional index, in
            cohort order. Stored (rather than derived) so cohort-level
            consumers see the exact frame the per-subject units provided,
            dtypes included.
        subject_ids: Owning subject id per row block, in cohort order.
        boundaries: ``(start, stop)`` row range of each subject's block
            inside ``frame``; ``subject_ids[i]`` owns
            ``frame.iloc[start:stop]``.
    """

    frame: pd.DataFrame
    subject_ids: Tuple[str, ...]
    boundaries: Tuple[Tuple[int, int], ...]

    @property
    def matrix(self) -> np.ndarray:
        """Return the pooled matrix as a float64 array, rows in cohort order."""
        return self.frame.to_numpy(dtype=np.float64)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Return the pooled feature columns in order."""
        return tuple(str(column) for column in self.frame.columns)

    def fan_out(self, values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split a cohort-length vector back into per-subject pieces.

        This is the numerical core of the fan-out movement: any quantity
        computed row-wise on the pooled matrix (cluster labels, distances,
        quality scores) returns to its subjects through the same index the
        fan-in recorded.

        Args:
            values: Array whose first axis has one entry per pooled row.

        Returns:
            Mapping of subject id to that subject's slice of ``values``,
            in cohort order.

        Raises:
            HABITAPIError: If ``values`` does not have one entry per row.
        """
        array = np.asarray(values)
        expected = int(self.frame.shape[0])
        if array.shape[0] != expected:
            raise HABITAPIError(
                f"fan_out expects one value per pooled row ({expected}); "
                f"got {array.shape[0]}."
            )
        return {
            subject_id: array[start:stop]
            for subject_id, (start, stop) in zip(self.subject_ids, self.boundaries)
        }


def fan_in(units: Sequence[Supervoxelization]) -> PooledUnits:
    """
    Merge per-subject clustering units into one indexed cohort matrix.

    Row order is cohort order and never sorted or shuffled, because
    clustering can be order-sensitive; the same contract holds in
    :func:`~habit.domain.habitat_model._base.pool_supervoxel_features`,
    which the model fitters pool with, so ``PooledUnits.matrix`` carries
    the very rows a fitter would see.

    Args:
        units: Clustering units in cohort order.

    Returns:
        The pooled matrix together with its subject index.

    Raises:
        HABITAPIError: If ``units`` is empty or two units share a subject
            id (fan-out could not route rows back unambiguously).
        CompatibilityError: If feature columns differ between subjects.
    """
    if not units:
        raise HABITAPIError("fan_in requires at least one clustering unit.")
    feature_names = tuple(str(column) for column in units[0].features.columns)
    frames = []
    boundaries = []
    subject_ids = []
    start = 0
    for unit in units:
        current = tuple(str(column) for column in unit.features.columns)
        if current != feature_names:
            raise CompatibilityError(
                f"Subject {unit.subject_id!r} provides features {current}, "
                f"but the cohort expects {feature_names}."
            )
        subject_id = str(unit.subject_id)
        if subject_id in subject_ids:
            raise HABITAPIError(
                f"fan_in received subject {subject_id!r} twice; pooled rows "
                "must trace back to one subject each."
            )
        frame = unit.feature_frame()
        frames.append(frame)
        subject_ids.append(subject_id)
        boundaries.append((start, start + int(frame.shape[0])))
        start += int(frame.shape[0])
    return PooledUnits(
        frame=pd.concat(frames, ignore_index=True),
        subject_ids=tuple(subject_ids),
        boundaries=tuple(boundaries),
    )
