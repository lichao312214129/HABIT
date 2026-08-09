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
"""In-memory :class:`~habit.contracts.inspection.StepObserver` implementation."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.contracts.habitat import HabitatMap, Supervoxelization, VoxelFeatureField
from habit.contracts.inspection import STEP_NAMES, StepRecord
from habit.contracts.table import FeatureTable
from habit.exceptions import HABITAPIError

__all__ = ["StepRecorder"]


def _payload_to_frame(payload: object) -> pd.DataFrame:
    """
    Convert a step payload to a pandas frame for lightweight retention.

    Args:
        payload: Domain object emitted at a pipeline boundary.

    Returns:
        A DataFrame view of the payload's tabular content, or a tiny summary
        frame for label maps that have no feature matrix.
    """
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, VoxelFeatureField):
        return payload.feature_frame().copy()
    if isinstance(payload, Supervoxelization):
        return payload.feature_frame().copy()
    if isinstance(payload, FeatureTable):
        return payload.frame.copy()
    if isinstance(payload, HabitatMap):
        labels = np.asarray(payload.label_array)
        present = sorted(int(v) for v in np.unique(labels) if int(v) != 0)
        return pd.DataFrame(
            {
                "subject_id": [payload.subject_id],
                "model_id": [payload.model_id],
                "n_voxels": [int(labels.size)],
                "n_habitats_present": [len(present)],
                "habitat_ids_present": [tuple(present)],
            }
        )
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    raise HABITAPIError(
        f"StepRecorder cannot convert payload type {type(payload)!r} to a "
        "DataFrame; use keep='objects' or extend the converter."
    )


class StepRecorder:
    """
    Collect selected step records in memory for debugging and QA.

    Args:
        steps: Step names to keep; ``None`` keeps every known step.
        subjects: Subject ids to keep; ``None`` keeps every subject until
            ``max_subjects`` is hit.
        max_subjects: Optional cap on distinct subject ids retained.
        keep: ``"frames"`` stores only tabular views (default, lighter);
            ``"objects"`` stores the original domain payloads.
    """

    def __init__(
        self,
        steps: Optional[Sequence[str]] = None,
        subjects: Optional[Sequence[str]] = None,
        max_subjects: Optional[int] = None,
        keep: Literal["frames", "objects"] = "frames",
    ) -> None:
        if keep not in ("frames", "objects"):
            raise HABITAPIError(
                f"StepRecorder.keep must be 'frames' or 'objects'; got {keep!r}."
            )
        if max_subjects is not None and int(max_subjects) < 1:
            raise HABITAPIError(
                f"StepRecorder.max_subjects must be >= 1; got {max_subjects!r}."
            )
        if steps is not None:
            # Accept legacy STEP_NAMES and stage-bound names (``{stage}.output``)
            # plus the cohort sentinel records emitted after pool/fit.
            unknown = tuple(
                name
                for name in steps
                if name not in STEP_NAMES and not str(name).endswith(".output")
            )
            if unknown:
                raise HABITAPIError(
                    f"Unknown inspection step name(s): {list(unknown)}. "
                    f"Known legacy steps: {list(STEP_NAMES)}; stage-bound "
                    "names must look like '<stage_name>.output'."
                )
        self._steps = None if steps is None else frozenset(steps)
        self._subjects = None if subjects is None else frozenset(subjects)
        self._max_subjects = None if max_subjects is None else int(max_subjects)
        self._keep = keep
        self._accepted_subjects: List[str] = []
        self._records: List[StepRecord] = []

    def wants(self, step: str) -> bool:
        """
        Return whether this recorder will accept ``step``.

        Subject filtering is applied in :meth:`__call__` because the subject
        id is not available at ``wants`` time.

        Args:
            step: Pipeline step name.

        Returns:
            ``False`` when the step was filtered out by ``steps=``.
        """
        return self._steps is None or step in self._steps

    def __call__(self, record: StepRecord) -> None:
        """
        Store ``record`` when it passes step / subject / capacity filters.

        Args:
            record: One observed boundary.
        """
        if not self.wants(record.step):
            return
        # Cohort-level records (after pool/fit) use the ``__cohort__`` sentinel
        # and are not subject to per-subject filters / max_subjects caps.
        is_cohort = record.subject_id == "__cohort__"
        if (
            not is_cohort
            and self._subjects is not None
            and record.subject_id not in self._subjects
        ):
            return
        if not is_cohort and record.subject_id not in self._accepted_subjects:
            if (
                self._max_subjects is not None
                and len(self._accepted_subjects) >= self._max_subjects
            ):
                return
            self._accepted_subjects.append(record.subject_id)
        payload = (
            record.payload
            if self._keep == "objects"
            else _payload_to_frame(record.payload)
        )
        self._records.append(
            StepRecord(
                step=record.step,
                subject_id=record.subject_id,
                payload=payload,
                produced_by=record.produced_by,
                spec_fingerprint=record.spec_fingerprint,
            )
        )

    def steps(self) -> Tuple[str, ...]:
        """Return distinct step names retained, in first-seen order."""
        seen: List[str] = []
        for record in self._records:
            if record.step not in seen:
                seen.append(record.step)
        return tuple(seen)

    def subjects(self) -> Tuple[str, ...]:
        """Return distinct subject ids retained, in first-seen order."""
        return tuple(self._accepted_subjects)

    def records(
        self,
        step: Optional[str] = None,
        subject_id: Optional[str] = None,
    ) -> Tuple[StepRecord, ...]:
        """
        Return stored records, optionally filtered.

        Args:
            step: Keep only this step name when set.
            subject_id: Keep only this subject when set.

        Returns:
            Matching records in arrival order.
        """
        out: List[StepRecord] = []
        for record in self._records:
            if step is not None and record.step != step:
                continue
            if subject_id is not None and record.subject_id != subject_id:
                continue
            out.append(record)
        return tuple(out)

    def frame(
        self,
        step: str,
        subject_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return the tabular payload for one step (and optional subject).

        Args:
            step: Step name to retrieve.
            subject_id: Subject id; required when more than one subject was
                retained for ``step``.

        Returns:
            A DataFrame for that step.

        Raises:
            HABITAPIError: If no record matches, or ``subject_id`` is ambiguous.
        """
        matched = self.records(step=step, subject_id=subject_id)
        if not matched:
            raise HABITAPIError(
                f"No inspection record for step={step!r}"
                + (f", subject_id={subject_id!r}" if subject_id is not None else "")
                + "."
            )
        if subject_id is None and len({r.subject_id for r in matched}) > 1:
            raise HABITAPIError(
                f"Step {step!r} has records for multiple subjects "
                f"{sorted({r.subject_id for r in matched})}; pass subject_id=."
            )
        record = matched[0]
        if isinstance(record.payload, pd.DataFrame):
            return record.payload.copy()
        return _payload_to_frame(record.payload)

    def summary(self) -> pd.DataFrame:
        """
        Return a compact table of retained records.

        Returns:
            Columns ``step``, ``subject_id``, ``n_rows``, ``n_cols``,
            ``produced_by``.
        """
        rows: List[dict] = []
        for record in self._records:
            frame = (
                record.payload
                if isinstance(record.payload, pd.DataFrame)
                else _payload_to_frame(record.payload)
            )
            rows.append(
                {
                    "step": record.step,
                    "subject_id": record.subject_id,
                    "n_rows": int(frame.shape[0]),
                    "n_cols": int(frame.shape[1]),
                    "produced_by": record.produced_by,
                }
            )
        return pd.DataFrame(
            rows,
            columns=["step", "subject_id", "n_rows", "n_cols", "produced_by"],
        )
