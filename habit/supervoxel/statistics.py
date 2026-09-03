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
"""Per-supervoxel statistics of the voxel signal: mean, std, percentiles.

These are single-modality (form 1) supervoxel extractors. Each one reduces
the per-voxel signal inside every supervoxel to one number per region:

- ``source="working"`` (default) reduces the *preprocessed* voxel feature
  columns produced by the voxel extractor / supervoxelizer.
- ``source="original"`` reduces the raw, pre-preprocessing modality signal
  the supervoxelizer saw; output columns gain the ``-original`` suffix,
  matching the v0.1 ``mean_voxel_features`` contract.

When the pipeline binds the voxel fields (:meth:`bind_fields`), the
statistic is recomputed from the voxel grid. Without binding, ``mean``
falls back to selecting columns from the partition's attached mean
features, which carries identical numbers by construction; ``std`` and
``percentile`` require binding because attached features only carry means.

Examples:
    >>> from habit.supervoxel import (
    ...     MeanSupervoxelFeatures,
    ...     StdSupervoxelFeatures,
    ...     PercentileSupervoxelFeatures,
    ... )

    >>> MeanSupervoxelFeatures(modality="T1").spec
    Spec(name='mean', params={'source': 'working', 'modality': 'T1'})
    >>> StdSupervoxelFeatures(modality="T1", source="original", as_="coarse").spec
    Spec(name='std', params={'source': 'original', 'modality': 'T1', 'as_': 'coarse'})
    >>> PercentileSupervoxelFeatures(modality="T1", q=90).spec
    Spec(name='percentile', params={'source': 'working', 'modality': 'T1', 'q': 90.0})
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import (
    Supervoxelization,
    VoxelFeatureField,
)
from habit.contracts.subject import Subject
from habit.spec.specs import Spec
from habit.supervoxel.features_base import (
    aggregate_voxel_statistic,
    with_features,
)
from habit.supervoxel.features_registry import (
    SupervoxelFeatureExtractorRegistry,
)

__all__ = [
    "MeanSupervoxelFeatures",
    "StdSupervoxelFeatures",
    "PercentileSupervoxelFeatures",
]

_STATISTIC_SOURCES = ("working", "original")


class _RegionStatisticBase:
    """
    Shared behaviour of the per-supervoxel statistic extractors.

    Class attributes ``name`` / ``column_prefix`` specialise the statistic;
    everything else -- field binding, column selection by modality,
    renaming with alias and the ``-original`` suffix -- is common.
    """

    #: Registered component name; also the statistic forwarded to
    #: :func:`aggregate_voxel_statistic`.
    name: str = ""
    #: Prefix prepended to output column names; ``mean`` keeps the
    #: historical bare names.
    column_prefix: str = ""

    def __init__(
        self,
        modality: Optional[str] = None,
        source: str = "working",
        as_: Optional[str] = None,
    ) -> None:
        if source not in _STATISTIC_SOURCES:
            raise HABITAPIError(
                f"{self.name}: 'source' must be one of "
                f"{_STATISTIC_SOURCES}, got {source!r}."
            )
        self.modality = modality
        self.source = source
        self.as_ = as_
        self._working: Optional[VoxelFeatureField] = None
        self._original: Optional[VoxelFeatureField] = None

    @property
    def spec(self) -> Spec:
        params: Dict[str, Any] = {"source": self.source}
        if self.modality is not None:
            params["modality"] = self.modality
        if self.as_ is not None:
            params["as_"] = self.as_
        return Spec(name=self.name, params=self._extra_spec_params(params))

    def _extra_spec_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for statistic-specific parameters (``q`` on percentile)."""
        return params

    def bind_fields(
        self,
        working: Optional[VoxelFeatureField] = None,
        original: Optional[VoxelFeatureField] = None,
    ) -> None:
        """
        Bind the per-voxel fields the statistic is computed from.

        Called by the subject pipeline before the extractor runs. A bound
        field lets the extractor recompute the statistic from the voxel
        grid instead of selecting from the partition's attached means.

        Args:
            working: Preprocessed per-voxel feature columns.
            original: Raw pre-preprocessing modality columns, required for
                ``source="original"``.
        """
        self._working = working
        self._original = original

    def __call__(
        self,
        subject: Subject,
        partition: Supervoxelization,
    ) -> Supervoxelization:
        """
        Reduce the voxel signal inside every supervoxel to one statistic.

        Args:
            subject: The subject being described.
            partition: The subject's supervoxel partition.

        Returns:
            The partition with the statistic columns attached.
        """
        field = self._field_for_source()
        if field is not None:
            features = self._aggregate(field, partition)
            return with_features(partition, features, self.spec)
        return self._from_partition(subject, partition)

    # ------------------------------------------------------------------
    # computation paths
    # ------------------------------------------------------------------

    def _field_for_source(self) -> Optional[VoxelFeatureField]:
        """Return the bound field matching ``source``, or ``None``."""
        if self.source == "original":
            if self._original is None:
                raise HABITAPIError(
                    f"{self.name}: source='original' requires the pipeline "
                    "to bind the pre-preprocessing voxel field (bind_fields); "
                    "standalone calls must bind it explicitly."
                )
            return self._original
        return self._working

    def _aggregate(
        self,
        field: VoxelFeatureField,
        partition: Supervoxelization,
    ) -> pd.DataFrame:
        """Recompute the statistic from a bound voxel field."""
        selected = self._select_columns(list(field.feature_names))
        features = aggregate_voxel_statistic(
            field,
            partition.label_array,
            statistic=self.name,
            q=self._q(),
            columns=selected,
        )
        features.columns = self._rename(selected)
        return features

    def _from_partition(
        self,
        subject: Subject,
        partition: Supervoxelization,
    ) -> Supervoxelization:
        """
        Fall back to the partition's attached mean features.

        Only ``mean`` may take this path: the attached features *are* the
        means of the working field by construction, so selecting columns
        from them carries identical numbers. ``std`` / ``percentile``
        cannot be derived from means and must fail loudly instead.
        """
        if self.name != "mean":
            raise HABITAPIError(
                f"{self.name}: requires the pipeline to bind the voxel "
                "feature field (bind_fields); the partition's attached "
                "features only carry means."
            )
        columns = [str(column) for column in partition.features.columns]
        selected = self._select_columns(columns)
        features = partition.features[selected].copy()
        features.columns = self._rename(selected)
        return with_features(partition, features, self.spec)

    # ------------------------------------------------------------------
    # column selection and naming
    # ------------------------------------------------------------------

    def _select_columns(self, columns: List[str]) -> List[str]:
        """
        Select the columns belonging to ``modality``.

        A column belongs to the modality when it equals the modality name
        (single-column feature such as ``raw``) or carries the structured
        ``{feature}-{modality}`` suffix.
        """
        if self.modality is None:
            return list(columns)
        selected = [
            column
            for column in columns
            if column == self.modality
            or column.endswith(f"-{self.modality}")
        ]
        if not selected:
            raise HABITAPIError(
                f"{self.name}: no feature column for modality "
                f"{self.modality!r}; available: {columns}."
            )
        return selected

    def _rename(self, columns: List[str]) -> List[str]:
        """
        Apply the statistic prefix, the ``as_`` alias and the
        ``-original`` source suffix to the selected column names.
        """
        renamed: List[str] = []
        for column in columns:
            name = f"{self.column_prefix}{column}"
            if self.as_ is not None:
                if column == self.modality:
                    name = f"{self.column_prefix}{self.as_}"
                elif column.endswith(f"-{self.modality}"):
                    stem = column[: -len(self.modality)]
                    name = f"{self.column_prefix}{stem}{self.as_}"
            if self.source == "original":
                name = f"{name}-original"
            renamed.append(name)
        return renamed

    def _q(self) -> float:
        """Return the percentile parameter; unused for mean/std."""
        return 90.0


@SupervoxelFeatureExtractorRegistry.register("mean")
class MeanSupervoxelFeatures(_RegionStatisticBase):
    """
    Average the voxel signal within each supervoxel, one modality at a time.

    With ``modality=None`` every column is averaged, matching the default
    summary attached by the supervoxelizers. ``source="original"`` averages
    the raw pre-preprocessing modality signal and appends the ``-original``
    suffix, matching the v0.1 ``mean_voxel_features`` contract.
    """

    name = "mean"
    column_prefix = ""


@SupervoxelFeatureExtractorRegistry.register("std")
class StdSupervoxelFeatures(_RegionStatisticBase):
    """
    Sample standard deviation of the voxel signal within each supervoxel.

    Captures intra-region heterogeneity (e.g. ADC heterogeneity inside a
    habitat). Uses pandas' ``std`` (``ddof=1``); single-voxel supervoxels
    yield ``NaN``.
    """

    name = "std"
    column_prefix = "std-"


@SupervoxelFeatureExtractorRegistry.register("percentile")
class PercentileSupervoxelFeatures(_RegionStatisticBase):
    """
    A percentile of the voxel signal within each supervoxel.

    Robust location summary alternative to the mean; ``q=90`` (default)
    gives the 90th percentile with pandas' linear interpolation.
    """

    name = "percentile"

    def __init__(
        self,
        modality: Optional[str] = None,
        source: str = "working",
        q: float = 90.0,
        as_: Optional[str] = None,
    ) -> None:
        if not 0 < q < 100:
            raise HABITAPIError(
                f"percentile: 'q' must be in (0, 100), got {q}."
            )
        super().__init__(modality=modality, source=source, as_=as_)
        self.q = float(q)

    @property
    def column_prefix(self) -> str:
        # ``q:g`` keeps integers tidy (``p90``) while allowing ``p97.5``.
        return f"p{self.q:g}-"

    def _extra_spec_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["q"] = self.q
        return params

    def _q(self) -> float:
        return self.q

