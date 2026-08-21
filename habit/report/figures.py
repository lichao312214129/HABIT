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
"""Built-in per-subject figure atoms for :class:`~habit.report.api.Report`.

Each atom is a plain Python object: construct it, put it in
``Report.figures``, and the streaming consumer draws it the moment a
subject completes. Atoms call :mod:`habit.viz` (which returns a Figure)
and never touch the filesystem -- persistence is the Report's job.

``at`` names the pipeline boundary the atom reads. First-phase atoms all
read ``"habitat_map"`` (label image + optional per-subject model). They
are not HabitatSpec stages and do not enter the scientific fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from habit.exceptions import HABITAPIError
from habit.kernels.habitat_graph import HabitatGraphFeatureOptions
from habit.kernels.habitat_metrics import (
    habitat_ith_dispersion,
    habitat_volume_fractions,
    ith_score,
    spatial_interaction_matrix,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from habit.report.api import SubjectContext

__all__ = [
    "Overlay",
    "VolumeFractions",
    "MSI",
    "ITH",
    "ClusterValidation",
    "GraphSlice",
    "GraphNetwork2D",
]


def _habitat_ids(ctx: "SubjectContext") -> tuple[int, ...]:
    """Return positive habitat ids from the subject's label image."""
    return tuple(int(v) for v in ctx.habitat_map.habitat_ids)


@dataclass(frozen=True)
class Overlay:
    """
    Habitat labels over one source modality.

    Attributes:
        modality: Image key on the subject (e.g. ``"T1"``).
        at: Pipeline boundary this atom reads. Fixed at ``habitat_map``.
    """

    modality: str
    at: str = "habitat_map"

    def stem(self, subject_id: str) -> str:
        """Return the PNG stem ``<subject>_<modality>_overlay``."""
        return f"{subject_id}_{self.modality}_overlay"

    def draw(self, ctx: "SubjectContext") -> Optional["Figure"]:
        """
        Draw the overlay, or raise if the modality is missing.

        Args:
            ctx: Completed subject's map and images.

        Returns:
            A matplotlib Figure.

        Raises:
            HABITAPIError: If ``ctx.subject`` has no image for ``modality``.
        """
        from habit.viz import plot_habitat_overlay

        try:
            image = ctx.subject.image(self.modality)
        except KeyError as exc:
            raise HABITAPIError(
                f"Overlay(modality={self.modality!r}) needs that image on "
                f"subject {ctx.subject.subject_id!r}."
            ) from exc
        title = f"{ctx.subject.subject_id} {self.modality} habitats"
        return plot_habitat_overlay(image, ctx.habitat_map, title=title)


@dataclass(frozen=True)
class VolumeFractions:
    """Bar chart of per-habitat volume fractions of the non-background VOI."""

    at: str = "habitat_map"

    def stem(self, subject_id: str) -> str:
        """Return the PNG stem ``<subject>_volume_fractions``."""
        return f"{subject_id}_volume_fractions"

    def draw(self, ctx: "SubjectContext") -> Optional["Figure"]:
        """Draw the bar chart, or ``None`` when the map has no habitats."""
        from habit.viz import plot_habitat_volume_fractions

        ids = _habitat_ids(ctx)
        if not ids:
            return None
        fractions = habitat_volume_fractions(ctx.habitat_map.label_array, ids)
        return plot_habitat_volume_fractions(fractions)


@dataclass(frozen=True)
class MSI:
    """Spatial-interaction (MSI) heatmap for the subject's habitat map."""

    at: str = "habitat_map"

    def stem(self, subject_id: str) -> str:
        """Return the PNG stem ``<subject>_msi_matrix``."""
        return f"{subject_id}_msi_matrix"

    def draw(self, ctx: "SubjectContext") -> Optional["Figure"]:
        """Draw the MSI matrix, or ``None`` when the map has no habitats."""
        from habit.viz import plot_msi_matrix

        ids = _habitat_ids(ctx)
        if not ids:
            return None
        n_classes = int(max(ids)) + 1
        matrix = spatial_interaction_matrix(
            ctx.habitat_map.label_array, n_classes=n_classes
        )
        return plot_msi_matrix(matrix, habitat_ids=tuple(range(1, n_classes)))


@dataclass(frozen=True)
class ITH:
    """ITH summary (global score plus optional per-habitat dispersion)."""

    at: str = "habitat_map"

    def stem(self, subject_id: str) -> str:
        """Return the PNG stem ``<subject>_ith_summary``."""
        return f"{subject_id}_ith_summary"

    def draw(self, ctx: "SubjectContext") -> Optional["Figure"]:
        """Draw the ITH panel, or ``None`` when the map has no habitats."""
        from habit.viz import plot_ith_summary

        ids = _habitat_ids(ctx)
        if not ids:
            return None
        labels = ctx.habitat_map.label_array
        return plot_ith_summary(
            ith_score(labels), dispersion=habitat_ith_dispersion(labels)
        )


@dataclass(frozen=True)
class ClusterValidation:
    """Auto-K / elbow / BIC curves from the subject's selection_report."""

    at: str = "habitat_map"

    def stem(self, subject_id: str) -> str:
        """Return the PNG stem ``<subject>_cluster_validation``."""
        return f"{subject_id}_cluster_validation"

    def draw(self, ctx: "SubjectContext") -> Optional["Figure"]:
        """
        Draw the validation curves, or ``None`` when the model carries no
        ``selection_report`` (fixed-K fits).
        """
        from habit.viz import plot_cluster_validation_from_report

        report: Optional[Any] = (ctx.model.preprocessing_state or {}).get(
            "selection_report"
        )
        if not report:
            return None
        return plot_cluster_validation_from_report(report)


@dataclass(frozen=True)
class GraphSlice:
    """
    Representative-slice habitat map with the node lattice overlaid.

    Pass the same :class:`~habit.kernels.habitat_graph.HabitatGraphFeatureOptions`
    used by ``Spec("graph", ...)`` so nodes and edges match the feature
    table. The PNG is a representative **2D slice (display-only)**;
    ``Spec("graph")`` metrics are computed on the **full 3D volume**.

    Attributes:
        options: Graph construction shared with the quantify stage.
            Library default is ``min_distance`` / ``block_size=8``.
        at: Pipeline boundary this atom reads. Fixed at ``habitat_map``.
    """

    options: HabitatGraphFeatureOptions = field(
        default_factory=HabitatGraphFeatureOptions
    )
    at: str = "habitat_map"

    def stem(self, subject_id: str) -> str:
        """Return the PNG stem ``<subject>_graph_slice``."""
        return f"{subject_id}_graph_slice"

    def draw(self, ctx: "SubjectContext") -> Optional["Figure"]:
        """Draw the slice lattice, or ``None`` when the map has no habitats."""
        from habit.viz import plot_habitat_graph_slice

        if not _habitat_ids(ctx):
            return None
        return plot_habitat_graph_slice(
            ctx.habitat_map.label_array, options=self.options
        )


@dataclass(frozen=True)
class GraphNetwork2D:
    """
    Intra- and inter-habitat 2D network on the representative slice.

    Same ``options`` object as :class:`GraphSlice` / ``Spec("graph")``.
    2D only (display-only slice graph). 3D surface / network renders
    stay on :mod:`habit.viz`, not this atom.

    Attributes:
        options: Graph construction shared with the quantify stage.
        at: Pipeline boundary this atom reads. Fixed at ``habitat_map``.
    """

    options: HabitatGraphFeatureOptions = field(
        default_factory=HabitatGraphFeatureOptions
    )
    at: str = "habitat_map"

    def stem(self, subject_id: str) -> str:
        """Return the PNG stem ``<subject>_graph_network_2d``."""
        return f"{subject_id}_graph_network_2d"

    def draw(self, ctx: "SubjectContext") -> Optional["Figure"]:
        """Draw the 2D network, or ``None`` when the map has no habitats."""
        from habit.viz import plot_habitat_graph_network_2d

        if not _habitat_ids(ctx):
            return None
        return plot_habitat_graph_network_2d(
            ctx.habitat_map.label_array, options=self.options
        )
