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
"""What a recipe hands back (L4).

``StudyResult`` lives here rather than in ``habit.contracts`` because it is
the *return type of the recipe layer*, not a contract the lower layers speak:
nothing in L0-L3 produces or consumes one. Keeping it at L4 is also what lets
``save()`` exist at all -- L2 is forbidden from knowing about output
directories, and the old placement forced an explicit architecture-test
exemption for the word ``out_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np

from habit.contracts.habitat import HabitatMap, HabitatModel, Supervoxelization
from habit.contracts.manifest import RunManifest
from habit.contracts.ops import ResultWriter
from habit.contracts.table import FeatureTable

__all__ = ["StudyResult"]

#: File stem used for the cohort feature table, matching the v0.1 layout.
_FEATURE_TABLE_NAME = "habitat_features"

#: Row granularity of the v0.1 ``habitats.parquet`` unit table, selected by
#: the recipe design that produced the result.
#:
#: * ``"supervoxel"`` -- one row per clustering unit (two-step design).
#: * ``"habitat"`` -- one row per assigned habitat within each subject
#:   (one-step design, where a unit IS a habitat by construction).
#: * ``"voxel"`` -- one row per ROI voxel (direct-pooling design).
_UNITS_TABLE_GRANULARITY = {
    "two_step": "supervoxel",
    "one_step": "habitat",
    "direct_pooling": "voxel",
}


@dataclass(frozen=True, eq=False)
class StudyResult:
    """
    What a fitted study hands back, entirely in memory.

    Nothing here has touched the filesystem. Writing is a separate, explicit
    act via :meth:`write` (any :class:`~habit.contracts.ops.ResultWriter`) or
    :meth:`save` (the conventional directory layout), which is what allows
    the identical code to run inside someone else's service where there is no
    output directory at all.

    Attributes:
        habitat_model: The population-level habitat definition. Named in
            full rather than ``model`` because ``model`` already means a
            trained classifier elsewhere in HABIT. ``None`` for designs that
            define habitats per subject rather than across the cohort -- the
            one-step design in particular, where "the definition" is not one
            object (see :attr:`subject_models`).
        pipeline: The subject-level procedure that applies that definition,
            so that model and procedure can be shipped together for external
            validation. ``None`` when no single procedure applies.
        features: Habitat-level features for the fitted cohort.
        habitat_maps: Per-subject habitat label images, in cohort order.
        manifest: Provenance and reporting for this run.
        subject_models: Per-subject habitat definitions, for designs that
            cluster each subject independently. Empty for cohort-level
            designs. Held in memory only: the writer protocol persists one
            habitat model per study, and inventing a per-subject file naming
            convention here would fix a layout no caller has asked for yet.
        units: Per-subject clustering units the habitat maps were labelled
            from, in cohort order, aligned with ``habitat_maps``. This is a
            REPORTING payload, not part of the scientific result: the v0.1
            ``habitats.parquet`` unit table and ``*_supervoxel.nrrd`` maps
            are derived views of it, assembled by the directory writer.
            Empty when the caller only needs the label maps. A design with
            no supervoxel step stores one-voxel units (see
            :func:`~habit.domain.pipeline.voxel_units`), so the field has one
            uniform type regardless of design.
    """

    habitat_model: Optional[HabitatModel]
    pipeline: Any
    features: FeatureTable
    habitat_maps: Tuple[HabitatMap, ...]
    manifest: RunManifest
    subject_models: Mapping[str, HabitatModel] = field(default_factory=dict)
    units: Tuple[Supervoxelization, ...] = ()

    def _units_table_granularity(self) -> Optional[str]:
        """
        Return the v0.1 row granularity for this result's units table.

        Derived from the recipe design recorded in the manifest; the
        apply-habitat-model design takes the granularity of the model it
        projects (supervoxel rows when the fitted procedure partitions the
        ROI, voxel rows when it clusters voxels directly).

        Returns:
            ``"supervoxel"``, ``"habitat"`` or ``"voxel"``; ``None`` when no
            units were collected, meaning no units table should be written.
        """
        if not self.units:
            return None
        design = self.manifest.provenance.produced_by.rsplit(".", maxsplit=1)[-1]
        if design == "apply_habitat_model":
            pipeline = self.pipeline
            has_partition = (
                pipeline is not None
                and getattr(pipeline, "supervoxelizer", None) is not None
            )
            return "supervoxel" if has_partition else "voxel"
        return _UNITS_TABLE_GRANULARITY.get(design)

    def _population_clustering_arrays(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """
        Pool per-subject units into arrays suitable for cohort clustering plots.

        Returns:
            ``(features, habitat_labels, centroids)`` when the study collected
            aligned units and habitat maps; ``None`` when plotting is not
            defined for this result (for example one-step designs with no
            cohort-level model).
        """
        if not self.units or not self.habitat_maps:
            return None
        if len(self.units) != len(self.habitat_maps):
            return None

        # Deferred import: assignment derivation lives with the directory
        # writer because it mirrors the v0.1 habitats table layout.
        from habit.adapters.writers import _unit_assignments

        feature_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        for units, habitat_map in zip(self.units, self.habitat_maps):
            unit_ids, assigned, _counts = _unit_assignments(units, habitat_map)
            if unit_ids.size == 0:
                continue
            features = units.features.loc[unit_ids].to_numpy(dtype=np.float64)
            feature_blocks.append(features)
            label_blocks.append(assigned.astype(np.int64, copy=False))

        if not feature_blocks:
            return None

        pooled_features = np.vstack(feature_blocks)
        pooled_labels = np.concatenate(label_blocks)
        centroids = None
        if self.habitat_model is not None:
            centroids = np.asarray(self.habitat_model.centroids, dtype=np.float64)
        return pooled_features, pooled_labels, centroids

    def write(self, writer: ResultWriter) -> None:
        """
        Hand every artefact to a writer.

        The result decides WHAT is persisted; the writer decides WHERE and in
        what format. That split is what makes an S3 writer, a DICOM-SEG
        writer or a no-op writer possible without touching this class.

        Args:
            writer: Destination implementing
                :class:`~habit.contracts.ops.ResultWriter`.
        """
        for habitat_map in self.habitat_maps:
            writer.write_habitat_map(habitat_map)
        if self.habitat_model is not None:
            writer.write_habitat_model(self.habitat_model)
        writer.write_feature_table(self.features, _FEATURE_TABLE_NAME)
        writer.write_manifest(self.manifest)

    def save(
        self,
        out_dir: Union[str, Path],
        *,
        table_format: str = "parquet",
        map_format: str = "nrrd",
        write_maps: bool = True,
        write_units_table: bool = True,
        write_cluster_plots: bool = False,
        write_cluster_plots_3d: bool = False,
        write_interactive_cluster_plots: bool = False,
    ) -> Path:
        """
        Write the artefacts of this study to a directory.

        Convenience sugar over :meth:`write` with the conventional directory
        writer; the layout itself belongs to
        :class:`~habit.adapters.writers.DirectoryResultWriter`. Unlike
        :meth:`write`, which hands over everything unconditionally (the
        protocol semantics), this entry point honours the two v0.1
        reporting switches so the CLI can keep them meaningful:
        ``write_maps=False`` skips every label map (v0.1 ``save_images: false``)
        and ``write_units_table=False`` skips the units table (v0.1
        ``save_results_csv: false``).

        Beyond the protocol artefacts, when the study collected its
        clustering :attr:`units` this also persists the derived v0.1
        reporting views of them: the ``habitats.parquet``/``habitats.csv``
        unit table (row granularity follows the recipe design) and, for the
        two-step training design, one ``<subject_id>_supervoxel.<ext>`` per
        subject. v0.1 wrote supervoxel maps during training only -- its
        predict path read them back rather than rewriting them -- so the
        apply design writes none either.

        When ``write_cluster_plots=True`` and a cohort-level
        :attr:`habitat_model` is present, a population-level 2D PCA habitat
        scatter is written under ``visualizations/habitat_clustering/``, mirroring
        the v0.1 ``ClusteringService.visualize_habitat_clustering`` layout for
        the static PNG only (interactive 3D HTML remains in the legacy stack).

        Args:
            out_dir: Destination directory, created when missing.
            table_format: On-disk format of the units table, ``"parquet"``
                (v0.1 default) or ``"csv"``.
            map_format: On-disk format of habitat / supervoxel label maps.
                ``"nrrd"`` (v0.1 default), ``"nii"``, ``"nii.gz"``, ``"mha"``,
                or ``"mhd"``.
            write_maps: Write habitat maps (and, for the two-step design,
                supervoxel maps) using ``map_format``.
            write_units_table: Write the ``habitats`` units table.
            write_cluster_plots: Write the population-level 2D PCA clustering
                scatter when cohort-level units and a habitat model exist.
            write_cluster_plots_3d: Also write a static 3D PCA scatter PNG.
            write_interactive_cluster_plots: Also write a rotatable plotly HTML
                file when plotly is installed.

        Returns:
            The directory written to.
        """
        # Imported here rather than at module scope: a caller who never
        # persists anything should not pay for the adapter layer, and L4
        # must not make the filesystem adapter a hard import dependency.
        from habit.adapters.writers import DirectoryResultWriter

        writer = DirectoryResultWriter(out_dir, map_format=map_format)
        if write_maps:
            for habitat_map in self.habitat_maps:
                writer.write_habitat_map(habitat_map)
        if self.habitat_model is not None:
            writer.write_habitat_model(self.habitat_model)
        writer.write_feature_table(self.features, _FEATURE_TABLE_NAME)
        writer.write_manifest(self.manifest)
        granularity = self._units_table_granularity() if write_units_table else None
        if granularity is not None:
            writer.write_units_table(
                self.units,
                self.habitat_maps,
                granularity=granularity,
                table_format=table_format,
            )
            design = self.manifest.provenance.produced_by.rsplit(".", maxsplit=1)[-1]
            if write_maps and design == "two_step":
                for unit in self.units:
                    writer.write_supervoxel_map(unit)
        if write_cluster_plots or write_cluster_plots_3d or write_interactive_cluster_plots:
            self._write_habitat_clustering_plots(
                out_dir,
                write_2d=write_cluster_plots,
                write_3d=write_cluster_plots_3d,
                write_interactive=write_interactive_cluster_plots,
            )
        return writer.root

    def _write_habitat_clustering_plots(
        self,
        out_dir: Union[str, Path],
        *,
        write_2d: bool,
        write_3d: bool,
        write_interactive: bool,
    ) -> None:
        """
        Persist habitat clustering visualisations when defined for this result.

        Args:
            out_dir: Destination directory root.
            write_2d: Write ``habitat_clustering_2D.png``.
            write_3d: Write ``habitat_clustering_3D.png``.
            write_interactive: Write ``habitat_clustering_3D_interactive.html``.
        """
        payload = self._population_clustering_arrays()
        if payload is None or self.habitat_model is None:
            return

        features, labels, centroids = payload
        destination = Path(out_dir) / "visualizations" / "habitat_clustering"
        destination.mkdir(parents=True, exist_ok=True)

        from habit.viz import (
            plot_habitat_clustering_pca_2d,
            plot_habitat_clustering_pca_3d,
            plot_habitat_clustering_pca_3d_interactive,
            use_style,
        )

        import matplotlib.pyplot as plt

        kwargs = dict(
            features=features,
            labels=labels,
            centers=centroids,
            n_clusters=self.habitat_model.n_habitats,
        )
        with use_style("radiology"):
            if write_2d:
                fig = plot_habitat_clustering_pca_2d(**kwargs)
                fig.savefig(destination / "habitat_clustering_2D.png", dpi=600, bbox_inches="tight")
                plt.close(fig)
            if write_3d:
                fig = plot_habitat_clustering_pca_3d(**kwargs)
                fig.savefig(destination / "habitat_clustering_3D.png", dpi=600, bbox_inches="tight")
                plt.close(fig)
        if write_interactive:
            try:
                interactive = plot_habitat_clustering_pca_3d_interactive(**kwargs)
                interactive.write_html(destination / "habitat_clustering_3D_interactive.html")
            except Exception:
                # Interactive export is optional; static PNGs remain the contract.
                pass

    def _write_habitat_clustering_plot(self, out_dir: Union[str, Path]) -> None:
        """Backward-compatible wrapper that writes the 2D PCA scatter only."""
        self._write_habitat_clustering_plots(out_dir, write_2d=True, write_3d=False, write_interactive=False)
