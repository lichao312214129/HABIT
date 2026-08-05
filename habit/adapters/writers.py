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
"""Filesystem implementation of the :class:`ResultWriter` protocol (L1).

Persisting results is deliberately a separate object from producing them.
The recipe layer hands finished, in-memory artefacts to a writer, so the same
recipe call runs unchanged inside a service that has no output directory --
and so the v0.1 output layout becomes ONE writer rather than a property of
the algorithms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.geometry import Geometry
from habit.contracts.habitat import (
    HabitatMap,
    HabitatModel,
    Supervoxelization,
)
from habit.contracts.manifest import RunManifest
from habit.contracts.table import FeatureTable
from habit.utils.habitats_results_io import (
    normalize_habitats_results_format,
    save_habitats_results,
)

__all__ = ["DirectoryResultWriter", "normalize_map_format"]

#: Habitat label dtype written to disk. v0.1 wrote label maps through
#: ``sitk.GetImageFromArray`` on an ``int32`` label array, and the golden
#: baseline hashes those bytes, so the dtype is part of the artefact
#: contract rather than an implementation detail.
_LABEL_DTYPE = np.int32

#: Identifier/metadata columns of the v0.1 units table, in canonical order
#: (feature columns follow). ``supervoxel`` and ``count`` exist only at
#: supervoxel/habitat granularity: a voxel-level row needs no voxel count
#: and has no partition id.
_SUBJECT_COLUMN = "subject"
_SUPERVOXEL_COLUMN = "supervoxel"
_HABITATS_COLUMN = "habitats"
_COUNT_COLUMN = "count"

#: Canonical stem -> file extension (including the leading dot) for habitat
#: and supervoxel label maps. SimpleITK selects the encoder from the path
#: suffix, so the extension IS the format contract.
_MAP_FORMAT_EXTENSIONS = {
    "nrrd": ".nrrd",
    "nii": ".nii",
    "nii.gz": ".nii.gz",
    "mha": ".mha",
    "mhd": ".mhd",
}


def _require_simpleitk() -> Any:
    """Import SimpleITK lazily so the adapter layer stays light to import."""
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:  # pragma: no cover - present in CI
        raise HABITAPIError(
            "SimpleITK is required to write image files to disk."
        ) from exc
    return sitk


def normalize_map_format(map_format: str) -> str:
    """
    Canonicalise a label-map on-disk format to a file extension.

    Accepted values (case-insensitive, leading dot optional)::

        nrrd | nii | nii.gz | mha | mhd

    Args:
        map_format: Format name or extension requested by the caller.

    Returns:
        Extension including the leading dot, e.g. ``".nii.gz"``.

    Raises:
        HABITAPIError: When ``map_format`` is not one of the supported values.
    """
    key = str(map_format).strip().lower().lstrip(".")
    try:
        return _MAP_FORMAT_EXTENSIONS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_MAP_FORMAT_EXTENSIONS))
        raise HABITAPIError(
            f"Unsupported map_format {map_format!r}; expected one of: {supported}."
        ) from exc


def _apply_geometry(image: Any, geometry: Geometry) -> None:
    """
    Stamp physical metadata onto a freshly created SimpleITK image.

    A label map without spacing/origin/direction is not merely untidy: it
    silently stops overlaying on the source series, and every downstream
    volume in physical units becomes wrong.

    Args:
        image: SimpleITK image created from a label array.
        geometry: Grid the label array refers to.
    """
    image.SetSpacing(tuple(float(v) for v in geometry.spacing))
    image.SetOrigin(tuple(float(v) for v in geometry.origin))
    image.SetDirection(tuple(float(v) for v in geometry.direction))


def _unit_assignments(
    units: Supervoxelization, habitat_map: HabitatMap
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive (unit ids, assigned habitat per unit, voxel count per unit).

    The assigner paints every voxel of a unit with that unit's habitat, so
    the habitat of a unit's first voxel (in stable sorted order) IS the
    unit's habitat. Re-deriving the assignment from the two label images --
    rather than replicating the assigner's internals -- keeps the table
    consistent with the written habitat map by construction.

    Args:
        units: One subject's clustering units.
        habitat_map: The same subject's habitat label image.

    Returns:
        Three equally sized arrays over the non-background units in
        ascending id order: unit ids, assigned habitat ids, voxel counts.
    """
    labels = np.asarray(units.label_array).ravel()
    habitats = np.asarray(habitat_map.label_array).ravel()
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    unique, first = np.unique(sorted_labels, return_index=True)
    keep = unique != 0
    unit_ids = unique[keep].astype(np.int64)
    if unit_ids.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return unit_ids, empty, empty
    assigned = habitats[order[first[keep]]].astype(np.int64)
    counts = np.bincount(
        labels[labels > 0], minlength=int(unit_ids.max()) + 1
    )
    return unit_ids, assigned, counts[unit_ids].astype(np.int64)


def _subject_units_frame(
    units: Supervoxelization, habitat_map: HabitatMap, granularity: str
) -> pd.DataFrame:
    """
    Build one subject's rows of the v0.1 units table.

    Args:
        units: The subject's clustering units.
        habitat_map: The subject's habitat label image.
        granularity: ``"supervoxel"`` (row per unit), ``"habitat"`` (row per
            assigned habitat, features pooled) or ``"voxel"`` (row per
            unit/voxel without partition metadata).
    """
    unit_ids, assigned, counts = _unit_assignments(units, habitat_map)
    # ``features`` is indexed by unit id; ``.loc`` keeps the table aligned
    # with the label image and raises loudly if a feature row is missing.
    features = units.features.loc[unit_ids].reset_index(drop=True)
    subject = units.subject_id
    if granularity == "supervoxel":
        frame = features.copy()
        frame.insert(0, _COUNT_COLUMN, counts)
        frame.insert(0, _HABITATS_COLUMN, assigned)
        frame.insert(0, _SUPERVOXEL_COLUMN, unit_ids)
        frame.insert(0, _SUBJECT_COLUMN, subject)
        return frame
    if granularity == "voxel":
        frame = features.copy()
        frame.insert(0, _HABITATS_COLUMN, assigned)
        frame.insert(0, _SUBJECT_COLUMN, subject)
        return frame
    # ``"habitat"``: pool the unit rows of each assigned habitat. One-step
    # habitats are defined inside their own subject, so a habitat never
    # spans subjects and grouping within this frame is exact; with
    # single-voxel units the pooled means are the cluster centroids.
    frame = features.copy()
    frame[_HABITATS_COLUMN] = assigned
    frame[_COUNT_COLUMN] = counts
    grouped = frame.groupby(_HABITATS_COLUMN, sort=True)
    pooled = grouped[list(features.columns)].mean()
    pooled_counts = grouped[_COUNT_COLUMN].sum()
    habitat_ids = pooled.index.to_numpy()
    out = pooled.reset_index(drop=True)
    out.insert(0, _COUNT_COLUMN, pooled_counts.to_numpy())
    out.insert(0, _HABITATS_COLUMN, habitat_ids)
    out.insert(0, _SUPERVOXEL_COLUMN, habitat_ids)
    out.insert(0, _SUBJECT_COLUMN, subject)
    return out


def _empty_units_frame(granularity: str) -> pd.DataFrame:
    """Return the header-only units table of a subject-less study."""
    meta = (
        [_SUBJECT_COLUMN, _HABITATS_COLUMN]
        if granularity == "voxel"
        else [
            _SUBJECT_COLUMN,
            _SUPERVOXEL_COLUMN,
            _HABITATS_COLUMN,
            _COUNT_COLUMN,
        ]
    )
    return pd.DataFrame(columns=meta)


class DirectoryResultWriter:
    """
    Write study artefacts into one directory, in the v0.1 layout.

    The layout is fixed here and nowhere else::

        <root>/<subject_id>_habitats.<ext>
        <root>/habitat_model.habitatmodel
        <root>/<name>.csv
        <root>/run_manifest.json

    ``<ext>`` defaults to ``nrrd`` (v0.1). Pass ``map_format`` to write
    NIfTI or MetaImage instead; SimpleITK chooses the encoder from the
    destination suffix.

    Args:
        root: Destination directory. Created on first write rather than in
            ``__init__``, so constructing a writer has no side effect -- a
            caller may build one, decide not to use it, and leave no empty
            directory behind. Named ``root`` to match
            :class:`~habit.adapters.directory.DirectoryDataSource`: a
            destination is a filesystem fact here, not a configuration
            setting.
        map_format: On-disk format for habitat and supervoxel label maps.
            One of ``"nrrd"`` (default), ``"nii"``, ``"nii.gz"``, ``"mha"``,
            ``"mhd"``. Leading dots are accepted (``".nii.gz"``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        map_format: str = "nrrd",
    ) -> None:
        self.root = Path(root)
        self.map_extension = normalize_map_format(map_format)

    def _destination(self, filename: str) -> Path:
        """Return an absolute path inside ``root``, creating it if needed."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / filename

    def write_habitat_map(self, habitat_map: HabitatMap) -> Optional[str]:
        """
        Write one subject's habitat label image.

        Args:
            habitat_map: Labels plus the grid they refer to.

        Returns:
            The path written (extension follows :attr:`map_extension`).
        """
        sitk = _require_simpleitk()
        destination = self._destination(
            f"{habitat_map.subject_id}_habitats{self.map_extension}"
        )
        array = np.ascontiguousarray(habitat_map.label_array, dtype=_LABEL_DTYPE)
        image = sitk.GetImageFromArray(array)
        _apply_geometry(image, habitat_map.geometry)
        sitk.WriteImage(image, str(destination))
        return str(destination)

    def write_feature_table(
        self, table: FeatureTable, name: str
    ) -> Optional[str]:
        """
        Write one feature table as CSV.

        Args:
            table: The table to persist.
            name: File stem, e.g. ``"habitat_features"``.

        Returns:
            The path written.
        """
        destination = self._destination(f"{name}.csv")
        table.frame.to_csv(destination, index=False)
        return str(destination)

    def write_supervoxel_map(self, units: Supervoxelization) -> Optional[str]:
        """
        Write one subject's supervoxel partition.

        Not part of the :class:`~habit.contracts.ops.ResultWriter` protocol:
        the partition map is a v0.1 reporting artefact (two-step training
        wrote ``<subject_id>_supervoxel.nrrd`` during clustering), derived
        from the study's clustering units rather than produced by the
        algorithms. Keeping it off the protocol lets third-party writers
        ignore it without structurally breaking the contract. The on-disk
        extension follows the writer's :attr:`map_extension`.

        Args:
            units: The subject's supervoxel partition.

        Returns:
            The path written.
        """
        sitk = _require_simpleitk()
        destination = self._destination(
            f"{units.subject_id}_supervoxel{self.map_extension}"
        )
        array = np.ascontiguousarray(units.label_array, dtype=_LABEL_DTYPE)
        image = sitk.GetImageFromArray(array)
        _apply_geometry(image, units.geometry)
        sitk.WriteImage(image, str(destination))
        return str(destination)

    def write_units_table(
        self,
        units: Sequence[Supervoxelization],
        habitat_maps: Sequence[HabitatMap],
        *,
        granularity: str,
        table_format: str = "parquet",
    ) -> Optional[str]:
        """
        Write the v0.1 ``habitats`` unit table derived from a study's
        clustering units and habitat maps.

        Like :meth:`write_supervoxel_map` this is a v0.1-layout extra beyond
        the writer protocol. Row granularity follows the recipe design:

        * ``"supervoxel"`` -- one row per clustering unit (two-step):
          ``subject, supervoxel, habitats, count, <features...>``.
        * ``"habitat"`` -- one row per assigned habitat within each subject
          (one-step, where units are single voxels and each defined cluster
          IS a habitat): same columns, aggregated per habitat.
        * ``"voxel"`` -- one row per ROI voxel (direct pooling):
          ``subject, habitats, <features...>``.

        Args:
            units: Per-subject clustering units, in cohort order.
            habitat_maps: Per-subject habitat label images, aligned with
                ``units``.
            granularity: ``"supervoxel"``, ``"habitat"`` or ``"voxel"``.
            table_format: ``"parquet"`` (v0.1 default) or ``"csv"``.

        Returns:
            The path written.

        Raises:
            HABITAPIError: On unknown granularity, length mismatch, or a
                units/map pair belonging to different subjects.
        """
        if granularity not in ("supervoxel", "habitat", "voxel"):
            raise HABITAPIError(
                f"Unknown units-table granularity {granularity!r}; expected "
                "'supervoxel', 'habitat' or 'voxel'."
            )
        normalize_habitats_results_format(table_format)
        if len(units) != len(habitat_maps):
            raise HABITAPIError(
                f"Cannot build the units table from {len(units)} unit sets "
                f"but {len(habitat_maps)} habitat maps; the two must align "
                "one per subject."
            )
        frames: List[pd.DataFrame] = []
        for subject_units, habitat_map in zip(units, habitat_maps):
            if subject_units.subject_id != habitat_map.subject_id:
                raise HABITAPIError(
                    "Units/habitat-map misalignment: units belong to "
                    f"{subject_units.subject_id!r} but the map belongs to "
                    f"{habitat_map.subject_id!r}."
                )
            frames.append(
                _subject_units_frame(subject_units, habitat_map, granularity)
            )
        table = (
            pd.concat(frames, ignore_index=True)
            if frames
            else _empty_units_frame(granularity)
        )
        destination = save_habitats_results(table, self.root, table_format)
        return str(destination)

    def write_habitat_model(self, model: HabitatModel) -> Optional[str]:
        """
        Write the fitted habitat definition in its versioned archive format.

        Args:
            model: The population-level habitat definition.

        Returns:
            The path written.
        """
        destination = self._destination("habitat_model.habitatmodel")
        model.save(destination)
        return str(destination)

    def write_manifest(self, manifest: RunManifest) -> Optional[str]:
        """
        Write the run manifest as JSON.

        Args:
            manifest: Provenance and reporting record for the run.

        Returns:
            The path written.
        """
        destination = self._destination("run_manifest.json")
        manifest.to_json(destination)
        return str(destination)
