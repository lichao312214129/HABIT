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
"""Shared on-disk layout for the v0.1-vs-v1.0 parity harness.

Both dump harnesses import ONLY this module from the parity package, so the
frozen v0.1 worktree is never asked to import anything from the v1 tree
beyond these pure-stdlib/pandas helpers. Every artefact is written in a
version-neutral format (parquet for tables, .npy for arrays, JSON for
scalars) so the comparator can load both legs without either library on
the path.

Directory layout produced per leg::

    <leg_dir>/
        ck1_voxel_raw/<subject>.parquet        # checkpoint 1
        ck1b_voxel_prep/<subject>.parquet      # subject-level preprocessed voxels
        ck2_units/<subject>.parquet            # per-subject clustering units
        ck2_units_cohort.parquet               # pooled units, cohort row order
        ck3_cohort_prep.parquet                # matrix entering k-means
        ck4_model.npz                          # centroids (raw + canonical order)
        ck5_labels/<subject>.npy               # habitat label volume (int32)
        ck6_habitat_features.parquet           # habitat feature table (optional)
        meta.json                              # roster, k, timings, versions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CK1_DIR = "ck1_voxel_raw"
CK1B_DIR = "ck1b_voxel_prep"
CK2_DIR = "ck2_units"
CK2_COHORT = "ck2_units_cohort.parquet"
CK3_COHORT_PREP = "ck3_cohort_prep.parquet"
CK4_MODEL = "ck4_model.npz"
CK5_DIR = "ck5_labels"
CK6_FEATURES = "ck6_habitat_features.parquet"
META = "meta.json"


def ensure_leg_dir(leg_dir: Path) -> Path:
    """
    Create the per-leg artefact tree.

    Args:
        leg_dir: Root directory for one leg's artefacts.

    Returns:
        The same path, with all checkpoint subdirectories created.
    """
    for name in (CK1_DIR, CK1B_DIR, CK2_DIR, CK5_DIR):
        (leg_dir / name).mkdir(parents=True, exist_ok=True)
    return leg_dir


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """
    Write a table preserving row order and column names.

    The frame's index is reset into an explicit ``_row`` column because
    parquet round-trips a RangeIndex but not every custom index, and row
    order is itself one of the parity checkpoints.

    Args:
        frame: Table to persist.
        path: Destination ``.parquet`` path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.reset_index(drop=False)
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    """
    Read a table written by :func:`write_table`.

    Args:
        path: Source ``.parquet`` path.

    Returns:
        The table with its stored row order intact.
    """
    return pd.read_parquet(path)


def write_array(array: np.ndarray, path: Path) -> None:
    """
    Persist a dense array.

    Args:
        array: Array to persist.
        path: Destination ``.npy`` path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def canonical_centroid_order(centroids: np.ndarray) -> np.ndarray:
    """
    Return the permutation sorting centroids into a canonical order.

    k-means labels are arbitrary up to permutation, so centroid tables can
    only be compared after both legs are put in the same order. Sorting
    lexicographically by the centroid coordinates is deterministic and does
    not depend on either leg's label numbering.

    Args:
        centroids: ``(k, n_features)`` centroid matrix.

    Returns:
        Index array of length ``k`` giving the canonical row order.
    """
    keys = tuple(centroids[:, i] for i in range(centroids.shape[1] - 1, -1, -1))
    return np.lexsort(keys)


def write_model(
    leg_dir: Path,
    centroids: np.ndarray,
    feature_names: Sequence[str],
) -> None:
    """
    Persist habitat centroids in both raw and canonical order.

    Args:
        leg_dir: Leg artefact root.
        centroids: ``(k, n_features)`` centroid matrix as fitted.
        feature_names: Column names matching the centroid columns.
    """
    order = canonical_centroid_order(centroids)
    np.savez(
        leg_dir / CK4_MODEL,
        centroids_raw=centroids,
        centroids_canonical=centroids[order],
        canonical_order=order,
        feature_names=np.asarray([str(n) for n in feature_names], dtype=object),
    )


def write_meta(leg_dir: Path, meta: Dict[str, Any]) -> None:
    """
    Persist the run's scalar metadata (roster, chosen k, versions, timings).

    Args:
        leg_dir: Leg artefact root.
        meta: JSON-serialisable metadata mapping.
    """
    (leg_dir / META).write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def read_meta(leg_dir: Path) -> Dict[str, Any]:
    """
    Load the metadata written by :func:`write_meta`.

    Args:
        leg_dir: Leg artefact root.

    Returns:
        The metadata mapping.
    """
    return json.loads((leg_dir / META).read_text(encoding="utf-8"))


def label_volume_from_units(
    mask_array: np.ndarray,
    voxel_unit_labels: np.ndarray,
    unit_to_habitat: Dict[int, int],
) -> np.ndarray:
    """
    Rebuild a 3-D habitat label volume from per-voxel unit ids.

    Written once and used by BOTH legs so the reconstruction itself can
    never be the source of a difference: each leg supplies only its own
    mask, its own per-voxel supervoxel ids, and its own supervoxel-to-
    habitat mapping.

    Args:
        mask_array: ROI mask volume; non-zero marks ROI voxels.
        voxel_unit_labels: 1-D supervoxel id per ROI voxel, in the same
            order as ``mask_array[mask_array > 0]``.
        unit_to_habitat: Mapping from supervoxel id to habitat label.

    Returns:
        Int32 volume with habitat labels inside the ROI and 0 outside.
    """
    roi = mask_array > 0
    volume = np.zeros(mask_array.shape, dtype=np.int32)
    mapped = np.array(
        [int(unit_to_habitat.get(int(u), 0)) for u in voxel_unit_labels],
        dtype=np.int32,
    )
    volume[roi] = mapped
    return volume


def subject_ids_from_dir(directory: Path) -> List[str]:
    """
    List subject ids from a per-subject artefact directory.

    Args:
        directory: Directory holding ``<subject>.parquet`` or ``<subject>.npy``.

    Returns:
        Sorted subject ids.
    """
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.iterdir() if p.is_file())


def optional_read_table(path: Path) -> Optional[pd.DataFrame]:
    """
    Read a table if it exists, else return ``None``.

    Args:
        path: Candidate ``.parquet`` path.

    Returns:
        The table, or ``None`` when the file is absent.
    """
    return read_table(path) if path.is_file() else None
