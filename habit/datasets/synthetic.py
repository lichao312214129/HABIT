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
"""Deterministic synthetic cohort and feature-table builders for tests."""

from __future__ import annotations

from typing import Literal, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import SeedSequence

from habit.contracts.geometry import Geometry
from habit.contracts.image import ArrayImageRef
from habit.contracts.outcome import BinaryOutcome, SurvivalOutcome
from habit.contracts.provenance import Provenance
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable

__all__ = ["make_synthetic_cohort", "make_synthetic_feature_table"]

#: Default ROI key used by the fast golden specs.
_DEFAULT_ROI = "tumor"


def _coerce_seed_sequence(
    rng: Union[int, np.random.Generator, SeedSequence],
) -> SeedSequence:
    """
    Normalise a caller seed into a :class:`SeedSequence`.

    Args:
        rng: Integer seed, an existing generator seed sequence, or a NumPy
            generator whose bit-generator seed is reused.

    Returns:
        The master seed sequence for cohort construction.
    """
    if isinstance(rng, SeedSequence):
        return rng
    if isinstance(rng, np.random.Generator):
        state = rng.bit_generator.state
        if isinstance(state, dict) and "state" in state:
            return SeedSequence(int(state["state"]["state"][0]))
        return SeedSequence(int(rng.integers(0, 2**32 - 1)))
    return SeedSequence(int(rng))


def _coerce_generator(
    rng: Union[int, np.random.Generator, SeedSequence],
) -> np.random.Generator:
    """
    Normalise a caller seed into a NumPy generator.

    Args:
        rng: Integer seed, seed sequence, or an existing generator.

    Returns:
        A generator ready for table construction.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, SeedSequence):
        return np.random.default_rng(rng)
    return np.random.default_rng(int(rng))


def _ellipsoid_mask(
    shape: Tuple[int, int, int],
    *,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
) -> np.ndarray:
    """
    Build a binary ellipsoid mask inside a cubic grid.

    Args:
        shape: Volume shape ``(z, y, x)``.
        center: Ellipsoid centre in voxel coordinates ``(z, y, x)``.
        radii: Semi-axes along ``(z, y, x)``.

    Returns:
        ``int32`` mask with ones inside the ellipsoid and zeros elsewhere.
    """
    grid_z, grid_y, grid_x = np.mgrid[
        0 : shape[0], 0 : shape[1], 0 : shape[2]
    ]
    cz, cy, cx = center
    rz, ry, rx = radii
    normalised = (
        ((grid_z - cz) / max(rz, 1.0)) ** 2
        + ((grid_y - cy) / max(ry, 1.0)) ** 2
        + ((grid_x - cx) / max(rx, 1.0)) ** 2
    )
    return (normalised <= 1.0).astype(np.int32)


def _label_subregions(
    mask: np.ndarray,
    n_subregions: int,
) -> np.ndarray:
    """
    Partition a binary mask into ``n_subregions`` contiguous bands.

    Subregions are assigned along the z axis so that downstream clustering
    sees three spatially separated intensity profiles inside the ROI.

    Args:
        mask: Binary ROI mask.
        n_subregions: Number of subregions to carve out.

    Returns:
        Label array with background ``0`` and subregion ids ``1..n_subregions``.
    """
    if n_subregions < 1:
        raise ValueError(f"n_subregions must be positive; got {n_subregions}.")
    labels = np.zeros_like(mask, dtype=np.int32)
    roi_coords = np.argwhere(mask > 0)
    if roi_coords.size == 0:
        return labels
    z_values = roi_coords[:, 0]
    z_min = int(z_values.min())
    z_max = int(z_values.max())
    edges = np.linspace(z_min, z_max + 1, n_subregions + 1)
    z_size = mask.shape[0]
    z_index = np.arange(z_size)[:, None, None]
    for region_id in range(1, n_subregions + 1):
        lower = edges[region_id - 1]
        upper = edges[region_id]
        band = (mask > 0) & (z_index >= lower) & (z_index < upper)
        labels[band] = region_id
    return labels


def _subject_from_seed(
    subject_id: str,
    seed_seq: SeedSequence,
    *,
    shape: Tuple[int, int, int],
    modalities: Sequence[str],
    n_subregions: int,
) -> Subject:
    """
    Build one synthetic subject from a dedicated child seed sequence.

    Args:
        subject_id: Unique identifier within the cohort.
        seed_seq: Child seed derived from the cohort master sequence.
        shape: Volume shape ``(z, y, x)``.
        modalities: Modality names to populate.
        n_subregions: Number of distinguishable subregions inside the ROI.

    Returns:
        A subject with in-memory image and mask references.
    """
    rng = np.random.default_rng(seed_seq)
    geometry = Geometry.from_array(shape)
    centre = tuple(float(v) / 2.0 for v in shape)
    radii = tuple(max(2.0, float(v) * 0.35) for v in shape)
    mask = _ellipsoid_mask(shape, center=centre, radii=radii)
    subregion_labels = _label_subregions(mask, n_subregions)

    # Per-subject baseline offsets keep subjects separable while preserving
    # the three within-ROI intensity profiles that clustering relies on.
    subject_offset = float(rng.normal(scale=0.05))
    region_profiles = {
        1: (0.8, 1.6),
        2: (1.5, 1.0),
        3: (2.2, 0.6),
    }
    images = {}
    for modality_index, modality in enumerate(modalities):
        array = np.zeros(shape, dtype=np.float64)
        for region_id in range(1, n_subregions + 1):
            t1_base, t2_base = region_profiles.get(
                region_id, (float(region_id), float(modality_index + 1))
            )
            base = t1_base if modality.upper().startswith("T1") else t2_base
            region_mask = subregion_labels == region_id
            array[region_mask] = base + subject_offset + modality_index * 0.15
        array += rng.normal(scale=0.02, size=shape)
        array[mask == 0] = 0.0
        images[modality] = ArrayImageRef(array=array, geometry=geometry)

    return Subject(
        subject_id=subject_id,
        images=images,
        masks={_DEFAULT_ROI: ArrayImageRef(array=mask, geometry=geometry)},
    )


def make_synthetic_cohort(
    n_subjects: int = 4,
    modalities: Tuple[str, ...] = ("T1", "T2"),
    shape: Tuple[int, int, int] = (32, 32, 32),
    n_subregions: int = 3,
    rng: Union[int, np.random.Generator, SeedSequence] = 0,
) -> Cohort:
    """
    Build a deterministic imaging cohort entirely in memory.

    Each subject receives its own child seed from ``SeedSequence.spawn`` so
    that per-subject noise differs while the cohort remains reproducible for
    a fixed master seed. The ROI contains ``n_subregions`` z-bands with
    distinct modality profiles to support three-habitat clustering tests.

    Args:
        n_subjects: Number of subjects to synthesise.
        modalities: Modality keys attached to every subject.
        shape: Cubic grid shape ``(z, y, x)``.
        n_subregions: Number of distinguishable subregions inside the ROI.
        rng: Master seed controlling the entire cohort.

    Returns:
        A :class:`~habit.contracts.subject.Cohort` with subjects ``subj001``,
        ``subj002``, ...
    """
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be positive; got {n_subjects}.")
    master = _coerce_seed_sequence(rng)
    child_seeds = master.spawn(n_subjects)
    subjects = [
        _subject_from_seed(
            f"subj{index + 1:03d}",
            child_seeds[index],
            shape=shape,
            modalities=modalities,
            n_subregions=n_subregions,
        )
        for index in range(n_subjects)
    ]
    return Cohort(subjects, name="synthetic")


def make_synthetic_feature_table(
    n_rows: int = 60,
    n_features: int = 12,
    task: Literal["binary", "survival"] = "binary",
    rng: Union[int, np.random.Generator, SeedSequence] = 0,
) -> FeatureTable:
    """
    Build a deterministic tabular dataset for fast ML golden tests.

    A single ``signal`` feature separates the endpoint classes or correlates
    with survival; the remaining columns are pure noise so selectors and
    classifiers have a stable correct answer to recover.

    Args:
        n_rows: Number of subjects / rows.
        n_features: Total number of model-input columns including ``signal``.
        n_features must be at least ``1``.
        task: ``"binary"`` for a single label column or ``"survival"`` for
            follow-up time plus an event indicator.
        rng: Seed controlling feature noise and endpoint draws.

    Returns:
        A :class:`~habit.contracts.table.FeatureTable` with explicit outcome
        semantics.
    """
    if n_rows < 2:
        raise ValueError(f"n_rows must be at least 2; got {n_rows}.")
    if n_features < 1:
        raise ValueError(f"n_features must be positive; got {n_features}.")
    generator = _coerce_generator(rng)
    subject_ids = [f"subj{index + 1:03d}" for index in range(n_rows)]
    frame_data = {"subject": subject_ids}

    if task == "binary":
        labels = (np.arange(n_rows) % 2).astype(int)
        signal = generator.normal(loc=0.0, scale=0.5, size=n_rows) + labels * 2.0
        frame_data["signal"] = signal
        noise_count = max(0, n_features - 1)
        for index in range(noise_count):
            frame_data[f"noise{index}"] = generator.normal(size=n_rows)
        frame_data["label"] = labels
        feature_columns = tuple(
            column for column in frame_data if column not in ("subject", "label")
        )
        outcome = BinaryOutcome(column="label", positive_label=1)
    elif task == "survival":
        event = (np.arange(n_rows) % 2).astype(int)
        signal = generator.normal(loc=0.0, scale=0.5, size=n_rows) + event * 1.5
        frame_data["signal"] = signal
        noise_count = max(0, n_features - 1)
        for index in range(noise_count):
            frame_data[f"noise{index}"] = generator.normal(size=n_rows)
        # Higher signal rows tend to experience the event earlier.
        base_time = generator.uniform(6.0, 36.0, size=n_rows)
        time = np.where(event == 1, base_time - signal, base_time + signal)
        time = np.clip(time, 0.5, None)
        frame_data["time"] = time
        frame_data["event"] = event
        feature_columns = tuple(
            column for column in frame_data if column not in ("subject", "time", "event")
        )
        outcome = SurvivalOutcome(time_column="time", event_column="event", event_value=1)
    else:
        raise ValueError(f"Unsupported task {task!r}; expected 'binary' or 'survival'.")

    provenance = Provenance.source("habit.datasets.synthetic")
    return FeatureTable(
        frame=pd.DataFrame(frame_data),
        id_columns=("subject",),
        feature_columns=feature_columns,
        outcome=outcome,
        provenance=provenance,
    )
