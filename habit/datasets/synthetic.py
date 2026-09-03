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
"""Deterministic synthetic cohort and feature-table builders for tests and demos."""

from __future__ import annotations

from typing import Dict, Literal, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from scipy import ndimage as ndi

from habit.contracts.geometry import Geometry
from habit.contracts.image import ArrayImageRef
from habit.contracts.outcome import BinaryOutcome, SurvivalOutcome
from habit.contracts.provenance import Provenance
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable

__all__ = ["make_synthetic_cohort", "make_synthetic_feature_table"]

#: Default ROI key used by the fast golden specs.
_DEFAULT_ROI = "tumor"

#: Realism presets for :func:`make_synthetic_cohort`.
RealismLevel = Literal["demo", "legacy"]


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


def _fit_shape(field: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Crop or pad a volume so it matches ``shape`` exactly.

    Args:
        field: Source array that may differ by a few voxels after zoom.
        shape: Target shape ``(z, y, x)``.

    Returns:
        Array with exactly ``shape``.
    """
    out = np.zeros(shape, dtype=np.float64)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(field.shape, shape))
    out[slices] = field[slices]
    return out


def _low_frequency_field(
    shape: Tuple[int, int, int],
    rng: np.random.Generator,
    *,
    scale: float = 1.0,
    coarseness: float = 0.22,
) -> np.ndarray:
    """
    Build a smooth spatial field via coarse Gaussian noise + zoom.

    Used for soft tissue background, lesion-boundary wobble, and weak
    intra-lesion texture. Pure low-frequency content reads as "MRI-like"
    shading in napari without looking like white noise.

    Args:
        shape: Output shape ``(z, y, x)``.
        rng: Per-subject random generator.
        scale: Standard deviation of the coarse noise before smoothing.
        coarseness: Fraction of each axis used for the coarse grid
            (smaller → smoother fields).

    Returns:
        Float64 volume with mean near zero and amplitude near ``scale``.
    """
    coarse_shape = tuple(max(2, int(round(dim * coarseness))) for dim in shape)
    coarse = rng.normal(loc=0.0, scale=scale, size=coarse_shape).astype(np.float64)
    # Light blur on the coarse grid so zoom artefacts stay soft.
    coarse = ndi.gaussian_filter(coarse, sigma=0.6)
    factors = tuple(float(dim) / float(coarse_dim) for dim, coarse_dim in zip(shape, coarse_shape))
    field = ndi.zoom(coarse, factors, order=1)
    return _fit_shape(field, shape)


def _irregular_lesion_mask(
    shape: Tuple[int, int, int],
    rng: np.random.Generator,
    *,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
) -> np.ndarray:
    """
    Build a slightly irregular blob mask around an ellipsoid core.

    A low-frequency field warps the ellipsoid iso-surface so the boundary
    is not a perfect math ellipse, then a tiny binary close fills one-voxel
    holes without erasing the irregularity.

    Args:
        shape: Volume shape ``(z, y, x)``.
        rng: Per-subject random generator.
        center: Nominal ellipsoid centre ``(z, y, x)``.
        radii: Nominal semi-axes ``(z, y, x)``.

    Returns:
        ``int32`` binary mask aligned with the synthetic lesion.
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
    # Surface wobble: stronger near the boundary, weak in the core.
    wobble = _low_frequency_field(shape, rng, scale=0.18, coarseness=0.28)
    warped = normalised * (1.0 + wobble)
    mask = (warped <= 1.0).astype(np.int32)
    if int(mask.sum()) == 0:
        # Extremely unlucky seed: fall back to a clean ellipsoid.
        return _ellipsoid_mask(shape, center=center, radii=radii)
    # Close 1-voxel gaps; keep the largest connected component as the lesion.
    mask = ndi.binary_closing(mask, iterations=1).astype(np.int32)
    labeled, n_components = ndi.label(mask)
    if n_components > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        keep = int(np.argmax(counts))
        mask = (labeled == keep).astype(np.int32)
    return mask


def _label_subregions(
    mask: np.ndarray,
    n_subregions: int,
) -> np.ndarray:
    """
    Partition a binary mask into ``n_subregions`` contiguous z-bands.

    Legacy path used by ``realism="legacy"``. Subregions are assigned along
    the z axis so clustering sees separated intensity profiles.

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


def _blob_subregions(
    mask: np.ndarray,
    n_subregions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Partition a binary mask into organic Voronoi-like intensity blobs.

    Seeds are sampled inside the ROI; each ROI voxel inherits the label of
    the nearest seed. This keeps ``n_subregions`` separable habitats while
    looking less artificial than pure z-bands in axial napari views.

    Args:
        mask: Binary ROI mask.
        n_subregions: Number of subregions to carve out.
        rng: Per-subject random generator.

    Returns:
        Label array with background ``0`` and subregion ids ``1..n_subregions``.
    """
    if n_subregions < 1:
        raise ValueError(f"n_subregions must be positive; got {n_subregions}.")
    labels = np.zeros_like(mask, dtype=np.int32)
    roi_coords = np.argwhere(mask > 0)
    if roi_coords.size == 0:
        return labels
    n_roi = int(roi_coords.shape[0])
    if n_roi <= n_subregions:
        # Tiny ROIs: fall back to sequential labels on available voxels.
        for index, coord in enumerate(roi_coords):
            labels[tuple(coord)] = (index % n_subregions) + 1
        return labels

    # Spread seeds: pick far-apart voxels via greedy max-min sampling.
    first = int(rng.integers(0, n_roi))
    seed_indices = [first]
    # Squared distances from every ROI voxel to the chosen seeds.
    chosen = roi_coords[first].astype(np.float64)
    min_dist2 = np.sum((roi_coords.astype(np.float64) - chosen) ** 2, axis=1)
    for _ in range(1, n_subregions):
        next_index = int(np.argmax(min_dist2))
        seed_indices.append(next_index)
        chosen = roi_coords[next_index].astype(np.float64)
        dist2 = np.sum((roi_coords.astype(np.float64) - chosen) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

    seeds = roi_coords[np.asarray(seed_indices, dtype=np.int64)].astype(np.float64)
    # Assign each ROI voxel to the nearest seed (Voronoi).
    delta = roi_coords.astype(np.float64)[:, None, :] - seeds[None, :, :]
    dist2 = np.sum(delta ** 2, axis=2)
    nearest = np.argmin(dist2, axis=1).astype(np.int32) + 1
    labels[tuple(roi_coords.T)] = nearest
    return labels


def _modality_weights(
    modality: str,
    modality_index: int,
) -> Tuple[float, float, float]:
    """
    Return ``(tissue, lesion, noise)`` mixing weights for one modality name.

    T1-like volumes emphasise lesion brightness gradients; T2-like volumes
    invert that contrast so the two channels are correlated but not copies.

    Args:
        modality: Modality key (case-insensitive prefix match for T1/T2).
        modality_index: Position in the modalities tuple (fallback weighting).

    Returns:
        Triple of non-negative mixing weights.
    """
    name = modality.upper()
    if name.startswith("T1"):
        return (0.45, 1.00, 0.035)
    if name.startswith("T2"):
        # Stronger tissue share so T2 is not a near-perfect invert of T1.
        return (0.85, 0.55, 0.040)
    # Unknown modality: mild tissue + lesion with a small index offset.
    return (0.60, 0.90 + 0.05 * modality_index, 0.035)


def _demo_subject_arrays(
    rng: np.random.Generator,
    *,
    shape: Tuple[int, int, int],
    modalities: Sequence[str],
    n_subregions: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Synthesise demo-realistic image arrays and an aligned lesion mask.

    The volumes are *not* clinical anatomy — they are smooth tissue fields
    with an irregular blob lesion and 2–3 intra-lesion intensity habitats so
    clustering / napari demos look plausible.

    Args:
        rng: Per-subject random generator.
        shape: Volume shape ``(z, y, x)``.
        modalities: Modality names to populate.
        n_subregions: Number of distinguishable habitats inside the ROI.

    Returns:
        ``(images_dict, mask)`` where ``images_dict`` maps modality → float64
        array and ``mask`` is the binary lesion mask.
    """
    # Slight per-subject jitter of lesion placement and size.
    centre = tuple(
        float(dim) * (0.50 + float(rng.uniform(-0.04, 0.04))) for dim in shape
    )
    radii = tuple(
        max(2.0, float(dim) * float(rng.uniform(0.28, 0.38))) for dim in shape
    )
    mask = _irregular_lesion_mask(shape, rng, center=centre, radii=radii)
    subregion_labels = _blob_subregions(mask, n_subregions, rng)

    # Soft tissue background: slow spatial shading, never pure zeros.
    tissue = 0.45 + 0.12 * _low_frequency_field(
        shape, rng, scale=1.0, coarseness=0.18
    )
    tissue = np.clip(tissue, 0.15, 0.85)

    # Per-modality region means: shared spatial habitats, different contrast
    # weighting. Region 2 rises on both channels so T1/T2 are not mirrors.
    region_profiles = {
        1: (0.95, 1.45),
        2: (1.55, 2.05),
        3: (2.25, 1.05),
    }
    texture = _low_frequency_field(shape, rng, scale=0.08, coarseness=0.35)

    # Weak intensity falloff just outside the lesion (partial-volume halo).
    dist_out = ndi.distance_transform_edt(mask == 0)
    halo_sigma = max(1.2, 0.06 * float(max(shape)))
    halo = np.exp(-dist_out / halo_sigma)

    subject_offset = float(rng.normal(scale=0.04))
    images: Dict[str, np.ndarray] = {}
    for modality_index, modality in enumerate(modalities):
        tissue_w, _lesion_w, noise_w = _modality_weights(modality, modality_index)
        name = modality.upper()
        use_t1 = name.startswith("T1") or not name.startswith("T2")
        tissue_term = tissue * tissue_w + (0.0 if use_t1 else 0.08)
        halo_gain = 0.55 if use_t1 else 0.40

        lesion_term = np.zeros(shape, dtype=np.float64)
        for region_id in range(1, n_subregions + 1):
            t1_base, t2_base = region_profiles.get(
                region_id, (1.0 + 0.4 * region_id, 1.5 - 0.2 * region_id)
            )
            base = t1_base if use_t1 else t2_base
            region_mask = subregion_labels == region_id
            lesion_term[region_mask] = base + texture[region_mask]

        lesion_edge = (
            float(np.mean(lesion_term[mask > 0])) if int(mask.sum()) else 1.2
        )
        falloff = halo * (0.35 * lesion_edge)

        array = tissue_term + subject_offset + modality_index * 0.03
        array = array + falloff * halo_gain
        # Inside ROI: mostly habitat means + a little tissue shading.
        inside_tissue_mix = 0.22 if use_t1 else 0.35
        array[mask > 0] = (
            tissue_term[mask > 0] * inside_tissue_mix
            + lesion_term[mask > 0]
            + subject_offset
            + modality_index * 0.05
        )
        array = array + rng.normal(scale=noise_w, size=shape)
        images[modality] = np.asarray(array, dtype=np.float64)

    return images, mask


def _legacy_subject_arrays(
    rng: np.random.Generator,
    *,
    shape: Tuple[int, int, int],
    modalities: Sequence[str],
    n_subregions: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Reproduce the pre-demo synthetic volumes (flat ROI, zero background).

    Kept so callers can opt into the old deterministic look via
    ``realism="legacy"`` without pinning demo aesthetics in tests that only
    need separable habitats.

    Args:
        rng: Per-subject random generator.
        shape: Volume shape ``(z, y, x)``.
        modalities: Modality names to populate.
        n_subregions: Number of z-band subregions inside the ROI.

    Returns:
        ``(images_dict, mask)`` with the legacy intensity model.
    """
    centre = tuple(float(v) / 2.0 for v in shape)
    radii = tuple(max(2.0, float(v) * 0.35) for v in shape)
    mask = _ellipsoid_mask(shape, center=centre, radii=radii)
    subregion_labels = _label_subregions(mask, n_subregions)

    subject_offset = float(rng.normal(scale=0.05))
    region_profiles = {
        1: (0.8, 1.6),
        2: (1.5, 1.0),
        3: (2.2, 0.6),
    }
    images: Dict[str, np.ndarray] = {}
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
        images[modality] = array
    return images, mask


def _subject_from_seed(
    subject_id: str,
    seed_seq: SeedSequence,
    *,
    shape: Tuple[int, int, int],
    modalities: Sequence[str],
    n_subregions: int,
    realism: RealismLevel,
) -> Subject:
    """
    Build one synthetic subject from a dedicated child seed sequence.

    Args:
        subject_id: Unique identifier within the cohort.
        seed_seq: Child seed derived from the cohort master sequence.
        shape: Volume shape ``(z, y, x)``.
        modalities: Modality names to populate.
        n_subregions: Number of distinguishable subregions inside the ROI.
        realism: ``"demo"`` for soft tissue + irregular lesion, or
            ``"legacy"`` for the flat zero-background volumes.

    Returns:
        A subject with in-memory image and mask references.
    """
    rng = np.random.default_rng(seed_seq)
    geometry = Geometry.from_array(shape)
    if realism == "legacy":
        arrays, mask = _legacy_subject_arrays(
            rng,
            shape=shape,
            modalities=modalities,
            n_subregions=n_subregions,
        )
    elif realism == "demo":
        arrays, mask = _demo_subject_arrays(
            rng,
            shape=shape,
            modalities=modalities,
            n_subregions=n_subregions,
        )
    else:
        raise ValueError(
            f"Unsupported realism {realism!r}; expected 'demo' or 'legacy'."
        )

    images = {
        modality: ArrayImageRef(array=array, geometry=geometry)
        for modality, array in arrays.items()
    }
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
    *,
    realism: RealismLevel = "demo",
) -> Cohort:
    """
    Build a deterministic imaging cohort entirely in memory.

    Each subject receives its own child seed from ``SeedSequence.spawn`` so
    that per-subject noise differs while the cohort remains reproducible for
    a fixed master seed. The ROI contains ``n_subregions`` intensity blobs
    with distinct modality profiles to support multi-habitat clustering.

    With the default ``realism="demo"``, volumes look like soft tissue plus
    an irregular lesion blob (demo-realistic, **not** clinical anatomy).
    Pass ``realism="legacy"`` for the older flat ROI / zero-background look.

    Args:
        n_subjects: Number of subjects to synthesise.
        modalities: Modality keys attached to every subject.
        shape: Cubic grid shape ``(z, y, x)``.
        n_subregions: Number of distinguishable subregions inside the ROI.
        rng: Master seed controlling the entire cohort.
        realism: ``"demo"`` (default) or ``"legacy"``.

    Returns:
        A :class:`~habit.contracts.subject.Cohort` with subjects ``subj001``,
        ``subj002``, ...

    Examples:
        >>> from habit.datasets import make_synthetic_cohort
        >>> cohort = make_synthetic_cohort(n_subjects=3, rng=42)
        >>> len(cohort)
        3
        >>> cohort.subject_ids
        ('subj001', 'subj002', 'subj003')
        >>> subject = cohort[0]
        >>> sorted(subject.images), sorted(subject.masks)
        (['T1', 'T2'], ['tumor'])
    """
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be positive; got {n_subjects}.")
    if realism not in ("demo", "legacy"):
        raise ValueError(
            f"Unsupported realism {realism!r}; expected 'demo' or 'legacy'."
        )
    master = _coerce_seed_sequence(rng)
    child_seeds = master.spawn(n_subjects)
    subjects = [
        _subject_from_seed(
            f"subj{index + 1:03d}",
            child_seeds[index],
            shape=shape,
            modalities=modalities,
            n_subregions=n_subregions,
            realism=realism,
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

    Examples:
        >>> from habit.datasets import make_synthetic_feature_table
        >>> table = make_synthetic_feature_table(n_rows=10, n_features=4, rng=42)
        >>> table.frame.shape
        (10, 6)
        >>> list(table.feature_columns)
        ['signal', 'noise0', 'noise1', 'noise2']
        >>> table.outcome.task
        'binary'
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
