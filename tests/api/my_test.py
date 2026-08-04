"""Runnable minimal example of the HABIT v1.0 in-memory API.

Nothing here touches the filesystem: the cohort is built from NumPy arrays,
fitted into a population-level habitat model, and applied back to a subject
through a ``SubjectPipeline``. Run it directly:

    & "E:\\conda\\mconda\\envs\\py310\\python.exe" tests/api/my_test.py
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from habit.contracts import (
    ArrayImageRef,
    Cohort,
    Geometry,
    ImageVolume,
    MaskVolume,
    Subject,
)
from habit.domain import (
    HabitatVolumeFeatures,
    IthHabitatFeatures,
    KMeansHabitatModelFitter,
    MsiHabitatFeatures,
    RawVoxelFeatures,
    SlicSupervoxelizer,
    SubjectPipeline,
)

# Small enough to run in a couple of seconds, large enough for SLIC to split.
SHAPE: Tuple[int, int, int] = (12, 24, 24)
MODALITIES: Tuple[str, ...] = ("T1", "T2")


def make_geometry(shape: Tuple[int, int, int]) -> Geometry:
    """
    Build a 1 mm isotropic geometry for a NumPy grid.

    ``shape`` follows the NumPy axis order ``(z, y, x)`` while ``spacing`` /
    ``origin`` / ``direction`` keep the SimpleITK axis order ``(x, y, z)``,
    so a round trip through ``SimpleITK.Image`` never transposes metadata.

    Args:
        shape: Voxel grid size in NumPy axis order ``(z, y, x)``.

    Returns:
        The geometry describing that grid.
    """
    return Geometry(
        shape=shape,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def make_subject(
    subject_id: str,
    *,
    seed: int,
    shape: Tuple[int, int, int] = SHAPE,
    modalities: Sequence[str] = MODALITIES,
) -> Subject:
    """
    Build one synthetic subject with two spatially separated intensity blobs.

    Images and masks are stored as ``ArrayImageRef`` -- the in-memory
    implementation of the lazy ``ImageRef`` protocol that ``Subject`` holds.
    They are materialised into ``ImageVolume`` / ``MaskVolume`` only when
    ``subject.image(...)`` / ``subject.mask(...)`` is called.

    Args:
        subject_id: Identifier, unique within the cohort.
        seed: Seed of the additive noise, so every subject differs slightly.
        shape: Voxel grid size in NumPy axis order ``(z, y, x)``.
        modalities: Modality keys to synthesise.

    Returns:
        The assembled subject.
    """
    rng = np.random.RandomState(seed)
    geometry = make_geometry(shape)
    half_z = shape[0] // 2

    images = {}
    for offset, modality in enumerate(modalities):
        array = np.zeros(shape, dtype=np.float64)
        array[:half_z] = 1.0                        # low-intensity compartment
        array[half_z:] = 10.0                       # high-intensity compartment
        array += rng.normal(scale=0.05, size=shape) + offset
        images[modality] = ArrayImageRef(array=array, geometry=geometry)

    # The mask MUST carry integer labels; 0 is background. A float array such
    # as np.random.rand(...) yields an EMPTY ROI, because label inference
    # truncates every value in [0, 1) to 0.
    mask = np.zeros(shape, dtype=np.int32)
    mask[1:-1, 2:-2, 2:-2] = 1
    return Subject(
        subject_id=subject_id,
        images=images,
        masks={"tumor": ArrayImageRef(array=mask, geometry=geometry)},
        metadata={"center": "A" if seed % 2 == 0 else "B"},
    )


def materialised_volumes_demo() -> None:
    """
    Show the eager counterpart of ``ArrayImageRef``.

    ``ImageVolume`` / ``MaskVolume`` subclass the stable public classes of
    ``habit.api.image``, whose constructor takes ``spacing`` / ``origin`` /
    ``direction`` separately rather than one ``Geometry`` object. Use
    ``from_geometry`` when a ``Geometry`` value is already at hand.
    """
    geometry = make_geometry(SHAPE)
    image = ImageVolume.from_geometry(
        np.random.rand(*SHAPE), geometry, modality="T1"
    )
    mask = MaskVolume.from_geometry(
        np.ones(SHAPE, dtype=np.int32), geometry, roi_name="tumor"
    )
    print(
        f"[volumes] image shape={image.data.shape} "
        f"geometry_matches={image.geometry == geometry} "
        f"mask labels={mask.labels} roi={mask.roi_name}"
    )


def main() -> None:
    """Fit a habitat model on an in-memory cohort and apply it to a subject."""
    materialised_volumes_demo()

    # 1) Data: an ordered container of lazy subjects, no directory convention.
    cohort = Cohort([make_subject(f"P{i:03d}", seed=i) for i in range(4)])
    print(f"[cohort] n={len(cohort)} ids={list(cohort.subject_ids)}")

    # 2) Subject-level operators: each is a single-argument callable.
    # SLIC is deterministic, so it does not implement Seedable -- only the
    # stochastic components below expose set_random_state.
    voxel_features = RawVoxelFeatures(modalities=list(MODALITIES))
    supervoxelizer = SlicSupervoxelizer(n_supervoxels=20)

    units = [supervoxelizer(voxel_features(subject)) for subject in cohort]
    print(f"[units] supervoxels of first subject={len(units[0].features)}")

    # 3) The only cohort-level step: the habitat definition must be shared.
    fitter = KMeansHabitatModelFitter(n_habitats=3)
    fitter.set_random_state(42)
    model = fitter.fit(units, cohort=cohort)
    print(f"[model] id={model.model_id} n_habitats={model.n_habitats}")

    # 4) Compose the subject-level chain into one reusable callable. This
    #    object plus the model is exactly what external validation needs.
    pipeline = SubjectPipeline(voxel_features, supervoxelizer, model.assigner())

    unseen = make_subject("P999", seed=99)
    habitat_map = pipeline(unseen)
    present = sorted(int(v) for v in np.unique(habitat_map.label_array) if v != 0)
    print(f"[map] subject={habitat_map.subject_id} habitats_present={present}")

    # 5) Habitat-level features; families are joined into one table.
    table = pipeline.extract_features(
        unseen,
        [MsiHabitatFeatures(), IthHabitatFeatures(), HabitatVolumeFeatures()],
    )
    print(f"[table] rows={table.frame.shape[0]} features={len(table.feature_columns)}")
    print(f"[table] ith_score={table.frame.iloc[0]['ith_score']:.4f}")

    # 6) The same pipeline over the whole cohort, in cohort order.
    maps = cohort.map(pipeline)
    print(f"[cohort.map] produced {len(maps)} habitat maps")


if __name__ == "__main__":
    main()


import numpy as np
from habit.contracts import Cohort, Geometry, ImageVolume, MaskVolume, Subject

geom = Geometry.from_array((32, 64, 64))
subject = Subject(
    subject_id="P001",
    images={
        "T1": ImageVolume(np.zeros(geom.shape, dtype=np.float32), geom),
        "T2": ImageVolume(np.zeros(geom.shape, dtype=np.float32), geom),
    },
    masks={"tumor": MaskVolume(np.ones(geom.shape, dtype=np.uint8), geom)},
    metadata={"center": "A"},
)
cohort = Cohort([subject], name="synthetic")
print(len(cohort), cohort.subject_ids)
fingerprint = cohort.summarize()  # -> CohortFingerprint