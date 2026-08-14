#!/usr/bin/env python
"""
Build Subject / Cohort from NumPy arrays (no HABIT directory layout).

Third-party pipelines often already hold ``(z, y, x)`` arrays from nibabel,
SimpleITK, or MONAI. This script shows the contracts-layer bridge:

* :class:`~habit.contracts.Geometry` — spatial frame
* :class:`~habit.contracts.ArrayImageRef` — lazy in-memory ImageRef
* :class:`~habit.contracts.ImageVolume` / :class:`~habit.contracts.MaskVolume`
* :class:`~habit.contracts.Subject` / :class:`~habit.contracts.Cohort`

Accompanies ``docs/source/examples/data_from_arrays.rst``.

Run from the repository root::

    python docs/source/examples/scripts/data_from_arrays_demo.py
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
from habit.domain import RawVoxelFeatures

# BEGIN example
SHAPE: Tuple[int, int, int] = (16, 16, 16)
MODALITIES: Sequence[str] = ("T1", "T2")


def make_subject_from_arrays(subject_id: str, *, seed: int) -> Subject:
    """
    Assemble one Subject from synthetic NumPy arrays.

    Args:
        subject_id: Unique id within the cohort.
        seed: RNG seed for modality noise.

    Returns:
        Subject with lazy ``ArrayImageRef`` handles for images and mask.
    """
    rng = np.random.RandomState(seed)
    geometry = Geometry.from_array(SHAPE, spacing=(1.0, 1.0, 1.0))
    half = SHAPE[0] // 2

    images = {}
    for offset, modality in enumerate(MODALITIES):
        array = np.zeros(SHAPE, dtype=np.float64)
        array[:half] = 1.0
        array[half:] = 8.0 + offset
        array += rng.normal(scale=0.05, size=SHAPE)
        images[modality] = ArrayImageRef(array=array, geometry=geometry)

    # Masks must be integer labels; 0 = background.
    mask = np.zeros(SHAPE, dtype=np.int32)
    mask[2:-2, 2:-2, 2:-2] = 1
    return Subject(
        subject_id=subject_id,
        images=images,
        masks={"tumor": ArrayImageRef(array=mask, geometry=geometry)},
        metadata={"center": "A" if seed % 2 == 0 else "B"},
    )


def main() -> None:
    """Print geometry / materialisation checks and a tiny voxel feature call."""
    geometry = Geometry.from_array(SHAPE)
    eager = ImageVolume.from_geometry(
        np.ones(SHAPE, dtype=np.float32),
        geometry,
        modality="T1",
    )
    eager_mask = MaskVolume.from_geometry(
        np.ones(SHAPE, dtype=np.int32),
        geometry,
        roi_name="tumor",
    )
    print(
        f"Eager volumes: image.shape={eager.data.shape}, "
        f"mask.roi={eager_mask.roi_name}, labels={eager_mask.labels}"
    )

    cohort = Cohort(
        [make_subject_from_arrays(f"P{i:03d}", seed=i) for i in range(3)],
        name="from_numpy",
    )
    print(f"Cohort: n={len(cohort)} ids={list(cohort.subject_ids)}")
    print(f"Fingerprint: {cohort.summarize()}")

    subject = cohort[0]
    t1 = subject.image("T1")
    roi = subject.mask("tumor")
    print(
        f"Materialised {subject.subject_id}: "
        f"T1 shape={t1.data.shape}, ROI foreground="
        f"{int((roi.data > 0).sum())}"
    )

    # Prove the subject is usable by a domain operator (no directory, no YAML).
    field = RawVoxelFeatures(modalities=list(MODALITIES))(subject)
    print(
        f"RawVoxelFeatures: voxels={field.values.shape[0]}, "
        f"names={list(field.feature_names)}"
    )
    print(
        "Replace ArrayImageRef arrays with your nibabel/SimpleITK buffers; "
        "downstream habitat recipes stay unchanged."
    )
    return cohort, t1


cohort, t1 = main()
# END example

# BEGIN figures
# Paste after the Script block. Uses cohort, t1, and MODALITIES.
from pathlib import Path

from habit import one_step_habitat
from habit.viz import plot_habitat_overlay

result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi="tumor"
).fit_predict(cohort)
fig = plot_habitat_overlay(
    t1.data,
    result.habitat_maps[0].label_array,
    axis=0,
    title="Habitats from NumPy Subject",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/data_from_arrays_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/data_from_arrays_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure

    save_example_figure(fig, "data_from_arrays_overlay.png")
