"""
Reusing workers across passes
=============================

A cold process pool pays process spawn and imports on every ``map``.
``with backend.reuse_workers():`` starts the workers once and keeps them
for every ``map`` inside the block. Leaving the block stops them.

This page extracts texture serially (cached), fits in this process, then
times habitat assignment three ways: a cold pool, the first map inside
the block, and a second map in the same block. The second map is the one
that skips spawn. Texture belongs in that same block when the cohort is
large; on one GPU a two-worker texture pool is slower, see
:doc:`/auto_examples/06_parallel_runs/plot_05_process_pool`.
"""

# %%
# Set up the cohort and one pool
# ------------------------------
# Building the backend does not start workers. The device is ``"auto"``.
# sphinx_gallery_thumbnail_number = 1
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, TypeVar

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend, backend_from_policy
from habit.feature_preprocessing import SubjectPreprocessingChain, ZScoreScaling
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.spec import RunPolicy
from habit.viz import plot_habitat_overlay
from habit.voxel_features import VoxelRadiomicsFeatures


class _NamedResult(Protocol):
    """Anything the pool returns that carries a subject id."""

    subject_id: str


_T = TypeVar("_T", bound=_NamedResult)

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(cohort)
Path("out").mkdir(exist_ok=True)
CACHE = str((Path("out") / "voxel_texture_cache").resolve())

texture = VoxelRadiomicsFeatures(
    modalities=list(MODALITIES),
    roi=ROI,
    kernel_radius=3,
    params={
        "imageType": {"Original": {}},
        "featureClass": {
            "firstorder": ["Mean", "Entropy"],
            "glcm": ["Contrast", "Correlation", "Idm", "JointEntropy"],
            "glrlm": ["ShortRunEmphasis", "LongRunEmphasis"],
        },
        "setting": {"binWidth": 12},
    },
    voxel_batch=1000,
    cache_dir=CACHE,
)
policy = RunPolicy(
    workers=2,
    backend="process",
    parallel_mode="persistent",
    auto_retry_rounds=0,
)


def by_subject(results: Iterable[_T]) -> dict[str, _T]:
    """Index results by subject id.

    A process pool yields subjects in the order they finish, not the
    order they were submitted.
    """
    return {result.subject_id: result for result in results}


# %%
# Time a cold pool against a reused pool
# --------------------------------------
# Texture, z-score and the fit stay in this process. A cold ``map`` pays
# spawn plus assignment. Inside ``reuse_workers`` the first ``map`` starts
# the workers and the second ``map`` reuses them. On Windows the pool is
# started under ``__main__``. Copy this cell together with the one above.
if __name__ == "__main__":
    fields = [slot.result() for slot in SerialBackend().map(texture, cohort)]
    zscore = SubjectPreprocessingChain([ZScoreScaling(across_features=False)])
    units = [
        voxel_units(
            field.with_feature_frame(
                zscore(field.feature_frame()),
                produced_by="subject_feature_preprocessor",
                spec_fingerprint=zscore.spec.fingerprint(),
            )
        )
        for field in fields
    ]
    fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
    fitter.set_random_state(0)
    model = fitter.fit(units, cohort=cohort)
    assigner = model.assigner()

    timings: dict[str, float] = {}
    started = time.perf_counter()
    cold = by_subject(slot.result() for slot in backend_from_policy(policy).map(assigner, units))
    timings["cold pool"] = time.perf_counter() - started

    warm_backend = backend_from_policy(policy)
    with warm_backend.reuse_workers():
        started = time.perf_counter()
        first = by_subject(slot.result() for slot in warm_backend.map(assigner, units))
        timings["session, first map"] = time.perf_counter() - started
        started = time.perf_counter()
        second = by_subject(slot.result() for slot in warm_backend.map(assigner, units))
        timings["session, second map"] = time.perf_counter() - started

    for name, seconds in timings.items():
        print(f"{name:>22}: {seconds:5.2f} s")
    for subject in cohort:
        same = cold[subject.subject_id].label_array
        assert (same == first[subject.subject_id].label_array).all()
        assert (same == second[subject.subject_id].label_array).all()
        print(subject.subject_id, "cold == both session maps")

    fig, ax = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
    ax.barh(list(timings), list(timings.values()), color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock seconds (assignment, 2 subjects)")
    ax.set_title("worker startup is paid once per session")
    fig.savefig("out/reuse_workers.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig_map = plot_habitat_overlay(
        cohort[0].image(ROI),
        second[cohort[0].subject_id],
        title="habitats (reused workers)",
        crop_to="labels",
    )
    fig_map.savefig("out/reuse_workers_habitats.png", dpi=150, bbox_inches="tight")
    plt.show()
