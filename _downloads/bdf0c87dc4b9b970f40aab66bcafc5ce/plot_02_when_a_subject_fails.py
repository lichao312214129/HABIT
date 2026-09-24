"""
When a subject fails
====================

One subject in this cohort has no portal-venous (``PVP``) series, so
voxel extraction raises for it. What happens next is set on the backend:

* ``on_subject_failure="continue"`` (the default) fits habitats on the
  subjects that worked and records the failure in the run manifest;
* ``on_subject_failure="fail_fast"`` stops the run at the first failure.

A :class:`~habit.execution.CheckpointStore` keeps every finished subject,
so a rerun after fixing the data only computes what is missing.
"""

# %%
# A cohort with one incomplete subject
# ------------------------------------
# The fourth subject is a copy of ``subj004`` without its ``PVP`` image.
# sphinx_gallery_thumbnail_number = 1
import dataclasses
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import CheckpointStore, SerialBackend
from habit.recipes import two_step_habitat
from habit.spec import RunPolicy
from habit.viz import plot_habitat_overlay

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
subjects = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:4]
no_pvp = dataclasses.replace(
    subjects[3],
    subject_id="subj004_no_PVP",
    images={name: ref for name, ref in subjects[3].images.items() if name != "PVP"},
)
cohort = Cohort(list(subjects[:3]) + [no_pvp], name="one_incomplete")
study = two_step_habitat(modalities=MODALITIES, roi=ROI, n_supervoxels=30, random_seed=0)

# %%
# Continue past the failure
# -------------------------
# Three subjects get habitat maps; the manifest says which one failed.
result = study.fit_predict(cohort, backend=SerialBackend(on_subject_failure="continue"))
print("maps:", [m.subject_id for m in result.habitat_maps])
print("outcomes:", result.manifest.subject_outcomes)

Path("out").mkdir(exist_ok=True)
fig = plot_habitat_overlay(
    cohort[0].image(ROI),
    result.habitat_maps[0],
    title=f"{cohort[0].subject_id}: habitats (run continued)",
    crop_to="labels",
)
fig.savefig("out/when_a_subject_fails_habitats.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Stop at the first failure
# -------------------------
# Use this when every subject must be present, for example before a
# final analysis.
try:
    study.fit_predict(cohort, backend=SerialBackend(on_subject_failure="fail_fast"))
except KeyError as error:
    print("stopped:", error)

# %%
# Resume from a checkpoint
# ------------------------
# The first call computes and stores every subject; the second reads them
# back. Here the second call runs on the same data; after fixing a
# failed subject, only that one would be computed again.
store = CheckpointStore(Path(tempfile.mkdtemp(prefix="habit_ckpt_")))
complete = subjects[:3]
for attempt in ("first run", "rerun"):
    start = time.perf_counter()
    study.fit_predict(complete, checkpoint=store)
    print(f"{attempt}: {time.perf_counter() - start:.1f} s")

# %%
# A time limit per subject
# ------------------------
# ``subject_timeout_sec`` stops one subject that runs too long and records
# it as failed. Only the process backend can enforce it, so it goes on a
# ``RunPolicy`` with ``backend="process"``; pass that backend to
# ``fit_predict`` as on :doc:`plot_01_run_a_cohort`.
policy = RunPolicy(workers=4, backend="process", on_subject_failure="continue", subject_timeout_sec=600)
print(policy.subject_timeout_sec, policy.on_subject_failure)
