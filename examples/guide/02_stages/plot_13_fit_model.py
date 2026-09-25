"""
Fitting a cohort habitat model
==============================

**Background.** After every subject is cut into supervoxels, ``fit``
learns one habitat definition from all of them together. Because the
same model labels every patient, habitat 2 means the same feature
profile in each subject.

**Purpose.** You get a fitted :class:`~habit.contracts.HabitatModel`
with three habitats, its printed summary, and a bar chart of the first
feature of each centroid (what each habitat looks like on average).

**Key terms.**

* **supervoxel** -- see :doc:`/auto_examples/02_stages/plot_12_supervoxels`.
* **pool** -- stacks every training subject's rows into one matrix so one
  model is fitted to the whole cohort and habitat ids mean the same thing
  in every patient.
* **fit** -- learns the habitat definition (for k-means: the number of
  habitats and their centroids).
* **centroid** -- the mean feature vector of one habitat; new rows are
  labelled by the centroid they are closest to.
* **elbow** -- a rule for picking the number of habitats: the candidate
  count after which adding one more habitat stops reducing
  within-cluster spread much. This page fixes ``n_habitats=3``, so no
  selection runs; the complete analysis passes ``min_habitats`` /
  ``max_habitats`` with ``validation="elbow"`` instead.

The ``fit`` stage of
:doc:`/auto_examples/00_full_pipeline/plot_01_full_pipeline`.
Input: one :class:`~habit.contracts.Supervoxelization` per subject.
Output: a :class:`~habit.contracts.HabitatModel`.

The calls below are that stage without a ``HabitatSpec``. Inside a spec
they are ``Stage("partition", Spec("slic", {"n_supervoxels": 100}))``
then ``Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 3}))``.
"""

# %%
# Load the cohort
# ---------------
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.supervoxel import SlicSupervoxelizer
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {list(cohort.subject_ids)}")

# %%
# Partition each subject
# ----------------------
# Each element of ``units`` is one subject's supervoxels. These rows
# are what the cohort model fits on.
voxel = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
slic = SlicSupervoxelizer(n_supervoxels=100, compactness=10.0)
units = [slic(voxel(subject)) for subject in cohort]
for one in units:
    print(f"{one.subject_id}: {len(one.features)} supervoxels")

# %%
# Fit one shared model
# --------------------
# The fitter pools every subject's units and learns one set of centroids.
# n_habitats=3 fixes the count; n_init=3 restarts k-means three times and
# keeps the run with the lowest within-cluster spread (inertia).
fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
# Fixing the seed makes a rerun give the same centroids (and ids).
fitter.set_random_state(0)
model = fitter.fit(units, cohort=cohort)
print(model.summary())

# %%
# First feature of each centroid
# ------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.2))
names = [f"habitat {index + 1}" for index in range(int(model.n_habitats))]
ax.bar(names, model.centroids[:, 0], color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_ylabel(str(model.feature_names[0]))
ax.set_title("habitat centroids")
Path("out").mkdir(exist_ok=True)
fig.savefig("out/cohort_centroids.png", dpi=150, bbox_inches="tight")
plt.show()
