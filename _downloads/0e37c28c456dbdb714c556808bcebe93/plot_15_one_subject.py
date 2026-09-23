"""
Clustering habitats inside one subject
======================================

Input: one :class:`~habit.contracts.Subject` wrapped as a
:class:`~habit.contracts.Cohort`. Output: that subject's
:class:`~habit.contracts.HabitatMap`. This is the one-step design with
no ``partition`` and no ``pool``.

A Python list is not a cohort. ``fit_predict`` reads ``cohort.name``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import one_step_habitat
from habit.viz import plot_partition_triptych

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
# A Python list is not a Cohort. fit_predict reads cohort.name, so wrap
# a single subject explicitly. The name is used in logs and provenance.
one = Cohort([subject], name="one")

result = one_step_habitat(
    modalities=MODALITIES,
    n_habitats=3,
    random_seed=0,
    roi=ROI,
).fit_predict(one)
habitat_map = result.habitat_maps[0]
print(habitat_map.subject_id, result.subject_models[subject.subject_id].summary())

fig = plot_partition_triptych(
    subject.image(ROI),
    result.units[0],
    habitat_map,
    axis=0,
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/one_subject_triptych.png", dpi=150, bbox_inches="tight")
plt.show()
