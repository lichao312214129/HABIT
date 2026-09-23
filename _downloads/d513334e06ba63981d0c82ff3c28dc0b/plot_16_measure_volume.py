"""
Measuring habitats on a label map
=================================

Input: a :class:`~habit.contracts.Subject` and a
:class:`~habit.contracts.HabitatMap`. Output: a
:class:`~habit.contracts.FeatureTable` of voxel counts and volume
fractions. Stage: ``quantify`` with ``volume``.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_features import HabitatVolumeFeatures
from habit.recipes import one_step_habitat

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
result = one_step_habitat(
    modalities=MODALITIES,
    n_habitats=3,
    random_seed=0,
    roi=ROI,
).fit_predict(Cohort([subject], name="one"))

table = HabitatVolumeFeatures()(subject, result.habitat_maps[0])
print(table.frame)
table.frame

fraction_columns = [
    str(name) for name in table.frame.columns if str(name).endswith("volume_fraction")
]
fractions = table.frame.iloc[0][fraction_columns].to_numpy(dtype=float)
labels = [name.replace("_volume_fraction", "").replace("_", " ") for name in fraction_columns]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.bar(labels, fractions, color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_ylabel("volume fraction")
ax.set_ylim(0, 1)
ax.set_title("habitat volume")
Path("out").mkdir(exist_ok=True)
fig.savefig("out/habitat_volume.png", dpi=150, bbox_inches="tight")
plt.show()
