"""
Measuring habitats on a label map
=================================

Input: a :class:`~habit.contracts.Subject` and a
:class:`~habit.contracts.HabitatMap`. Output: a
:class:`~habit.contracts.FeatureTable` of voxel counts and volume
fractions. Stage: ``quantify`` with ``volume``.

MSI, ITH, graphs, and radiomics are in
:doc:`/auto_examples/05_quantify/index`.
Repeatability before reporting is in
:doc:`/auto_examples/03_precision/plot_01_precise_features`.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
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
