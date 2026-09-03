"""
Volume and fractions
====================

Atomic volume metrics from a habitat label map:
:func:`~habit.kernels.habitat_volume_fractions` and
:func:`~habit.kernels.habitat_region_stats`.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# One-step habitats give a map to quantify.
from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import habitat_region_stats, habitat_volume_fractions
from habit.recipes import one_step_habitat
from habit.viz import plot_habitat_overlay, plot_habitat_volume_fractions

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort)
habitat_map = result.habitat_maps[0]
labels = habitat_map.label_array

# %%
# Volume fractions and connected-component stats per habitat id.
ids = tuple(sorted({int(v) for v in labels.ravel() if int(v) != 0}))
frac = habitat_volume_fractions(labels, ids)
stats = habitat_region_stats(labels)
table = pd.DataFrame(
    [
        {
            "habitat": f"H{hid}",
            "volume_fraction": float(frac[hid]),
            "num_regions": int(stats[hid][0]),
            "largest_region_voxels": int(stats[hid][1]),
        }
        for hid in ids
    ]
)
print(table.to_string(index=False))
table

Path("out").mkdir(exist_ok=True)
fig = plot_habitat_overlay(
    cohort[0].image(ROI),
    habitat_map,
    title="Habitats (K=3)",
)
fig.savefig("out/volume_fractions_overlay.png", dpi=150, bbox_inches="tight")
fig_frac = plot_habitat_volume_fractions(frac)
fig_frac.savefig("out/volume_fractions_bar.png", dpi=150, bbox_inches="tight")
if os.environ.get("HABIT_NO_VIEW") != "1":
    plt.show()
