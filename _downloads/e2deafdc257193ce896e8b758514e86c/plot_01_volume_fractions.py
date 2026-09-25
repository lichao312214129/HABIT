"""
Volume and fractions
====================

**Background.** The simplest habitat number is how much of the tumour each
habitat occupies. Volume fractions let you ask, for example, whether a
larger share of one habitat goes with a worse outcome.

**Purpose.** You get a table with one row per habitat (volume fraction,
number of separate pieces, size of the largest piece), an overlay of the
habitat map, and a bar chart of the fractions.

**Key terms.**

* **habitat** -- a sub-region inside the tumour (the ROI) whose voxels behave
  alike across the input images; HABIT paints each ROI voxel with a habitat
  id (1, 2, 3, ...).
* **volume fraction** -- voxels of one habitat divided by all non-background
  voxels of the map; fractions sum to 1.
* **region** -- one face-connected piece of a habitat; ``num_regions`` counts
  the pieces and ``largest_region_voxels`` is the size of the biggest one.
* **one-step habitats** -- clustered inside this one subject; see
  :doc:`/auto_examples/04_designs/plot_02_inside_each_subject`.

Atomic volume metrics from a habitat label map:
:func:`~habit.kernels.habitat_volume_fractions` and
:func:`~habit.kernels.habitat_region_stats`.
"""

# %%
# One-step habitats give a map to quantify.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

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
# Collect the non-background ids actually present in this map.
ids = tuple(sorted({int(v) for v in labels.ravel() if int(v) != 0}))
# Share of the ROI per habitat: {habitat id: fraction in [0, 1]}.
frac = habitat_volume_fractions(labels, ids)
# Fragmentation per habitat: {habitat id: (num_regions, largest_region_voxels)}.
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
plt.show()
