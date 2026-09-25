"""
Intratumoral heterogeneity (ITH)
================================

**Background.** Intratumoral heterogeneity (ITH) describes how mixed a
tumour is. Here it is measured from the habitat map as fragmentation: do
habitats form a few compact blobs, or many scattered pieces?

**Purpose.** You get one ITH score for the whole tumour plus one dispersion
value per habitat, as a table and a summary figure.

**Key terms.**

* **ITH score** -- intratumoral heterogeneity as fragmentation: 0 when each
  habitat is one connected blob, approaching 1 when habitats break into many
  small pieces. Formula: ``1 - sum_i(S_i,max / n_i) / S_total``, with
  ``S_i,max`` the largest piece of habitat ``i``, ``n_i`` its number of
  face-connected pieces and ``S_total`` all non-background voxels.
* **per-habitat dispersion** -- the same idea for one habitat,
  ``1 - (S_i,max / n_i) / S_i``; the ITH score is their volume-weighted mean.

Atomic ITH metrics from a habitat label map:
:func:`~habit.kernels.ith_score` and
:func:`~habit.kernels.habitat_ith_dispersion`.
"""

# %%
# One-step habitats
# -----------------
# Same idea as the complete analysis, without partition or pool, so one
# subject is enough. ``one_step_habitat`` builds these stages:
# extract ``Spec("raw")``, fit ``Spec("kmeans", {"n_habitats": 3})``,
# assign ``Spec("nearest_centroid")``.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import habitat_ith_dispersion, ith_score
from habit.recipes import one_step_habitat
from habit.viz import plot_ith_summary

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort)
labels = result.habitat_maps[0].label_array

# %%
# ITH score and per-habitat dispersion
# ------------------------------------
# One tumour-level number, then one value per habitat id (same formula).
ith = float(ith_score(labels))
dispersion = habitat_ith_dispersion(labels)
ith_table = pd.DataFrame(
    [{"habitat": "ITH", "score": ith}]
    + [{"habitat": f"H{hid}", "score": float(val)} for hid, val in sorted(dispersion.items())]
)
print("ITH scores:")
print(ith_table.to_string(index=False))
ith_table

Path("out").mkdir(exist_ok=True)
fig_ith = plot_ith_summary(ith, dispersion=dispersion)
fig_ith.savefig("out/ith_summary.png", dpi=150, bbox_inches="tight")
plt.show()
