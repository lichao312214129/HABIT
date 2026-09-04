#!/usr/bin/env python
"""
Match habitat ids: same-tumour overlap vs habitat summary features
(multimodal intensities or radiomics).

Three synthetic patients, two habitats, two unscaled summary features
(Energy, Coarseness). Shows why a cohort z-score must be fit on
**all** habitat rows before Hungarian assignment.

Accompanies ``docs/source/examples/habitat_label_match.rst``.

Run from the repository root::

    python docs/source/examples/scripts/habitat_label_match_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from habit.kernels.habitat_label_match import (
    fit_feature_match_scale,
    match_labels_by_features,
    match_labels_by_overlap,
    remap_label_array,
)

# BEGIN example
# Unscaled habitat means. Rows are habitats, columns are Energy, Coarseness.
# Patient A is the atlas. B and C have swapped raw ids and a scanner shift.
PATIENTS: Dict[str, np.ndarray] = {
    "A": np.array([[8.0e6, 0.12], [2.0e7, 0.03]], dtype=np.float64),
    "B": np.array([[2.2e7, 0.04], [9.0e6, 0.11]], dtype=np.float64),
    "C": np.array([[3.5e7, 0.03], [2.3e7, 0.12]], dtype=np.float64),
}
IDS = np.array([1, 2], dtype=np.int64)

print("Unscaled habitat means (Energy, Coarseness)")
for name, rows in PATIENTS.items():
    print(f"  {name} id1={rows[0].tolist()}  id2={rows[1].tolist()}")

location, scale = fit_feature_match_scale(list(PATIENTS.values()))
print(
    "Cohort z-score fit on all 6 rows:\n"
    f"  mean={location.tolist()}\n"
    f"  std ={scale.tolist()}"
)

mappings: Dict[str, Dict[int, int]] = {"A": {1: 1, 2: 2}}
for name in ("B", "C"):
    mappings[name] = match_labels_by_features(
        IDS,
        PATIENTS["A"],
        IDS,
        PATIENTS[name],
        metric="euclidean",
        standardize="zscore",
        location=location,
        scale=scale,
    )
    print(f"  {name} -> A  {mappings[name]}")

# Same-tumour observer pair: two 2x2 blocks, ids swapped, no texture.
reference = np.zeros((4, 4), dtype=np.int32)
reference[0:2, 0:2] = 1
reference[2:4, 0:2] = 2
moving = np.zeros((4, 4), dtype=np.int32)
moving[0:2, 0:2] = 2
moving[2:4, 0:2] = 1
observer_map = match_labels_by_overlap(reference, moving)
aligned = remap_label_array(moving, observer_map, reserved_ids=(1, 2))
print(f"Observer overlap mapping (same tumour): {observer_map}")
assert np.array_equal(aligned, reference)

# Compose: observer ids already sit in physician-2 space, then inherit A.
composed = {
    int(mov): int(mappings["B"][int(phys2)])
    for mov, phys2 in observer_map.items()
    if int(phys2) in mappings["B"]
}
print(f"Observer ids after inheriting B->A: {composed}")
# END example

# BEGIN figures
# Paste after the Script block. Uses PATIENTS, mappings, location, scale.
from pathlib import Path

from habit.viz import use_style

with use_style("radiology"):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    colors = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
    markers = {"A": "o", "B": "s", "C": "D"}

    ax = axes[0]
    for name, rows in PATIENTS.items():
        ax.scatter(
            rows[:, 0] / 1e6,
            rows[:, 1],
            c=colors[name],
            marker=markers[name],
            s=70,
            label=name,
        )
        for hid, row in enumerate(rows, start=1):
            ax.annotate(f"{name}{hid}", (row[0] / 1e6, row[1]), fontsize=8)
    ax.set_xlabel("Energy (millions)")
    ax.set_ylabel("Coarseness")
    ax.set_title("Unscaled means (Energy dominates)")

    ax = axes[1]
    for name, rows in PATIENTS.items():
        z = (rows - location) / scale
        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=colors[name],
            marker=markers[name],
            s=70,
            label=name,
        )
        for hid, row in enumerate(z, start=1):
            ax.annotate(f"{name}{hid}", (row[0], row[1]), fontsize=8)
    atlas_z = (PATIENTS["A"] - location) / scale
    for name in ("B", "C"):
        mov_z = (PATIENTS[name] - location) / scale
        for mov_id, ref_id in mappings[name].items():
            src = mov_z[int(mov_id) - 1]
            dst = atlas_z[int(ref_id) - 1]
            ax.annotate(
                "",
                xy=(dst[0], dst[1]),
                xytext=(src[0], src[1]),
                arrowprops={"arrowstyle": "->", "color": colors[name], "lw": 1.2},
            )
    ax.set_xlabel("Energy (cohort z)")
    ax.set_ylabel("Coarseness (cohort z)")
    ax.set_title("Hungarian after one cohort z-score")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    Path("out").mkdir(exist_ok=True)
    fig.savefig("out/habitat_label_match.png", dpi=150, bbox_inches="tight")
print("Wrote out/habitat_label_match.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(["habitat_label_match.png"])
