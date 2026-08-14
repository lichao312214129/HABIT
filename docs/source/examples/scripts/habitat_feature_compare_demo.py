#!/usr/bin/env python
"""
Habitat-feature contrast figures (heatmap, Cliff's delta, violin, bars).

Accompanies ``docs/source/reference/features/whole_each_habitat.rst`` and
``docs/source/examples/visualization.rst``.
Run from the repository root::

    python docs/source/examples/scripts/habitat_feature_compare_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import numpy as np
import pandas as pd

from habit import FeatureTable, compare_habitat_features
from habit.viz import (
    plot_habitat_feature_bars,
    plot_habitat_feature_effect,
    plot_habitat_feature_heatmap,
    plot_habitat_feature_violin,
)

# Swap `table` for your each_habitat / extract_habitat_features FeatureTable
# (wide columns habitat_{id}_{feature}, one row per subject).
rng = np.random.default_rng(4)
n_subjects = 12
feature_specs = (
    ("original_firstorder_Mean_of_T2", 120.0, 8.0),
    ("original_firstorder_Median_of_T2", 110.0, 7.0),
    ("original_firstorder_Energy_of_T2", 1.8e9, 2.0e8),
    ("original_firstorder_Kurtosis_of_T2", 3.2, 0.4),
    ("volume_fraction", 0.35, 0.08),
)
rows = []
for index in range(n_subjects):
    row = {"subject": f"subj{index:03d}"}
    for hid, shift in ((1, 0.0), (2, 0.35)):
        row[f"has_habitat_{hid}"] = 1.0
        for name, loc, scale in feature_specs:
            value = float(rng.normal(loc * (1.0 + 0.15 * shift), scale))
            if name == "volume_fraction":
                value = float(np.clip(value, 0.05, 0.95))
            row[f"habitat_{hid}_{name}"] = value
    rows.append(row)
table = FeatureTable(
    frame=pd.DataFrame(rows),
    id_columns=("subject",),
    feature_columns=tuple(
        name for name in rows[0] if name != "subject"
    ),
)
cmp = compare_habitat_features(table)
print(cmp.n_subjects, "subjects;", len(cmp.panel.feature_names), "features")
# END example

# BEGIN figures
# Paste after the Script block. Uses cmp.
Path("out").mkdir(exist_ok=True)
FEATURES = (
    "original_firstorder_Mean_of_T2",
    "original_firstorder_Median_of_T2",
    "original_firstorder_Energy_of_T2",
    "original_firstorder_Kurtosis_of_T2",
    "volume_fraction",
)
fig = plot_habitat_feature_heatmap(
    cmp, features=FEATURES, title="Cohort mean habitat x feature"
)
fig.savefig("out/habitat_feature_heatmap_cohort.png", dpi=150, bbox_inches="tight")
fig = plot_habitat_feature_heatmap(
    cmp,
    subject_id="subj000",
    features=FEATURES,
    title="Habitat feature profile (subj000)",
)
fig.savefig("out/habitat_feature_heatmap_subject.png", dpi=150, bbox_inches="tight")
fig = plot_habitat_feature_effect(cmp, pair=(2, 1), top_k=5)
fig.savefig("out/habitat_feature_effect.png", dpi=150, bbox_inches="tight")
fig = plot_habitat_feature_violin(cmp, features=FEATURES)
fig.savefig("out/habitat_feature_violin.png", dpi=150, bbox_inches="tight")
fig = plot_habitat_feature_bars(cmp, features=FEATURES)
fig.savefig("out/habitat_feature_bars.png", dpi=150, bbox_inches="tight")
print("Wrote habitat-feature contrast figures under out/")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(
        (
            "habitat_feature_heatmap_cohort.png",
            "habitat_feature_heatmap_subject.png",
            "habitat_feature_effect.png",
            "habitat_feature_violin.png",
            "habitat_feature_bars.png",
        )
    )
