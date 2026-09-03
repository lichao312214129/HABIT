#!/usr/bin/env python
"""
Habitat features → Cox survival model → publication figures.

Accompanies the survival section of ``docs/source/examples/visualization.rst``.
Uses a synthetic imaging cohort (swap for ``cohort_from_directory``) and
**synthetic** time/event labels driven by one habitat feature — not a
clinical claim. The model path is HABIT ``TablePipeline`` + ``CoxPH``.

Run from the repository root::

    python docs/source/examples/scripts/habitat_survival_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import FeatureTable
from habit.pipeline import TablePipeline
from habit.datasets import make_synthetic_cohort
from habit.recipes import one_step_habitat
from habit.contracts import SurvivalOutcome
from habit.survival import SurvivalModelRegistry
from habit.table_preprocessing import TablePreprocessorRegistry
from habit.viz import (
    plot_cox_forest,
    plot_kaplan_meier,
    plot_survival_calibration,
    use_style,
)

# Synthetic cohort stand-in. Swap for:
#   DATA = "demo_data/preprocessed"
#   cohort = cohort_from_directory(DATA, modalities=("LAP",), roi="LAP")
# and keep the same habitat_features / survival attachment below.
cohort = make_synthetic_cohort(n_subjects=28, shape=(20, 20, 20), rng=4)
result = one_step_habitat(
    modalities=("T1", "T2"),
    n_habitats=3,
    habitat_features=("volume", "ith_score"),
    random_seed=4,
    roi="tumor",
).fit_predict(cohort)
base = result.features
frame = base.frame.copy()
print("Habitat feature columns:", list(base.feature_columns))

# Synthetic survival labels correlated with ITH (swap for your time/event CSV).
signal = frame["ith_score"].to_numpy(dtype=float)
signal = (signal - np.nanmean(signal)) / (np.nanstd(signal) + 1e-8)
surv_rng = np.random.default_rng(4)
true_time = np.clip(
    -np.log(surv_rng.uniform(size=len(signal))) / np.exp(0.65 * signal) * 24.0,
    0.5,
    None,
)
censor_time = surv_rng.uniform(12.0, 40.0, size=len(signal))
event = true_time <= censor_time
time = np.where(event, true_time, censor_time)
frame["os_time"] = time
frame["os_event"] = event.astype(int)

keep = [
    name
    for name in base.feature_columns
    if name == "ith_score" or "volume_fraction" in name
]
table = FeatureTable(
    frame=frame,
    id_columns=base.id_columns,
    feature_columns=tuple(keep),
    outcome=SurvivalOutcome(time_column="os_time", event_column="os_event"),
)

pipe = TablePipeline(
    steps=[TablePreprocessorRegistry.create("zscore")],
    model=SurvivalModelRegistry.create("CoxPH", penalizer=0.2),
)
pipe.fit(table)
risk = pipe.predict(table).to_numpy()
group = (risk >= np.median(risk)).astype(int)
print(f"Fitted CoxPH on {len(keep)} habitat features; n={len(frame)}")
# END example

# BEGIN figures
# Paste after the Script block. Uses pipe, table, time, event, group, keep.
Path("out").mkdir(exist_ok=True)
horizon = float(np.median(time))
predicted = pipe.predict_survival_function(
    table, np.asarray([horizon], dtype=float)
).to_numpy()[:, 0]
# Hazard ratios from the fitted CoxPH (lifelines summary on the same model).
summary = pipe.model._model.summary

with use_style("radiology"):
    fig_km = plot_kaplan_meier(
        time,
        event,
        group=group,
        group_names=("Low risk", "High risk"),
        title="Kaplan-Meier by habitat-feature Cox risk",
    )
    fig_km.savefig("out/habitat_survival_km.png", dpi=150, bbox_inches="tight")
    fig_forest = plot_cox_forest(
        names=list(summary.index.astype(str)),
        hazard_ratio=summary["exp(coef)"].to_numpy(),
        lower=summary["exp(coef) lower 95%"].to_numpy(),
        upper=summary["exp(coef) upper 95%"].to_numpy(),
        p_values=summary["p"].to_numpy(),
        title="Cox hazard ratios (habitat features)",
    )
    fig_forest.savefig(
        "out/habitat_survival_forest.png", dpi=150, bbox_inches="tight"
    )
    fig_cal = plot_survival_calibration(
        time, event, predicted, horizon=horizon, n_groups=3
    )
    fig_cal.savefig(
        "out/habitat_survival_calibration.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_km)
    plt.close(fig_forest)
    plt.close(fig_cal)
print(
    "Wrote out/habitat_survival_km.png, "
    "out/habitat_survival_forest.png, "
    "out/habitat_survival_calibration.png"
)
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(
        (
            "habitat_survival_km.png",
            "habitat_survival_forest.png",
            "habitat_survival_calibration.png",
        )
    )
