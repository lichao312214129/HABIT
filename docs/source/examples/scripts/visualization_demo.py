#!/usr/bin/env python
"""
Publication figures with ``habit.viz`` (pure matplotlib, English labels).

Demonstrates:

* habitat-clustering PCA scatter from a two-step ``StudyResult``,
* Kaplan-Meier curves stratified by a risk split (not by the event itself),
* regression diagnostics (predicted vs observed, residuals, Q-Q, Bland-Altman),
* Cox forest, risk triptych, time-dependent AUC, Brier, survival calibration,

Figures are returned to the caller; persistence is explicit (``savefig``).

This script accompanies ``docs/source/examples/visualization.rst``.

Run from the repository root::

    python docs/source/examples/scripts/visualization_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from habit.spec import HabitatSpec, Spec, Stage
from habit.datasets import make_synthetic_cohort
import habit.recipes as recipes
from habit.viz import (
    plot_bland_altman,
    plot_brier_curve,
    plot_cox_forest,
    plot_habitat_clustering_pca_2d,
    plot_kaplan_meier,
    plot_predicted_vs_observed,
    plot_residual_qq,
    plot_residuals,
    plot_risk_triptych,
    plot_survival_calibration,
    plot_time_dependent_auc,
    use_style,
)

Path("out").mkdir(exist_ok=True)


def _save(fig: object, name: str) -> None:
    """Save a figure under out/ and close it."""
    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote out/{name}")


# Habitat clustering PCA (synthetic cohort; swap for cohort_from_directory)
cohort = make_synthetic_cohort(n_subjects=6, shape=(20, 20, 20), rng=3)
spec = HabitatSpec(
    name="viz_two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 3})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
                    "validation": "elbow",
                    "n_init": 3,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
    ),
    random_seed=3,
)
result = recipes.Study(spec=spec).fit_predict(cohort)
arrays = result._population_clustering_arrays()
if arrays is None:
    raise RuntimeError("Expected pooled clustering arrays from two-step result.")
features, labels, centers = arrays

with use_style("radiology"):
    fig_pca = plot_habitat_clustering_pca_2d(
        features,
        labels,
        centers=centers,
        title="Habitat clustering (PCA)",
    )
_save(fig_pca, "habitat_pca_2d.png")

# Survival table with a moderate effect and independent censoring so the
# KM strata overlap (not a perfect split). Swap for your own time/event CSV.
surv_rng = np.random.default_rng(9)
n_surv = 120
risk = surv_rng.normal(loc=0.0, scale=1.0, size=n_surv)
noise0 = surv_rng.normal(size=n_surv)
noise1 = surv_rng.normal(size=n_surv)
true_time = np.clip(
    -np.log(surv_rng.uniform(size=n_surv)) / np.exp(0.55 * risk) * 20.0,
    0.5,
    None,
)
censor_time = surv_rng.uniform(10.0, 40.0, size=n_surv)
event = true_time <= censor_time
time = np.where(event, true_time, censor_time)
group = (risk >= np.median(risk)).astype(int)

with use_style("nature"):
    fig_km = plot_kaplan_meier(
        time,
        event,
        group=group,
        group_names=("Low risk", "High risk"),
        title="Kaplan-Meier by risk split",
    )
_save(fig_km, "kaplan_meier.png")

cox_df = pd.DataFrame(
    {"time": time, "event": event.astype(int), "risk": risk, "noise0": noise0, "noise1": noise1}
)
cph = CoxPHFitter()
cph.fit(cox_df, duration_col="time", event_col="event")
summary = cph.summary
horizon = float(np.median(time))
predicted_surv = (
    cph.predict_survival_function(cox_df, times=[horizon]).T.iloc[:, 0].to_numpy()
)
times_grid = np.linspace(float(np.percentile(time, 15)), float(np.percentile(time, 70)), 12)
surv_mat = cph.predict_survival_function(cox_df, times=list(times_grid)).T.to_numpy()

with use_style("radiology"):
    _save(
        plot_cox_forest(
            names=list(summary.index.astype(str)),
            hazard_ratio=summary["exp(coef)"].to_numpy(),
            lower=summary["exp(coef) lower 95%"].to_numpy(),
            upper=summary["exp(coef) upper 95%"].to_numpy(),
            p_values=summary["p"].to_numpy(),
            title="Cox hazard ratios",
        ),
        "cox_forest.png",
    )
    _save(plot_risk_triptych(time, event, risk), "risk_triptych.png")
    _save(
        plot_survival_calibration(time, event, predicted_surv, horizon=horizon),
        "survival_calibration.png",
    )
    _save(plot_time_dependent_auc(time, event, risk), "time_dependent_auc.png")
    _save(plot_brier_curve(time, event, surv_mat, times_grid), "brier_curve.png")

# Regression diagnostics with a noisy linear relationship (not an identity).
rng = np.random.default_rng(7)
observed = rng.normal(loc=0.0, scale=1.0, size=80)
predicted = 0.65 * observed + rng.normal(loc=0.0, scale=0.55, size=80)
with use_style("radiology"):
    fig_reg = plot_predicted_vs_observed(observed, predicted)
    fig_reg.suptitle("Predicted vs observed")
    _save(fig_reg, "predicted_vs_observed.png")
    _save(plot_residuals(observed, predicted), "residuals.png")
    _save(plot_residual_qq(observed, predicted), "residual_qq.png")
    _save(plot_bland_altman(observed, predicted), "bland_altman.png")
# END example

if __name__ == "__main__":
    gallery = Path("docs/source/_static/images/examples")
    gallery.mkdir(parents=True, exist_ok=True)
    mapping = {
        "out/habitat_pca_2d.png": "viz_habitat_pca_2d.png",
        "out/kaplan_meier.png": "viz_kaplan_meier.png",
        "out/predicted_vs_observed.png": "viz_predicted_vs_observed.png",
        "out/residuals.png": "viz_residuals.png",
        "out/residual_qq.png": "viz_residual_qq.png",
        "out/bland_altman.png": "viz_bland_altman.png",
        "out/cox_forest.png": "viz_cox_forest.png",
        "out/risk_triptych.png": "viz_risk_triptych.png",
        "out/survival_calibration.png": "viz_survival_calibration.png",
        "out/time_dependent_auc.png": "viz_time_dependent_auc.png",
        "out/brier_curve.png": "viz_brier_curve.png",
    }
    for src, name in mapping.items():
        (gallery / name).write_bytes(Path(src).read_bytes())
    print("Copied gallery PNGs")
