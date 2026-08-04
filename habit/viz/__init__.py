# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Publication figures for HABIT results.

Every function in this package is PURE: it takes contract objects (or plain
arrays), draws on a matplotlib ``Figure``, and returns that figure. Nothing
here touches the filesystem -- there is no ``savefig``, no ``show``, no
output-directory parameter anywhere in the package. Where a figure ends up is
entirely the caller's decision, which is what makes the same function usable
from a notebook (``fig`` is shown and can be restyled), from a script
(``fig.savefig(...)``), and from the CLI's sink (which owns persistence).

Two consequences of that rule:

- a figure's geometry, typography and palette come from a STYLE PRESET
  (:func:`use_style`), never from a per-plot hard-coding, so one figure can be
  re-rendered for a different journal without touching the plotting code;
- every piece of text drawn on a figure is guaranteed ASCII via
  :func:`~habit.viz.labels.sanitize_label`, because data-driven labels (a
  feature or group name) can otherwise leak non-ASCII characters onto an
  axis that a journal will reject.
"""

from __future__ import annotations

from habit.viz.labels import sanitize_label
from habit.viz.regression import (
    plot_bland_altman,
    plot_coefficient_forest,
    plot_predicted_vs_observed,
    plot_residual_qq,
    plot_residuals,
)
from habit.viz.style import (
    StyleSpec,
    available_styles,
    get_style,
    register_style,
    use_style,
)
from habit.viz.habitat_clustering import plot_habitat_clustering_pca_2d
from habit.viz.survival import (
    plot_brier_curve,
    plot_cox_forest,
    plot_kaplan_meier,
    plot_risk_triptych,
    plot_survival_calibration,
    plot_time_dependent_auc,
)

__all__ = [
    "StyleSpec",
    "use_style",
    "get_style",
    "register_style",
    "available_styles",
    "sanitize_label",
    # survival
    "plot_kaplan_meier",
    "plot_risk_triptych",
    "plot_time_dependent_auc",
    "plot_survival_calibration",
    "plot_brier_curve",
    "plot_cox_forest",
    # regression
    "plot_predicted_vs_observed",
    "plot_residuals",
    "plot_residual_qq",
    "plot_bland_altman",
    "plot_coefficient_forest",
    # habitat clustering
    "plot_habitat_clustering_pca_2d",
]
