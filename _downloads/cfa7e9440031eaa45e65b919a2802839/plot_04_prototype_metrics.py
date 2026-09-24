"""
Choosing the distance for prototype matching
============================================

The prototype matcher accepts four distances. Each comes with the
prototype update that minimises it, so the loop still converges:

=================  ===========================  ================================
``metric``          prototype                    compares
=================  ===========================  ================================
``"sqeuclidean"``   mean (k-means, default)      values
``"manhattan"``     per-feature median           values, robust to one outlier
``"cosine"``        mean direction (spherical)   direction only, ignores level
``"correlation"``   cosine of row-centred rows   shape only, ignores level/offset
=================  ===========================  ================================

Synthetic habitats with three enhancement features (arterial, portal,
delayed) and four known types make the difference visible:

* **weak** and **strong** persistent enhancement have the *same curve
  shape* at different levels -- cosine and correlation cannot tell them
  apart;
* **washout** and **plateau** have other shapes;
* one subject's strong habitat is an outlier, which pulls a mean
  prototype but not a median one.
"""

# %%
# Five subjects, known habitat types
# ----------------------------------
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.kernels.habitat_label_match import PROTOTYPE_METRICS, match_rows_to_prototypes
from habit.viz import plot_prototype_matching

FEATURES = ("enh_arterial", "enh_portal", "enh_delayed")
TYPES = {
    "weak": [0.4, 0.6, 0.7],
    "strong": [1.6, 2.4, 2.8],  # 4 x weak: same shape, higher level
    "washout": [2.0, 1.2, 0.8],
    "plateau": [1.2, 1.3, 1.35],
}
type_names = list(TYPES)
rng = np.random.default_rng(7)
blocks, truth = [], []
for subject in range(5):
    order = rng.permutation(len(type_names))  # each subject numbers habitats its own way
    rows = np.array([TYPES[type_names[i]] for i in order])
    rows = rows * rng.normal(1.0, 0.06, rows.shape)
    blocks.append(rows)
    truth.append([type_names[i] for i in order])
# The last subject's strong habitat enhances twice as much: an outlier.
outlier_row = truth[-1].index("strong")
blocks[-1][outlier_row] *= 2.0
names = [f"S{i + 1}" for i in range(len(blocks))]

# %%
# Match with every metric
# -----------------------
# ``truth`` is only used to score the result: a perfect naming puts each
# habitat type under exactly one prototype.
results = {metric: match_rows_to_prototypes(blocks, metric=metric) for metric in PROTOTYPE_METRICS}
purity = {}
for metric, result in results.items():
    table = pd.crosstab(
        pd.Series(np.concatenate(truth), name="true type"),
        pd.Series(np.concatenate(result.assignments) + 1, name=f"{metric} name"),
    )
    purity[metric] = int(table.max(axis=0).sum())
    print(table, end="\n\n")
print("habitats named consistently with their type (of 20):", purity)

# %%
# The four namings in the arterial / portal plane
# -----------------------------------------------
# Crosses are fitted prototypes (means for ``sqeuclidean``, medians for
# ``manhattan``). Cosine prototypes are directions, drawn as rays: every
# point on a ray is identical to cosine. Correlation prototypes live in
# row-centred space, so only the colours are shown.
Path("out").mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.8), sharex=True, sharey=True)
for ax, (metric, result) in zip(axes.flat, results.items()):
    plot_prototype_matching(blocks, result, block_names=names, feature_names=FEATURES, ax=ax)
fig.tight_layout()
fig.savefig("out/prototype_metrics.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Mean vs median prototype of the strong habitats
# -----------------------------------------------
# The outlier drags the ``sqeuclidean`` mean away from the other four
# strong habitats; the ``manhattan`` median stays with them. Both still
# name the outlier "strong": it cannot be dropped, only matched.
typical = np.array(TYPES["strong"])
for metric in ("sqeuclidean", "manhattan"):
    result = results[metric]
    strong_id = int(result.assignments[0][truth[0].index("strong")])
    print(f"{metric:12s} strong prototype {result.prototypes[strong_id].round(2)} (typical {typical})")

# %%
# When cosine / correlation are the right choice
# ----------------------------------------------
# They answer "which habitats have the same curve shape?", e.g. when the
# overall enhancement level differs between scanners and only the
# kinetics are comparable. With a level difference that matters (weak vs
# strong here), keep ``"sqeuclidean"``; put features of different units
# on one scale with ``standardize="zscore"`` in
# :func:`~habit.precision.align_habitat_maps_to_prototypes`.
#
# They degenerate with few features. HABIT does not refuse fewer than
# three features, but with **one** feature of constant sign every row has
# the same cosine direction (all distances 0, names arbitrary) and every
# row is constant for correlation (undefined, so it raises):
one_feature = [block[:, :1] for block in blocks]
print("cosine, 1 feature, objective:", match_rows_to_prototypes(one_feature, metric="cosine").objective)
try:
    match_rows_to_prototypes(one_feature, metric="correlation")
except ValueError as error:
    print("correlation, 1 feature:", error)

# %%
# With **two** features a row-centred vector is ``(+d, -d)``, so
# correlation only sees which feature is larger (r = +1 or -1).
two_feature = [block[:, :2] for block in blocks]
two = match_rows_to_prototypes(two_feature, metric="correlation")
print("correlation, 2 features, distances:", np.unique(np.round(np.concatenate(two.distances), 6)))
