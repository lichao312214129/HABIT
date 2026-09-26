"""
Texture feature maps, alone and with intensity
==============================================

**Background.** Two regions of a tumour can have the same mean
brightness but a different local pattern: one smooth, one speckled.
Intensity alone cannot separate them. Voxel-wise texture gives every
voxel a few numbers that describe its small neighbourhood. Those
numbers can replace the intensity in the feature table that the
habitat model clusters, or sit next to it.

**Purpose.** You will compute two GLCM texture maps (Contrast and Idm)
around every voxel of the arterial-phase (LAP) image and run the
two-step analysis twice: once on the two texture columns alone, once on
LAP intensity plus the same two texture columns. You will then compare
the two runs by the number of habitats the elbow rule picks, by their
centroids, and by how much their habitat maps agree (adjusted Rand
index, which ignores how the habitat ids are numbered).

**When to use.** You expect habitats to differ in local structure
(e.g. heterogeneous enhancement or fine necrosis), not only in how
bright they are, and you want to know how much the intensity column
changes the habitats. Voxel-wise texture is slower to extract than raw
intensity.

**Analysis definition.** The approved two-step definition with LAP as
the only image, and exactly ONE factor varied between the two runs:
the feature set of the ``extract`` stage (texture only vs intensity +
texture). Subject-level winsorize + z-score, 30 k-means supervoxels,
cohort-level 10-bin binning, k-means 2..10 habitats by elbow,
nearest-centroid assignment and the seed are identical in both runs.
Both runs are fitted on subj001 + subj002. The quantify stages are cut
to ``volume`` on this page to keep it fast.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **voxel-wise GLCM** -- a grey-level co-occurrence matrix computed in a
  small cube around each voxel; *Contrast* is high where neighbouring
  grey levels differ a lot, *Idm* (inverse difference moment) is high
  where the neighbourhood is homogeneous.
* **kernel radius** -- half the size of that cube: radius 1 means a
  3 x 3 x 3 neighbourhood.
* **bin width** -- the grey-level discretisation used before the GLCM is
  counted, in the image's own intensity units.
* **concat extractor** -- several voxel feature extractors run on the
  same ROI and their columns are placed side by side, one row per voxel.
* **adjusted Rand index (ARI)** -- agreement between two label maps of
  the same voxels: 1 = the same partition (whatever the ids are called),
  about 0 = no more agreement than chance.

Texture parameters are part of the method: report them with the
results.
"""

# %%
# Load the cohort
# ---------------
# Only the arterial phase is loaded: both the intensity column and the
# texture columns are computed on it. Two subjects define the habitats.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from habit.contracts import HabitatModel, HabitatMap, cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import (
    plot_cluster_validation_from_report,
    plot_habitat_overlay,
    plot_voxel_texture_slice,
)
from habit.voxel_features import build_voxel_extractor

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train = cohort[:2]
print(train)
Path("out").mkdir(exist_ok=True)

# %%
# The two feature sets
# --------------------
# ``TEXTURE`` holds the voxel-wise GLCM settings shared by both runs, so
# the texture columns are computed in exactly the same way:
#
# * ``kernel_radius=1`` -- a 3 x 3 x 3 cube around each voxel. A larger
#   radius smooths the map and costs much more time; radius 1 keeps this
#   page fast.
# * ``binWidth=25`` -- intensities are grouped into bins 25 units wide
#   before the co-occurrence matrix is counted. The units are those of
#   the image (raw MR signal), so the same ``binWidth`` gives a
#   different number of grey levels on a brighter or darker scan. This
#   is an IBSI discretisation choice (fixed bin width vs fixed bin
#   count): choose it for your data and report it.
#
# The only difference between the two ``extract`` specs is whether the
# ``raw`` LAP intensity column is added in front of the texture columns.
TEXTURE = {
    "modalities": list(MODALITIES),
    "kernel_radius": 1,
    # One (or a few) features from every texture family the voxel-radiomics
    # extractor supports. Families, not every IBSI name: enough to show
    # the route without making this page a full IBSI dump.
    "params": {
        "setting": {"binWidth": 25},
        "featureClass": {
            "glcm": ["Contrast", "Idm", "Correlation", "JointEntropy"],
            "glrlm": ["ShortRunEmphasis", "LongRunEmphasis", "RunEntropy"],
            "glszm": ["SmallAreaEmphasis", "LargeAreaEmphasis", "ZoneEntropy"],
            "gldm": ["SmallDependenceEmphasis", "LargeDependenceEmphasis"],
            "ngtdm": ["Coarseness", "Contrast", "Busyness"],
        },
    },
}
# Texture only: two columns per voxel (GLCM Contrast, GLCM Idm).
EXTRACT_TEXTURE = Spec("voxel_radiomics", {**TEXTURE, "roi": ROI})
# Intensity + texture: the LAP intensity column, then the same two
# texture columns, side by side (one row per voxel).
EXTRACT_BOTH = Spec(
    "concat",
    {
        "extractors": [
            {"name": "raw", "params": {"modalities": list(MODALITIES)}},
            {"name": "voxel_radiomics", "params": TEXTURE},
        ],
        "roi": ROI,
    },
)


def texture_spec(name: str, extract: Spec) -> HabitatSpec:
    """
    Build the two-step spec used by both runs, with a given extract stage.

    Args:
        name: Name of the analysis (stored in the model and manifest).
        extract: The ``extract`` component; the only thing that differs
            between the two runs on this page.

    Returns:
        A :class:`~habit.spec.HabitatSpec` whose every other stage is the
        approved two-step definition.
    """
    return HabitatSpec(
        name=name,
        stages=(
            Stage("extract", extract),
            # preprocess (subject-level): clip each column to that patient's
            # own 1st..99th percentile.
            Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
            # preprocess2 (subject-level): rescale each column to 0..1 with
            # that patient's own min and max. Intensity (hundreds), Contrast
            # (can be much larger) and Idm (0..1) have different units;
            # k-means uses Euclidean distance, so without this step the
            # widest column would decide the habitats alone.
            Stage("preprocess2", Spec("zscore")),
            # partition: 30 supervoxels per tumour, on all extracted columns.
            Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
            # pool: stack both training subjects' supervoxels.
            Stage("pool", Spec("pool")),
            # preprocess_cohort (cohort-level): 10 equal-width bins per
            # column, edges learned on the pooled training supervoxels.

            # fit: one k-means for the cohort, 2..10 habitats, elbow rule.
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10},
                ),
            ),
            # assign: nearest centroid for every supervoxel, then its voxels.
            Stage("assign", Spec("nearest_centroid")),
            # quantify: volume fractions only on this page.
            Stage("volume", Spec("volume")),
        ),
        random_seed=0,
    )


spec_texture = texture_spec("texture_only_two_step", EXTRACT_TEXTURE)
spec_both = texture_spec("intensity_texture_two_step", EXTRACT_BOTH)

# %%
# Look at one texture map
# -----------------------
# Before fitting, build only the ``extract`` component and call it on
# the first subject. The result is the per-voxel table the model will
# see: one row per ROI voxel, one column per feature. The column names
# carry the feature and the modality it was computed on.
subject = train[0]
field = build_voxel_extractor(EXTRACT_BOTH)(subject)
print(f"{field.values.shape[0]} voxels, columns: {list(field.feature_names)}")
print(field.feature_frame().describe().round(3))

contrast_name = next(name for name in field.feature_names if "Contrast" in name)
fig = plot_voxel_texture_slice(
    field,
    feature=contrast_name,
    anatomy=subject.image("LAP"),
    roi_mask=subject.mask(ROI),
    title=f"{subject.subject_id}: GLCM Contrast, kernel radius 1",
    crop_to="roi",
)
fig.savefig("out/texture_maps_contrast.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Fit both runs
# -------------
# Same training subjects, same seed, same stages; only ``extract``
# differs. The texture columns are computed inside each ``fit_predict``
# (the log lines are the voxel batches).
result_texture = Study(spec_texture).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
result_both = Study(spec_both).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
model_texture = result_texture.habitat_model
model_both = result_both.habitat_model
print("texture only        : K =", model_texture.n_habitats, "| features:", list(model_texture.feature_names))
print("intensity + texture : K =", model_both.n_habitats, "| features:", list(model_both.feature_names))

# %%
# How many habitats
# -----------------
# HABIT validation ``elbow`` is the same rule as ``kneedle``. It runs
# ``KneeLocator`` on k-means inertia (within-cluster sum of squares),
# curve convex, direction decreasing (Satopaa, Albrecht, Irwin, and
# Raghavan, 2011, Finding a "Kneedle" in a Haystack, IEEE ICDCS).
# Normalize k and inertia to the unit square, draw the chord from the
# first point to the last, and take the k farthest from that chord; a
# smooth curve often places this knee to the right of the bend a person
# sees. Literature "elbow" means inspecting within-cluster dispersion
# versus k (Thorndike RL, 1953, Who belongs in the family?, Psychometrika
# 18(4):267-276) — not a second-difference formula. The discrete-curvature
# elbow (visual elbow, computed) maximizes the second difference of
# inertia (HABIT pre-v1.0 elbow); it is not a Thorndike formula. The
# habitat map keeps the Kneedle K; other Ks are only in the table below.
# Full comparison: :doc:`/auto_examples/03_clustering/plot_03_clustering_algorithm`.

def discrete_curvature_elbow_k(cluster_range, inertia_scores):
    """Return k maximizing the second difference of inertia (pre-v1.0 elbow)."""
    values = np.asarray(inertia_scores, dtype=float)
    cluster_range = [int(k) for k in cluster_range]
    if values.size < 3:
        return int(cluster_range[int(np.argmin(values))])
    return int(cluster_range[int(np.argmax(np.diff(values, n=2))) + 1])


from habit.habitat_model import KMeansHabitatModelFitter

for label, model, units_ref in (
    ("texture only", model_texture, result_texture.units),
    ("intensity + texture", model_both, result_both.units),
):
    report = model.preprocessing_state["selection_report"]
    candidates = [int(k) for k in report["candidates"]]
    inertia = np.asarray(report["scores"][report["methods"][0]], dtype=float)
    k_kneedle = int(model.n_habitats)
    k_discrete = discrete_curvature_elbow_k(candidates, inertia)
    _k_table = {"elbow_kneedle": k_kneedle, "discrete_curvature_elbow": k_discrete}
    for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
        _fitter = KMeansHabitatModelFitter(
            min_habitats=2, max_habitats=10, validation=_name, n_init=10
        )
        _fitter.set_random_state(0)
        _k_table[_name] = _fitter.fit(units_ref, cohort=train).n_habitats
    print(f"{label}:")
    for _name, _k in _k_table.items():
        print(f"  {_name}: {_k}")
    # Plot every supported k-means criterion (skip gap).
    _criteria = ("elbow", "silhouette", "calinski_harabasz", "davies_bouldin")
    _vote = KMeansHabitatModelFitter(
        min_habitats=2, max_habitats=10, validation=list(_criteria), n_init=10
    )
    _vote.set_random_state(0)
    _vote_report = dict(
        _vote.fit(units_ref, cohort=train).preprocessing_state["selection_report"]
    )
    _vote_report["selected"] = {
        "elbow": k_kneedle,
        "silhouette": _k_table["silhouette"],
        "calinski_harabasz": _k_table["calinski_harabasz"],
        "davies_bouldin": _k_table["davies_bouldin"],
    }
    fig = plot_cluster_validation_from_report(
        _vote_report,
        title=f"{label}: K criteria (elbow panel: x = Kneedle)",
    )
    fig.axes[0].plot(
        k_discrete,
        float(inertia[candidates.index(k_discrete)]),
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor="C2",
        markeredgewidth=1.4,
        linestyle="none",
        label=f"discrete-curvature k={k_discrete}",
    )
    fig.axes[0].legend(loc="best", fontsize=8)
    stem = label.replace(" + ", "_").replace(" ", "_")
    fig.savefig(f"out/texture_maps_elbow_{stem}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# What each habitat means
# -----------------------
# Each centroid is one row; its columns are the model's feature names in
# order. The centroids live in the space the model was fitted in: after
# subject-level 0..1 scaling and cohort-level binning, so every value is
# a bin index between 0 and 9 (0 = low for that column, 9 = high). High
# Contrast is a heterogeneous neighbourhood, high Idm a homogeneous one.


def centroid_table(model: HabitatModel) -> pd.DataFrame:
    """
    Return the model's centroids as a table, one row per habitat.

    Args:
        model: A fitted habitat model.

    Returns:
        DataFrame indexed ``habitat_1``, ``habitat_2``, ... with one column
        per feature name stored in the model.
    """
    centroids = np.asarray(model.centroids)
    return pd.DataFrame(
        centroids,
        columns=list(model.feature_names),
        index=[f"habitat_{hid}" for hid in range(1, centroids.shape[0] + 1)],
    )


# "+ 0.0" turns a printed "-0.00" (a tiny negative float) into "0.00".
print("texture only:")
print((centroid_table(model_texture).round(2) + 0.0).to_string())
print("\nintensity + texture:")
print((centroid_table(model_both).round(2) + 0.0).to_string())

# %%
# Habitat maps side by side
# -------------------------
# The same patient, both runs. Colours are not comparable between the
# two rows: each run numbers its own habitats.
for one, map_texture, map_both in zip(train, result_texture.habitat_maps, result_both.habitat_maps):
    for label, one_map in (("texture only", map_texture), ("intensity + texture", map_both)):
        fig = plot_habitat_overlay(
            one.image("LAP"),
            one_map,
            title=f"{one.subject_id}: {label}",
            crop_to="labels",
        )
        stem = label.replace(" + ", "_").replace(" ", "_")
        fig.savefig(f"out/texture_maps_{one.subject_id}_{stem}.png", dpi=150, bbox_inches="tight")
        plt.show()

# %%
# How much do the two maps agree
# ------------------------------
# Habitat ids are arbitrary names, so "habitat 2 in run A" need not be
# "habitat 2 in run B". The adjusted Rand index compares the two
# partitions of the ROI voxels without using the ids. The cross-table
# shows which texture-only habitat each intensity + texture habitat
# draws its voxels from (row fractions).


def map_agreement(map_a: HabitatMap, map_b: HabitatMap) -> tuple[float, pd.DataFrame]:
    """
    Compare two habitat maps of the same subject voxel by voxel.

    Args:
        map_a: First habitat map (rows of the cross-table).
        map_b: Second habitat map on the same grid (columns).

    Returns:
        The adjusted Rand index over voxels labelled in both maps, and a
        row-normalised cross-table of voxel counts.
    """
    labels_a = np.asarray(map_a.label_array)
    labels_b = np.asarray(map_b.label_array)
    # Only ROI voxels: label 0 is background in both maps.
    inside = (labels_a > 0) & (labels_b > 0)
    a, b = labels_a[inside], labels_b[inside]
    table = pd.crosstab(pd.Series(a, name="both"), pd.Series(b, name="texture_only"), normalize="index")
    return float(adjusted_rand_score(a, b)), table


for map_texture, map_both in zip(result_texture.habitat_maps, result_both.habitat_maps):
    ari, table = map_agreement(map_both, map_texture)
    print(f"{map_both.subject_id}: ARI(intensity + texture, texture only) = {ari:.3f}")
    print(table.round(2).to_string(), "\n")

# %%
# How to read the result
# ----------------------
# When this page was built (subj001 + subj002, seed 0):
#
# * **K.** The elbow rule kept K = 4 on texture alone and K = 5 with
#   intensity added.
# * **Centroids (bin index 0..9).** Texture alone gave two homogeneous
#   habitats (Idm 7.7 with Contrast 0.0; Idm 4.2 with Contrast 1.4) and
#   two heterogeneous ones (Contrast 7.8 and 3.5, Idm about 1). With
#   intensity added, the heterogeneous habitats appear twice, once dark
#   (LAP 1.7) and once bright (LAP 7.2), with similar Contrast (6.5 and
#   6.4); the homogeneous side splits into a dark (LAP 1.2) and bright
#   (LAP 6.8 and 7.0) group.
# * **Map agreement.** ARI between the two maps was 0.342 in subj001
#   and 0.302 in subj002: the partitions are related but far from the
#   same. The cross-tables show why: the dark and the bright
#   heterogeneous habitats of the combined run each take about half of
#   their voxels from texture-only habitats 2 and 3, so intensity cuts
#   across the texture split instead of refining it; the most
#   homogeneous combined habitat (5) takes 87 % of its voxels from the
#   most homogeneous texture-only habitat (4) in both patients.
#
# So the feature set is a real modelling choice: adding one intensity
# column changed K and moved many voxels to a differently defined
# habitat. Neither run is "correct" on two demo patients; choose the
# feature set from the biological question and report it. These numbers
# describe this demo and these settings only.

# %%
# Where to go next
# ----------------
# * The texture kernel on its own, with more features and the GPU path:
#   :doc:`/auto_examples/02_features/plot_02_texture_feature_maps` and
#   :doc:`/auto_examples/02_features/plot_02_texture_feature_maps`.
# * Clustering a texture field directly on one subject:
#   :doc:`/auto_examples/02_features/plot_02_texture_feature_maps`.
# * How the subject-level and cohort-level preprocessing stages behave:
#   :doc:`/auto_examples/07_advanced/plot_02_subject_preprocess`.
# * The same design on intensity alone:
#   :doc:`/auto_examples/02_features/plot_02_texture_feature_maps`.
