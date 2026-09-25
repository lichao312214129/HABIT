"""
Prototype matching step by step
===============================

**Background.** Two patients share no voxels, so their habitats cannot be
matched by overlap. Instead each habitat is described by a summary row and
all subjects are named against one shared set of reference rows.

**Purpose.** You watch the matching loop round by round on synthetic data,
confirm it equals the one-line HABIT call, and see three demo tumours
before and after their ids are put on one shared scale.

**Key terms.**

* **prototype** -- a shared reference habitat: one feature row (for
  example mean arterial and portal-venous enhancement) that every subject's
  habitats are compared with; habitat ``k`` after matching means "closest
  to prototype ``Pk``".
* **Hungarian assignment** -- see
  :doc:`/auto_examples/06_matching/plot_01_label_switching`; here it
  minimises total squared distance instead of maximising shared voxels.
* **objective** -- the sum of squared distances from every habitat to its
  assigned prototype; lower means tighter groups.
* **centroid** -- the mean feature row of one habitat as fitted by
  k-means.

Between **different subjects** there is no shared voxel, so habitats are
matched by what describes them: one summary row per habitat (a centroid,
a mean feature vector). HABIT names every subject against ``K`` shared
prototypes, ``K`` = the largest habitat count, by alternating two steps:

1. **assign** -- match each subject one-to-one onto the prototypes
   (Hungarian on squared Euclidean distance), so two habitats of one
   tumour never get the same name;
2. **update** -- move every prototype to the mean of the habitats
   assigned to it.

Both steps can only lower the objective (sum of squared distances to the
assigned prototype), so the loop stops. This is Stephens' (2000)
relabelling algorithm, i.e. k-means with a cannot-link constraint inside
each subject.

This page runs the loop by hand on five synthetic subjects with 2-4
habitats, shows each round, and checks it against the one-line call.
Then the two-subject case, where the method reduces to pairwise
Hungarian matching on squared distance, and finally three real tumours
before and after matching.
"""

# %%
# Five subjects, 2-4 habitats each
# --------------------------------
# Two features per habitat (think arterial and portal-venous
# enhancement). Four underlying habitat types; no subject has all of
# them in a clean form, and the noise is large enough that the first
# guess is wrong.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from habit.kernels.habitat_label_match import match_rows_to_prototypes
from habit.viz import plot_prototype_matching

types = np.array([[0.6, 0.7], [1.2, 2.2], [2.2, 0.9], [2.4, 2.4]])
present = [(0, 1), (0, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (1, 2, 3)]
rng = np.random.default_rng(54)
# One block per subject: row i is habitat i's summary, in that subject's own order.
blocks = [np.round(types[list(p)] + rng.normal(0, 0.38, (len(p), 2)), 2) for p in present]
names = [f"S{i + 1}" for i in range(len(blocks))]
for name, block in zip(names, blocks):
    print(name, block.tolist())
K = max(block.shape[0] for block in blocks)

# %%
# The loop by hand
# ----------------
# Start from the habitats of S3 (a subject with ``K`` habitats). The
# assign step is :func:`~habit.kernels.habitat_label_match.match_rows_to_prototypes`
# with ``prototypes=`` fixed (one assignment pass, no update); the update
# step is a plain mean.
prototypes = blocks[2].copy()
rounds = []
while True:
    step = match_rows_to_prototypes(blocks, prototypes=prototypes)
    changed = not rounds or any(
        not np.array_equal(new, old)
        for new, old in zip(step.assignments, rounds[-1].assignments)
    )
    rounds.append(step)
    if not changed:
        # Same names as last round: the prototypes are already the means.
        break
    prototypes = np.vstack(
        [
            np.vstack(
                [block[assigned == k] for block, assigned in zip(blocks, step.assignments)]
            ).mean(axis=0)
            for k in range(K)
        ]
    )
print(f"names unchanged in round {len(rounds)}: converged")
print("objective per round:", [round(r.objective, 3) for r in rounds])

# %%
# Each panel is one assign step: colours are the names given, crosses
# the prototypes they were matched to. Watch habitats change colour as the
# prototypes move toward the true groups.
Path("out").mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6), sharex=True, sharey=True)
for index, (ax, step) in enumerate(zip(axes.flat, rounds)):
    plot_prototype_matching(
        blocks,
        step,
        block_names=names,
        feature_names=("enhancement 1", "enhancement 2"),
        title=f"round {index + 1}: objective {step.objective:.2f}",
        ax=ax,
    )
for ax in list(axes.flat)[len(rounds):]:
    ax.set_visible(False)
fig.tight_layout()
fig.savefig("out/prototype_rounds.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# The one-line call
# -----------------
# :func:`~habit.kernels.habitat_label_match.match_rows_to_prototypes`
# runs the same loop from **every** subject with ``K`` habitats and keeps
# the lowest objective, so the result does not depend on who starts.
# Prototypes are then sorted by feature 1 (then 2) for stable numbering.
fit = match_rows_to_prototypes(blocks)
order = np.lexsort(prototypes.T[::-1])
print(f"objective: by hand {rounds[-1].objective:.4f}, call {fit.objective:.4f}")
print("same prototypes:", np.allclose(prototypes[order], fit.prototypes))
print(f"winning start: S{fit.init_block + 1}, rounds {fit.n_iter}, converged {fit.converged}")

fig = plot_prototype_matching(
    blocks, fit, block_names=names, feature_names=("enhancement 1", "enhancement 2")
)
fig.savefig("out/prototype_final.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Two subjects: a special case
# ----------------------------
# With two subjects each prototype holds one habitat of each, and a pair's
# squared distance to its midpoint is :math:`\tfrac12\lVert a-b\rVert^2`.
# Minimising the objective is therefore pairwise Hungarian on **squared**
# distance -- and the loop reaches that global optimum.
a = np.array([[0.0, 2.9], [2.7, 2.6], [2.7, 2.3]])
b = np.array([[0.5, 2.7], [0.0, 0.7], [1.7, 1.5]])
distance = cdist(a, b)
_, pairs_squared = linear_sum_assignment(distance**2)
_, pairs_plain = linear_sum_assignment(distance)
pair = match_rows_to_prototypes([a, b])
# Rows of a and b that share a prototype are partners.
pairs_prototype = [int(np.flatnonzero(pair.assignments[1] == k)[0]) for k in pair.assignments[0]]
print("partner of each row of a")
print("  Hungarian, squared distance:", pairs_squared.tolist())
print("  Hungarian, plain distance:  ", pairs_plain.tolist())
print("  prototype matching:         ", pairs_prototype)

# %%
# Plain distance prefers a different pairing here: it tolerates one long
# link to keep two short ones, squared distance penalises the long link.
# The two agree in most cohorts; when they disagree, the prototype method
# follows squared distance.
fig = plot_prototype_matching([a, b], pair, block_names=["A", "B"])
fig.savefig("out/prototype_two_subjects.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# The same on real tumours
# ------------------------
# Three demo lesions, each clustered on its own into 3 habitats on
# relative arterial and portal-venous enhancement. Before matching,
# "habitat 1" is whatever k-means numbered first in that tumour. After
# :func:`~habit.precision.align_habitat_maps_to_prototypes`, habitat ``k``
# is prototype ``Pk`` in every tumour, so a colour means the same
# enhancement pattern across the three figures below.
from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.precision import align_habitat_maps_to_prototypes
from habit.viz import plot_habitat_label_compare
from habit.voxel_features import ExpressionVoxelFeatures

# Change DATA / MODALITIES / ROI and the expressions to your own layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:3]
extractor = ExpressionVoxelFeatures(
    features={
        "rel_enh_lap": "(LAP - pre_contrast) / (pre_contrast + eps)",
        "rel_enh_pvp": "(PVP - pre_contrast) / (pre_contrast + eps)",
    },
    roi=ROI,
)
raw_maps, models = [], []
for subject in cohort:
    # Per-subject fit: each tumour numbers its own 3 habitats independently.
    units = voxel_units(extractor(subject))
    fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
    fitter.set_random_state(0)
    model = fitter.fit([units], cohort=Cohort([subject], name=subject.subject_id))
    raw_maps.append(model.assigner()(units))
    models.append(model)

# models= describes each habitat by its fitted k-means centroid.
matched = align_habitat_maps_to_prototypes(raw_maps, models=models)
print(matched.assignments.to_string(index=False))

# %%
# Left: the tumour's own k-means ids. Right: the same voxels renamed onto
# the shared prototypes. Compare the right-hand panels across tumours.
for subject, raw, named in zip(cohort, raw_maps, matched.habitat_maps):
    fig = plot_habitat_label_compare(
        subject.image(ROI),
        raw,
        named,
        titles=(f"{subject.subject_id}: own ids", f"{subject.subject_id}: prototype ids"),
        align_labels=False,
        show_disagreement=False,
        crop_to="labels",
    )
    fig.savefig(f"out/prototype_real_{subject.subject_id}.png", dpi=150, bbox_inches="tight")
    plt.show()
