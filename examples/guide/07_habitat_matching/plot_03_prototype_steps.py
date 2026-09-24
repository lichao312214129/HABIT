"""
Prototype matching step by step
===============================

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
habitats, shows each round, and checks it against the one-line call. It
ends with the two-subject case, where the method reduces to pairwise
Hungarian matching on squared distance.
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
