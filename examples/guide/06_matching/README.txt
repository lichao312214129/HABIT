6. Matching Habitat Labels
==========================

**Background.** A clustering run numbers its habitats in arbitrary order,
so the same tissue can be habitat 1 in one run or patient and habitat 3 in
another (**label switching**). **Purpose.** These pages show how HABIT
matches the ids so that Dice, volume fractions and cohort tables compare
like with like.

Calling it on a fitted study:
:doc:`/auto_examples/06_matching/plot_07_match_labels`.
The pages below show why ids switch and how each matcher works.

Habitat ids from independent clusterings are arbitrary: habitat 1 of one
fit can be habitat 3 of another. Anything that compares habitats by id
(Dice, volume fractions, a cohort feature table) must match the ids
first. HABIT has two matchers, chosen by what the two sides share:

* **Voxel overlap** -- the maps label the *same voxels* (a restart,
  another ``k``, another feature set, a perturbed image, a second reader).
  Hungarian assignment on the voxel-overlap table.
  :func:`~habit.precision.align_habitat_map`,
  :func:`~habit.precision.habitat_stability`.
* **Shared prototypes** -- the maps label *different subjects*. Each
  subject's habitat summaries are matched one-to-one onto ``K`` shared
  prototypes, iterated until stable; prototypes can be frozen to name a
  new cohort. :func:`~habit.precision.align_habitat_maps_to_prototypes`.

A shared cohort model (two-step, direct pooling, an applied saved model)
already uses one id space and needs neither. Formulas, proofs, and
literature: :doc:`/reference/habitat_matching`.
