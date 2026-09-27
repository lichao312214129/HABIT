Matching habitat labels
=======================

Clustering numbers habitats in arbitrary order when each fit is
**separate**. All three short Guide examples use prototype matching
(:func:`~habit.precision.align_habitat_maps_to_prototypes`):

* Same subject, two seeds:
  :doc:`/auto_examples/05_validation_and_reuse/plot_02_match_same_subject`
* Two subjects, same Spec and seed:
  :doc:`/auto_examples/05_validation_and_reuse/plot_03_match_two_subjects`
* Group one-step fits:
  :doc:`/auto_examples/05_validation_and_reuse/plot_04_match_group`

Two-step and pooled-voxel fits assign labels from **one shared model**,
so those habitat ids are already comparable across subjects and do
**not** need matching. Matching is for separate fits (two seeds, or
one-step per subject). API reference: :doc:`/api/precision`.
