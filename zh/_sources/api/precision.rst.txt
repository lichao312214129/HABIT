:mod:`habit.precision`: repeatability screen
============================================

.. automodule:: habit.precision
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.precision

**User guide:** Habitat Guide
:doc:`../auto_examples/05_validation_and_reuse/plot_05_precise_features` ·
:doc:`../auto_examples/05_validation_and_reuse/plot_02_match_same_subject` ·
:doc:`../auto_examples/05_validation_and_reuse/plot_03_match_two_subjects` ·
:doc:`../auto_examples/05_validation_and_reuse/plot_04_match_group`.
Recipe wrapper: :func:`~habit.recipes.identify_precise_voxel_features`.

Simulated-retest perturbations and the ICC intersection that decides
which extracted voxel columns may define habitats (Prior et al.,
Radiol Artif Intell 2024;6(2):e230118). Guide matching examples use
:func:`~habit.precision.align_habitat_maps_to_prototypes`. Two-step and
pooled-voxel fits share one model and do not need matching. Short pointer:
:doc:`../reference/habitat_matching`.

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ImagePerturbation
   GaussianNoisePerturbation
   TranslationPerturbation
   RotationPerturbation
   RigidPerturbation
   BSplineDeformPerturbation
   MorphologicalPerturbation
   GradientWeightedPerturbation
   SliceExtentPerturbation
   PerturbationChain
   PreciseFeatureSet
   ImagePerturbationRegistry
   HabitatPrototypeAlignment

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   perturb_image
   prior2024_retest_perturbation
   precision_panel
   aggregate_panels
   identify_precise_features
   align_habitat_map
   align_habitat_maps_to_prototypes
   habitat_stability
