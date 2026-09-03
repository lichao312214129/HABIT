:mod:`habit.precision`: repeatability screen
============================================

.. automodule:: habit.precision
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.precision

**User guide:** Habitat Guide
:doc:`../auto_examples/03_precision/plot_01_precise_features` ·
:doc:`../auto_examples/04_habitat_maps/plot_05_match_labels`.
Recipe wrapper: :func:`~habit.recipes.identify_precise_voxel_features`.

Simulated-retest perturbations and the ICC intersection that decides
which extracted voxel columns may define habitats (Prior et al.,
Radiol Artif Intell 2024;6(2):e230118). Label matching
(``align_habitat_map``) lives here too.

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
   habitat_stability
