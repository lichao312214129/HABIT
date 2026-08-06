Precise features: screen before you cluster
===========================================

Voxel-wise radiomics is noisy: extracting a feature map twice from the same
anatomy — a simulated re-acquisition, or a slightly different kernel radius
or bin width — does not yield the same numbers twice. Clustering features
that do not survive such perturbation produces habitats nobody can
reproduce. The precision screen of Prior et al. (*Radiol Artif Intell*
2024;6(2):e230118) answers the question **which voxel features deserve to
define habitats at all**, and HABIT implements it as a first-class recipe.

The three experiments
---------------------

Per subject, :func:`~habit.recipes.identify_precise_voxel_features` runs up
to three experiments over the feature maps:

* **repeatability** — ICC(3A,1) between the base-setting maps of the
  original image and of one perturbed copy (Gaussian noise estimated by
  Chang's wavelet method, a sub-voxel translation, a 0.5-degree in-plane
  rotation — the paper's simulated retest);
* **reproducibility_kernel_radius** — ICC(3C,1) between maps at the given
  kernel radii (the paper contrasts R1 with R3) at fixed bin width;
* **reproducibility_bin_width** — ICC(3C,1) between maps at the given bin
  widths (the paper contrasts B12 with B25 HU) at fixed radius.

Per-feature per-subject ICCs are aggregated by the cohort median, and a
feature is *precise* when its median lower confidence limit reaches
``lcl_threshold`` (0.5, the paper's "at least good" boundary) in **every**
experiment — subject to the ``include`` / ``exclude`` expert overrides (the
paper used ``include`` for NGTDM Coarseness).

The default extractor is the bundled CT voxel-radiomics preset
(``voxel_radiomics_factory`` forwards the grid point as
``kernel_radius`` / ``binWidth``); any extractor with a
``(kernel_radius, bin_width) -> extractor`` factory can be screened instead.

From screen to habitat run
--------------------------

The returned :class:`~habit.domain.precision.PreciseFeatureSet` is a
serialisable artefact (``save`` / ``load``) carrying the full evidence
panels — publish it alongside the study so others cluster the SAME
features. Its ``preprocessor()`` bridges into a habitat spec: a
``feature_whitelist`` preprocessing step that restricts the clustering
matrix to exactly the precise features. Place it first in
``voxel_feature_preprocessors`` so only precise features reach scaling and
clustering.

Runnable demo
-------------

The demo uses raw intensities so it runs without PyRadiomics; with the
default voxel-radiomics factory the same call is the full paper design.

.. literalinclude:: scripts/precise_features_demo.py
   :language: python

Output::

   === precision screen (repeatability experiment) ===
     precise features: ['T1', 'T2']
      experiment feature  value   lcl   ucl  n_voxels  precise
   repeatability      T1  0.884 0.878 0.890    5326.0     True
   repeatability      T2  0.868 0.862 0.875    5326.0     True

   === whitelist bridge into a habitat spec ===
     whitelist spec: {'name': 'feature_whitelist', 'params': {'features': ['T1', 'T2']}, 'version': '1.0'}
     habitat spec fingerprint: ed7eafe731aeac8cda189e29a4ba9146e77415d3063d71505f398cacaaff4790

   === artefact round trip ===
     saved precise_features.json; reloaded features: ['T1', 'T2']

Custom perturbations
--------------------

The simulated retest is a composable chain of ``ImagePerturbation``
components (domain ``image_perturbation``): ``gaussian_noise``,
``translation``, ``rotation``. Third-party perturbations registered through
the ``habit.image_perturbation`` entry-point group slot into the same
chain — pass any chain (or single component) as ``perturbation=``.

.. code-block:: python

   from habit.domain import (
       GaussianNoisePerturbation,
       PerturbationChain,
       TranslationPerturbation,
   )

   gentler = PerturbationChain(
       [GaussianNoisePerturbation(), TranslationPerturbation(max_shift_voxels=0.5)]
   )
   precise = identify_precise_voxel_features(cohort, perturbation=gentler)

Habitat stability under perturbation
------------------------------------

Beyond feature-level screening, :func:`~habit.domain.precision.habitat_stability`
scores a habitat MAP against perturbed re-computations: clusters are matched
with the Hungarian algorithm and compared by Dice coefficient, per habitat.

What to read next
-----------------

* :doc:`../api/domain` — the ``ImagePerturbation`` protocol and registry
* :doc:`../api/kernels` — the perturbation and voxel-ICC numeric kernels
* :doc:`habitat_preprocessing_api` — the preprocessing chains the whitelist
  joins
