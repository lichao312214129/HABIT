Morphology-aware precise screening
==================================

Goal: decide **which voxel features are allowed to define habitats**, then
cluster only those features. This is not a new clustering algorithm. It is
the precision screen of Prior et al. (*Radiol Artif Intell*
2024;6(2):e230118), taught as the same combinatorial experiments the
paper ran.

Runnable gallery (atoms, paper combinations, all-vs-precise habitats):
:doc:`../examples/precise_features`. Volume-level atoms:
:func:`~habit.perturb_image` and
:func:`~habit.extract_voxel_texture`.

What is screened
----------------

A voxel radiomic map :math:`F(\mathbf{x})` is a **local morphological
descriptor**: each voxel's value is computed in a neighbourhood whose size
is the kernel radius (and, for many features, a grey-level bin width). If
that map does not survive a simulated re-acquisition, or a change of
neighbourhood scale, clustering it produces partitions nobody can
reproduce.

The paper's combinations are two atoms, then a small composition:

.. code-block:: python

   from habit import extract_voxel_texture, perturb_image, precision_panel
   import numpy as np

   retest_rng = np.random.default_rng(7)
   noisy = perturb_image(image, method="gaussian_noise", rng=retest_rng)
   shifted = perturb_image(noisy, method="translation", shift_fraction=0.5, rng=retest_rng)
   perturbed = perturb_image(shifted, method="rotation", angle_degrees=0.5, rng=retest_rng)
   feat_r3 = extract_voxel_texture(image, mask, kernel_radius=3, bin_width=12)
   feat_pert = extract_voxel_texture(perturbed, mask, kernel_radius=3, bin_width=12)
   feat_r1 = extract_voxel_texture(image, mask, kernel_radius=1, bin_width=12)
   feat_b25 = extract_voxel_texture(image, mask, kernel_radius=3, bin_width=25)

   repeat = precision_panel(
       {"original": feat_r3, "perturbed": feat_pert}, agreement="absolute"
   )
   kernel = precision_panel({"R1": feat_r1, "R3": feat_r3}, agreement="consistency")
   binw = precision_panel({"B12": feat_r3, "B25": feat_b25}, agreement="consistency")

* **repeatability** — ICC(3A,1) between the base-setting maps of the
  original image and of one perturbed copy;
* **reproducibility_kernel_radius** — ICC(3C,1) between maps at the given
  kernel radii (the paper contrasts R1 with R3) at fixed bin width;
* **reproducibility_bin_width** — ICC(3C,1) between maps at the given bin
  widths (the paper contrasts B12 with B25 HU) at fixed radius.

A feature is *precise* when the **lower confidence limit** of its ICC
reaches ``lcl_threshold`` (default ``0.5``) in **every** experiment that
was actually run. :func:`~habit.identify_precise_features` is that
intersection; :func:`~habit.aggregate_panels` takes the cohort median
when you have more than one subject.

.. figure:: ../_static/images/examples/precise_features_original_vs_perturbed.png
   :alt: Original greyscale slice beside a Gaussian-noise perturbed copy
   :width: 720

   Intensity perturbation: original vs the Appendix S2 chain (Chang
   noise, then 0.5-voxel translation, then 0.5° rotation)
   (:func:`~habit.viz.plot_intensity_slice`). One cropped
   ``demo_data`` subject; not a clinical claim.

.. figure:: ../_static/images/examples/precise_features_perturb_methods.png
   :alt: Original plus three perturb_image methods on one slice
   :width: 720

   Sequential Appendix S2 atoms on one slice: original, after Chang
   noise, after +0.5-voxel translation, after +0.5° rotation. Each
   panel is one :func:`~habit.perturb_image` call. Geometric methods
   resample back onto the original grid.

Why "morphology-aware"
----------------------

Three morphological facts are built into the HABIT screen. They are
easy to miss if one treats ICC as a generic correlation.

1. **The features themselves are morphological.** Kernel radius is a
   neighbourhood scale. The kernel-radius reproducibility experiment asks
   whether the *spatial pattern* of a feature survives a change in that
   scale. A feature that is precise only at one radius is not a stable
   habitat coordinate.
2. **Acquisition perturbation moves anatomy.** Translation and rotation
   are rigid morphological changes of the patient in the scanner. Noise
   is not morphology, but it is part of the paper's simulated retest.
   HABIT can transform **masks with the image** (nearest neighbour) when
   you call the registered component on a Subject, because a shifted
   acquisition images a shifted patient.
3. **Agreement is computed on the common ROI.** HABIT aligns condition
   fields on shared ROI coordinates and drops any voxel that is NaN in
   any condition. The pairing is therefore the **intersection of the
   morphologies**, not a rectangular crop filled with dummy intensities.

How to apply
------------

Work from a directory that has images in HABIT's preprocessed layout
(``demo_data/preprocessed`` in the gallery). One physical line per
shell command. Full copy-ready script:
:doc:`../examples/precise_features`.

**1. Atoms.** Load one image and mask, then perturb and extract::

   import numpy as np
   from habit import cohort_from_directory, extract_voxel_texture, perturb_image

   DATA = "demo_data/preprocessed"
   cohort = cohort_from_directory(DATA, modalities=("LAP",), roi="LAP")
   image = cohort[0].image("LAP")
   mask = cohort[0].mask("LAP")
   retest_rng = np.random.default_rng(7)
   noisy = perturb_image(image, method="gaussian_noise", rng=retest_rng)
   shifted = perturb_image(noisy, method="translation", shift_fraction=0.5, rng=retest_rng)
   perturbed = perturb_image(shifted, method="rotation", angle_degrees=0.5, rng=retest_rng)
   feat_r1 = extract_voxel_texture(image, mask, kernel_radius=1, bin_width=12)
   feat_r3 = extract_voxel_texture(image, mask, kernel_radius=3, bin_width=12)

**2. Combine.** Build one :func:`~habit.precision_panel` per experiment,
aggregate if you have several subjects, then
:func:`~habit.identify_precise_features`.

**3. Publish the artefact.** ``precise.save("precise_features.json")``
writes the feature names plus the evidence panels. Another lab should
cluster the **same** names, not re-screen after seeing the endpoint.

**4. Whitelist into a habitat spec.** Place the whitelist **first** in
``voxel_feature_preprocessors`` so only precise columns reach scaling
and clustering::

   from habit import HabitatSpec, Spec
   from habit.recipes import Study

   whitelist = precise.preprocessor()
   spec = HabitatSpec(
       name="precise_habitats",
       voxel_feature_extractor=Spec("raw", {"modalities": ["LAP"]}),
       voxel_feature_preprocessors=(
           whitelist.spec,
           Spec("minmax", {"across_features": False}),
       ),
       habitat_model_fitter=Spec(
           "kmeans",
           {"min_habitats": 2, "max_habitats": 10, "validation": "elbow"},
       ),
       habitat_assigner=Spec("nearest_centroid"),
       random_seed=11,
   )
   result = Study(spec).fit_predict(cohort)

.. figure:: ../_static/images/examples/precise_features_icc_lcl.png
   :alt: ICC point estimates with 95 percent confidence intervals
   :width: 640

   Repeatability forest plot: ICC (point) and 95% CI (horizontal
   whisker) per voxel texture feature, with the 0.5 LCL threshold
   (:func:`~habit.viz.plot_precision_icc`, ``orientation="row"``).
   This gallery figure is **one cropped subject**. Kernel / bin ICC
   panels and the all-vs-precise habitat compare live on
   :doc:`../examples/precise_features`.

.. figure:: ../_static/images/examples/precise_features_all_vs_precise.png
   :alt: Habitat maps from all texture features versus the precise subset
   :width: 720

   Same subject, same extractor, same ``k`` search: all texture
   features vs precise features only
   (:func:`~habit.viz.plot_habitat_label_compare`,
   ``align_labels=True``).

Optional ROI-edge follow-up
---------------------------

The paper chain does not wrinkle the ROI. Subject-level
``bspline_deform`` (MONAI ``Rand3DElastic``) warps image and mask
together. That is **not** Prior Appendix S2 and **not** MIRP
``perturbation_roi_adapt_size``. Copy-ready code lives on
:doc:`../examples/precise_features` (the ``# BEGIN roi_followup``
block). Same crop and slice for the three map figures; the ICC panel
uses three cropped subjects.

.. figure:: ../_static/images/examples/precise_perturb_mask_edge.png
   :alt: Original and warped ROI contours on the same anatomy slice
   :width: 720

   Cyan solid = original ROI; vermillion dashed = warped ROI; yellow =
   membership change (XOR, right panel). Same axial index as the
   habitat compare below.

.. figure:: ../_static/images/examples/precise_habitat_stability_compare.png
   :alt: One-step habitats on the original ROI versus the warped ROI
   :width: 720

   One-step habitats on the original vs warped subject. Warped ids are
   remapped by mean-intensity Hungarian pairing
   (:func:`~habit.align_habitat_map`, ``method="centroid"``,
   ``force=True``) so the same colour is the same intensity-defined
   habitat on both panels.

.. figure:: ../_static/images/examples/precise_habitat_dice.png
   :alt: Per-habitat Dice after Hungarian matching
   :width: 480

   Per-habitat Dice from :func:`~habit.habitat_stability`
   (``method="centroid"``): ordinary
   :math:`2|A\cap B|/(|A|+|B|)` after the same mean-intensity pairing.

.. figure:: ../_static/images/examples/precise_habitat_feature_icc.png
   :alt: Habitat-table feature ICC point estimates with 95 percent confidence intervals
   :width: 720

   Habitat-table features (volume fraction, ITH, and the default
   graph family) scored with ICC(3A,1) **and the 95% CI**, not a
   point-only ICC. The gallery figure is a random subset of 24
   columns (``random_state=0``); the script still scores every
   shared column. Three demo subjects, original vs mean-aligned FFD
   map (:func:`~habit.icc3a_1`, :func:`~habit.viz.plot_precision_icc`).
   Colour is ``LCL >= 0.5`` only — this is **not** the voxel-texture
   PreciseFeatureSet. Wide intervals at ``n=3`` are expected.

Optional one-call recipe
------------------------

:func:`~habit.recipes.identify_precise_voxel_features` loops the same
atoms over a cohort and aggregates by the per-feature median. Use it
when you do not want to write the three ``precision_panel`` calls
yourself. It is not required to generate a perturbation or a texture
table.

.. code-block:: python

   from habit import cohort_from_directory
   from habit.recipes import identify_precise_voxel_features

   DATA = "demo_data/preprocessed"
   cohort = cohort_from_directory(DATA, modalities=("LAP",), roi="LAP")
   precise = identify_precise_voxel_features(cohort, seed=7)

What is not claimed
-------------------

* **Not a biomarker.** Passing the screen means the map is *repeatable /
  reproducible under the stated perturbations*, not that it encodes a
  cell type, a driver mutation, or a clinical outcome.
* **Not ROI grow/shrink.** MIRP ``perturbation_roi_adapt_size``
  (morphological dilation/erosion of the mask) is **not** in Prior 2024
  Appendix S2 and is **not** implemented. The optional
  ``bspline_deform`` follow-up warps the contour with the image; it is
  not grow/shrink and is not part of the default Precise chain.
* **Not leakage-proof.** Screening on the same cohort that will be
  clustered, then testing a classifier on that cohort, is still a
  discovery analysis. An external test set is a different protocol.
* **Not a substitute for a pre-specified** :math:`k`. Auto-selection (HABIT
  default: inertia **elbow**, :math:`k\in[2,10]`) remains a modelling
  choice. Precise features plus an unblinded :math:`k` search can still
  overfit.
* **Zeros vs missing.** Absent experiments (single radius, single bin
  width) are skipped, not failed. Read ``precise.to_frame()`` before
  claiming "all three experiments".

References
----------

Prior O, Macarro C, Navarro V, et al. Identification of Precise 3D CT
Radiomics for Habitat Computation by Machine Learning in Cancer.
*Radiol Artif Intell* 2024;6(2):e230118.
(`DOI <https://doi.org/10.1148/ryai.230118>`__).

See also
--------

* Gallery: :doc:`../examples/precise_features`
* Domain API: :doc:`../api/domain` (``perturb_image``,
  ``extract_voxel_texture``, ``identify_precise_features``)
* Kernels: :doc:`../api/kernels` (perturbation and voxel ICC)
* Clustering defaults: :doc:`../configuration/habitat`
