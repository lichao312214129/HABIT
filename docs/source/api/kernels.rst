Numeric kernels (``habit.kernels``)
===================================

Pure NumPy / SciPy functions. No ``Subject``, no YAML, no IO.

.. code-block:: python

   from habit.kernels import (
       HabitatGraphFeatureOptions,
       delong_roc_ci,
       delong_roc_test,
       delong_roc_variance,
       extract_graph_features,
       fast_delong,
       habitat_region_stats,
       habitat_volume_fractions,
       hosmer_lemeshow_test,
       icc2_1,
       icc3_1,
       icc3a_1,
       icc3c_1,
       ith_score,
       msi_features_from_matrix,
       spatial_interaction_matrix,
       spiegelhalter_z_test,
       two_way_mean_squares,
       compute_midrank,
   )

Habitat metrics
---------------

.. code-block:: python

   import numpy as np

   labels = np.zeros((16, 32, 32), dtype=np.int32)
   labels[4:12, 8:24, 8:24] = 1
   labels[6:10, 12:20, 12:20] = 2

   matrix = spatial_interaction_matrix(labels, n_classes=3)
   msi = msi_features_from_matrix(matrix)          # dict[str, float]
   ith = ith_score(labels)                         # float
   fractions = habitat_volume_fractions(labels, habitat_ids=(1, 2))
   stats = habitat_region_stats(labels)            # id -> (voxels, components)

Graph topology kernels
----------------------

Region graphs + NetworkX metrics (same definitions as the built-in
``graph`` habitat feature family). Arrays in, ``dict`` out — no YAML / IO.

.. code-block:: python

   from habit import HabitatGraphFeatureOptions, extract_graph_features

   options = HabitatGraphFeatureOptions(
       edge_method="adjacency",
       adjacency_connectivity="face",
       adjacency_min_voxels=10,
       erosion_radius=1,
       subdivide_region_voxels=1000,
   )
   graph_feats = extract_graph_features(
       labels,
       options=options,
       expected_labels=(1, 2),
   )
   # Keys look like single_h1_n_nodes, pair_h1_h2_modularity, ...

Also exported: :func:`~habit.extract_graph_features_for_labels`,
:func:`~habit.extract_habitat_nodes`,
:func:`~habit.build_centroid_distance_graph`,
:func:`~habit.build_adjacency_graph`, :func:`~habit.pair_count`.
See :doc:`../reference/features/graph`.

ICC kernels
-----------

.. code-block:: python

   # n_targets x k_raters design matrices of mean squares helpers
   ms = two_way_mean_squares(n_targets, k_raters)
   icc_agreement = icc2_1(n_targets, k_raters)
   icc_consistency = icc3_1(n_targets, k_raters)

Voxel-level reliability (the precision screen's statistics) returns point
estimates with confidence limits; negative values truncate at 0:

.. code-block:: python

   # matrix: n_voxels x n_conditions, one column per condition
   est = icc3a_1(matrix)   # absolute agreement -> ICCEstimate(value, lcl, ucl)
   est = icc3c_1(matrix)   # consistency

Image perturbation
------------------

Simulated-retest kernels behind the ``image_perturbation`` domain. Noise
estimation and addition work on plain arrays; the geometric kernels take
and return ``sitk.Image`` so spacing, origin and direction are honoured,
and resample back onto the original grid. The default recipe chain matches
Prior et al. (*Radiol Artif Intell* 2024;6(2):e230118, Appendix S2 / MIRP
1.2.0): Chang-estimated Gaussian noise, a 0.5-voxel translation fraction,
and a 0.5° in-plane rotation. :func:`~habit.kernels.rigid_transform_image`
composes translation+rotation into one affine (MIRP ≥ 2).

.. code-block:: python

   from habit.kernels import (
       add_gaussian_noise,
       estimate_noise_sigma,
       rigid_transform_image,
       rotate_image,
       translate_image,
   )

   sigma = estimate_noise_sigma(array, method="chang")  # wavelet estimator
   noisy = add_gaussian_noise(array, sigma, rng)        # zero-mean Gaussian
   shifted = translate_image(image, shift_voxels=(0.3, -0.2, 0.0))
   rotated = rotate_image(image, angle_degrees=0.5, axis="z")
   # MIRP ≥ 2: translation + rotation in one resample
   rigid = rigid_transform_image(image, (0.5, 0.5, 0.5), angle_degrees=0.5)

Classification statistics
-------------------------

.. code-block:: python

   midranks = compute_midrank(scores)
   auc, var = fast_delong(predictions_sorted)
   result = delong_roc_test(y_true, scores_a, scores_b)
   ci = delong_roc_ci(y_true, scores)
   var_ab = delong_roc_variance(y_true, scores_a, scores_b)
   hl = hosmer_lemeshow_test(y_true, scores, n_groups=10)
   sp = spiegelhalter_z_test(y_true, scores)

Stability
---------

Published metrics (MSI, ITH, ICC, DeLong, Hosmer–Lemeshow, Spiegelhalter) are a
**stable** subset within v1.x.
