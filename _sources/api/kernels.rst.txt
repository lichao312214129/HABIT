Numeric kernels (``habit.kernels``)
===================================

Pure NumPy / SciPy functions. No ``Subject``, no YAML, no IO.

.. code-block:: python

   from habit.kernels import (
       delong_roc_ci,
       delong_roc_test,
       delong_roc_variance,
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
and resample back onto the original grid:

.. code-block:: python

   from habit.kernels import (
       add_gaussian_noise,
       estimate_noise_sigma,
       rotate_image,
       translate_image,
   )

   sigma = estimate_noise_sigma(array, method="chang")  # wavelet estimator
   noisy = add_gaussian_noise(array, sigma, rng)        # zero-mean Gaussian
   shifted = translate_image(image, shift_voxels=(0.3, -0.2, 0.0))
   rotated = rotate_image(image, angle_degrees=0.5, axis="z")

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
