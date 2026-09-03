Voxel texture and GPU
=====================

Texture maps are **inputs to clustering**, not post-label tables.
GPU is a faster implementation of the same IBSI / PyRadiomics definition —
the numbers do not change because of GPU.

Compute one map
---------------

:func:`~habit.kernels.local_entropy_map` plus
:func:`~habit.viz.plot_voxel_texture_slice`. Pass ``ImageVolume`` /
``MaskVolume`` to the plotter (not ``.data``) so direction and spacing
stay attached.

.. literalinclude:: scripts/voxel_texture_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

The first ``plot_voxel_texture_slice(entropy, anatomy=image_vol, roi_mask=mask_vol)``
writes the figure below (the script also saves sibling panels).

.. figure:: ../_static/images/examples/voxel_texture_overlay.png
   :alt: Opaque local-entropy overlay on greyscale anatomy
   :width: 720

   Default overlay: grey anatomy outside the ROI, opaque entropy colours
   inside, cyan ROI contour.

GPU radiomics
-------------

Install CUDA torch, then the optional extra, then confirm the device::

   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install "habitat-analysis[torch]"
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Same extractor as the script. Use CUDA when ``torch.cuda.is_available()``
is true; otherwise this block uses CPU so it still runs::

   import torch

   from habit.contracts import cohort_from_directory
   from habit.datasets import fetch_demo
   from habit.voxel_features import VoxelFeatureExtractorRegistry

   DATA = fetch_demo()
   MODALITY = "LAP"
   subject = cohort_from_directory(DATA, modalities=(MODALITY,), roi=MODALITY)[0]
   device = "cuda" if torch.cuda.is_available() else "cpu"
   glcm = VoxelFeatureExtractorRegistry.create(
       "voxel_radiomics",
       modality=MODALITY,
       kernel_radius=1,
       torch_device=device,
       use_gpu_matrices="auto",
       params={
           "imageType": {"Original": {}},
           "featureClass": {"glcm": ["Contrast"]},
           "setting": {"binWidth": 25},
       },
   )(subject)
   print(device, getattr(glcm, "shape", type(glcm)))

Speedup (RTX 3070 Laptop, 4096-voxel ROI)
-----------------------------------------

End-to-end vs PyRadiomics CPU (same definition):

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Class
     - PyRadiomics CPU (s)
     - HABIT CUDA (s)
     - Speedup
   * - First-order
     - 0.760
     - 0.037
     - ~21×
   * - GLCM
     - 2.556
     - 0.197
     - ~13×
   * - GLRLM
     - 1.567
     - 0.530
     - ~3×
   * - GLDM / NGTDM
     - 0.189 / 0.175
     - 0.154 / 0.131
     - ~1.2×
   * - GLSZM
     - 0.138
     - 0.168
     - no win on this toy

Accuracy: integer count matrices are bit-identical; texture formulas sit at
about :math:`10^{-14}` vs PyRadiomics; first-order is about
:math:`10^{-8}` (different quantile algorithms).

IBSI
----

Definition check uses the IBSI-1 Phase 1 digital phantom,
**3-D averaged** texture. Reference table:
:doc:`../reference/features/traditional`.

**Next:** :doc:`habitat_preprocessing`
