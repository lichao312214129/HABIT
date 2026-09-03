:orphan:

Voxel texture feature maps
==========================

Goal: compute and display **voxel-level texture** maps — local neighbourhood
entropy and densified ``voxel_radiomics`` columns (e.g. GLCM) — as publication
2D slices.

``local_entropy`` is a built-in extractor (fast; no PyRadiomics).
``voxel_radiomics`` uses PyRadiomics / TorchRadiomics for per-voxel maps; keep
the enabled ``featureClass`` list small for interactive demos.

Walk-through (Guide): :doc:`../examples/voxel_texture`. This page keeps
runtime / backend notes that the gallery does not repeat.

Who builds the texture matrices
-------------------------------

A voxel texture class has two stages: **build the matrix** (GLCM, GLDM,
GLRLM, GLSZM, NGTDM), then **evaluate the feature formulas**. First-order
has no co-occurrence matrix: it gathers the kernel window and reduces it.
HABIT exposes three runtimes:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Runtime
     - Texture matrices
     - Feature formulas
   * - PyRadiomics (CPU)
     - ``radiomics.cMatrices`` C extension
     - NumPy
   * - C matrices + TorchRadiomics
     - Same C extension
     - PyTorch (GPU)
   * - HABIT CUDA
     - ``habit.kernels.radiomics.gpumatrices`` (GPU)
     - Same TorchRadiomics formulas (GPU)

Upstream `pytorchradiomics <https://github.com/lyhyl/pytorchradiomics>`_
is the middle row: every ``_calculateMatrix`` still calls
``cMatrices.calculate_glcm`` (and the GLDM / GLRLM / GLSZM / NGTDM
siblings) and then ``self.tensor(...)``. HABIT vendors that code under
``habit.kernels.radiomics.torchradiomics`` and adds GPU matrix builders
behind ``use_gpu_matrices`` (default ``"auto"``: on when the torch device
is CUDA). Their README GLCM 636 s → 23.8 s on a :math:`16^3` job is the
middle row vs PyRadiomics (CPU), **not** HABIT CUDA.

First-order in HABIT gathers the kernel window and computes the
reductions on CUDA. That is not a C texture matrix, so the middle column
does not apply.

Simulated three-way comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

End-to-end voxel-based ``execute()``, ``dtype=torch.float64``, **n = 5**
repeats on a fixed volume (mean ± sample standard deviation):

* Gaussian ``N(300, 80)``, shape ``(20, 48, 48)``, seed ``0``
* 4096 ROI voxels, ``kernelRadius=3``, ``binWidth=25``, ``voxelBatch=512``
* first-order: Energy, Entropy, Mean, 90Percentile, Uniformity;
  GLCM: JointEntropy, Idm, Contrast; GLDM: DependenceEntropy,
  LargeDependenceEmphasis, SmallDependenceEmphasis; GLRLM:
  ShortRunEmphasis, RunPercentage, GrayLevelNonUniformity; GLSZM:
  ZonePercentage, LargeAreaEmphasis, ZoneVariance; NGTDM: Coarseness,
  Contrast, Busyness
* hardware: NVIDIA GeForce RTX 3070 Laptop GPU

Column names:

* **PyRadiomics (CPU)** — C-extension matrices and NumPy formulas
  (first-order: CPU neighbourhood stats)
* **C matrices + TorchRadiomics** — C-extension matrices, GPU formulas
  (``use_gpu_matrices=false``)
* **HABIT CUDA** — GPU matrices and GPU formulas
  (``use_gpu_matrices=true``; first-order gather + formulas on CUDA)

**Time (s), mean ± s.d.**

.. list-table::
   :header-rows: 1
   :widths: 16 28 32 24

   * - Class
     - PyRadiomics (CPU)
     - C matrices + TorchRadiomics
     - HABIT CUDA
   * - First-order
     - 0.760 ± 0.030
     - n/a
     - 0.037 ± 0.023
   * - GLCM
     - 2.556 ± 0.042
     - 0.331 ± 0.033
     - 0.197 ± 0.087
   * - GLDM
     - 0.189 ± 0.007
     - 0.182 ± 0.014
     - 0.154 ± 0.040
   * - GLRLM
     - 1.567 ± 0.074
     - 1.060 ± 0.014
     - 0.530 ± 0.070
   * - GLSZM
     - 0.138 ± 0.004
     - 0.152 ± 0.008
     - 0.168 ± 0.006
   * - NGTDM
     - 0.175 ± 0.007
     - 0.150 ± 0.003
     - 0.131 ± 0.005

GLCM and GLRLM are the C-extension bottlenecks, so GPU formulas help and
GPU matrices help again. First-order has no texture matrix; HABIT CUDA
is the kernel gather plus reductions on device. GLSZM on this 4096-voxel
toy is already cheap on CPU (launch overhead).

**Accuracy**, max abs. error of the feature maps (worst feature in the
class), mean ± s.d. over the same 5 repeats:

.. list-table::
   :header-rows: 1
   :widths: 16 28 28 28

   * - Class
     - C+Torch vs PyRadiomics
     - HABIT CUDA vs PyRadiomics
     - HABIT CUDA vs C+Torch
   * - First-order
     - n/a
     - :math:`1.49\times 10^{-8} \pm 0`
     - n/a
   * - GLCM
     - :math:`1.42\times 10^{-14} \pm 0`
     - :math:`1.42\times 10^{-14} \pm 0`
     - :math:`7.77\times 10^{-17} \pm 3.0\times 10^{-17}`
   * - GLDM
     - :math:`1.78\times 10^{-15} \pm 0`
     - :math:`1.78\times 10^{-15} \pm 0`
     - 0
   * - GLRLM
     - :math:`3.55\times 10^{-15} \pm 0`
     - :math:`3.55\times 10^{-15} \pm 0`
     - 0
   * - GLSZM
     - :math:`8.88\times 10^{-16} \pm 0`
     - :math:`8.88\times 10^{-16} \pm 0`
     - 0
   * - NGTDM
     - :math:`5.55\times 10^{-16} \pm 0`
     - :math:`5.55\times 10^{-16} \pm 0`
     - :math:`1.67\times 10^{-16} \pm 0`

Texture C+Torch / HABIT CUDA vs PyRadiomics is NumPy vs torch formula
order, not a matrix mismatch. HABIT CUDA vs C+Torch sits at machine
epsilon because integer count matrices are bit-identical
(``tests/kernels/test_*_gpu_parity.py``, 81 cases). First-order is
looser (:math:`10^{-8}`) because percentiles use different quantile
algorithms. Force the matrix backend with ``use_gpu_matrices: true`` /
``false`` on the ``voxel_radiomics`` Spec.

Python API (sklearn-short)
--------------------------

The figure below is written by the voxel-texture gallery
(:doc:`../examples/voxel_texture`). Reproduce it::

   python docs/source/examples/scripts/voxel_texture_demo.py

Or paste the same load + plot the gallery shows::

   from habit.contracts import cohort_from_directory
   from habit.datasets import fetch_demo
   from habit.kernels import local_entropy_map
   from habit.viz import plot_voxel_texture_slice

   # Change DATA / MODALITY / ROI to your preprocessed layout
   DATA = fetch_demo()  # or "demo_data/preprocessed"
   MODALITY = "LAP"
   ROI = "LAP"
   subject = cohort_from_directory(DATA, modalities=(MODALITY,), roi=ROI)[0]
   image_vol = subject.image(MODALITY)
   mask_vol = subject.mask(ROI)
   entropy = local_entropy_map(image_vol.data, kernel_size=5, bins=32)
   fig = plot_voxel_texture_slice(
       entropy, anatomy=image_vol, roi_mask=mask_vol,
   )

For ``voxel_radiomics`` GLCM columns, create a
:class:`~habit.contracts.habitat.VoxelFeatureField` then pass it with
``feature=0`` (or a column name) — see the :doc:`../examples/voxel_texture`
script. Default ``mode="overlay"`` paints **opaque** feature colours inside
the ROI on greyscale anatomy (optional cyan contour). ``alpha<1`` is the
explicit translucent option. ``mode="side_by_side"`` adds a sibling anatomy
panel when you also want the raw image.

Layouts
-------

:func:`~habit.viz.plot_voxel_texture_slice` is **2D-slice only** (matplotlib
``[viz]`` extra):

* ``mode="overlay"`` — greyscale anatomy + opaque feature in ROI (default)
* ``mode="side_by_side"`` — anatomy + ROI contour | feature (sibling panel)
* ``mode="feature_only"`` — feature map alone

Omit ``axis`` on 3D volumes for three orthogonal panel rows. Panels use
``display_convention="radiological"`` (pass anatomy as an ``ImageVolume`` so
direction is not dropped). There is no built-in 3D volume renderer for
texture maps; use ITK-SNAP / 3D Slicer / napari for full volumetric browsing.

.. figure:: ../_static/images/examples/voxel_texture_overlay.png
   :alt: Opaque local-entropy overlay on anatomy
   :width: 520

   Default ``overlay`` layout
   (:func:`~habit.viz.plot_voxel_texture_slice`).

Also see
--------

* Examples gallery: :doc:`../examples/voxel_texture`
* Kernel: :func:`~habit.kernels.local_entropy_map`
* Habitat-map graph figures: :doc:`graph_features`
