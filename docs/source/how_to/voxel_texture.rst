Voxel texture feature maps
==========================

Goal: compute and display **voxel-level texture** maps — local neighbourhood
entropy and densified ``voxel_radiomics`` columns (e.g. GLCM) — as publication
2D slices.

``local_entropy`` is a built-in extractor (fast; no PyRadiomics).
``voxel_radiomics`` uses PyRadiomics / TorchRadiomics for per-voxel maps; keep
the enabled ``featureClass`` list small for interactive demos.

Runnable gallery: :doc:`../examples/voxel_texture`.

Python API (sklearn-short)
--------------------------

::

   from habit import local_entropy_map
   from habit.viz import plot_voxel_texture_slice

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
