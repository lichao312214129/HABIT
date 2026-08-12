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

   import habit.domain
   from habit import local_entropy_map
   from habit.domain import VoxelFeatureExtractorRegistry
   from habit.viz import dense_voxel_feature_map, plot_voxel_texture_slice, use_style

   entropy = local_entropy_map(image, kernel_size=5, bins=32)
   field = VoxelFeatureExtractorRegistry.create(
       "voxel_radiomics",
       modality="LAP",
       kernel_radius=1,
       params={
           "imageType": {"Original": {}},
           "featureClass": {"glcm": ["Contrast", "Correlation", "JointEntropy"]},
           "setting": {"binWidth": 25},
       },
   )(subject)
   contrast = dense_voxel_feature_map(
       field, next(n for n in field.feature_names if "Contrast" in n)
   )

   with use_style("radiology"):
       fig = plot_voxel_texture_slice(
           entropy,
           anatomy=image,
           roi_mask=mask,
           axis=0,
           mode="side_by_side",  # anatomy + ROI contour | texture
           feature_label="Local entropy (bits)",
       )

``mode="side_by_side"`` (default) draws the ROI as a **contour** on the anatomy
panel; the texture map is a separate panel (no alpha blend onto anatomy).
``mode="overlay"`` still exists for translucent texture-on-anatomy when needed.

Layouts
-------

:func:`~habit.viz.plot_voxel_texture_slice` is **2D-slice only** (matplotlib
``[viz]`` extra):

* ``mode="side_by_side"`` — anatomy + ROI contour | feature (recommended)
* ``mode="overlay"`` — translucent feature on greyscale anatomy
* ``mode="feature_only"`` — feature map alone

Omit ``axis`` on 3D volumes for three orthogonal panel rows. There is no
built-in 3D volume renderer for texture maps; use ITK-SNAP / 3D Slicer / napari
for full volumetric browsing.

Also see
--------

* Examples gallery: :doc:`../examples/voxel_texture`
* Kernel: :func:`~habit.kernels.local_entropy_map`
* Habitat-map graph figures: :doc:`graph_features`
