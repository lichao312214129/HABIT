Voxel texture feature maps
==========================

Goal: compute and display **voxel-level texture** maps (local neighbourhood
entropy, or densified ``voxel_radiomics`` / custom ``VoxelFeatureField``
columns) as publication 2D slices.

``local_entropy`` is a built-in voxel feature extractor (fast; no
PyRadiomics). Heavy per-voxel radiomics maps use the same plotter once you
have a dense volume or a :class:`~habit.contracts.habitat.VoxelFeatureField`.

Runnable gallery (demo_data figures):
:doc:`../examples/voxel_texture`.

Python API
----------

Kernel path (arrays in, same-shaped map out)::

   from habit import local_entropy_map
   from habit.viz import plot_voxel_texture_slice, use_style

   entropy = local_entropy_map(image, kernel_size=5, bins=32)
   with use_style("radiology"):
       fig = plot_voxel_texture_slice(
           entropy,
           anatomy=image,
           roi_mask=mask,
           axis=0,
           mode="side_by_side",
           feature_label="Local entropy (bits)",
       )
   # caller owns fig.savefig(...)

Domain path (``Subject`` → ``VoxelFeatureField`` → dense map)::

   import habit.domain  # registers built-ins
   from habit.domain import VoxelFeatureExtractorRegistry
   from habit.viz import dense_voxel_feature_map, plot_voxel_texture_slice

   fx = VoxelFeatureExtractorRegistry.create(
       "local_entropy",
       modality="LAP",
       kernel_size=5,
       bins=32,
   )
   field = fx(subject)
   dense = dense_voxel_feature_map(field, "local_entropy-LAP")
   fig = plot_voxel_texture_slice(
       field,  # or dense
       anatomy=subject.image("LAP").data,
       feature="local_entropy-LAP",
       mode="overlay",
   )

``voxel_radiomics`` (and any other extractor that returns a
``VoxelFeatureField``) uses the same densify + plot path — pick the feature
column by name or index.

Layouts
-------

:func:`~habit.viz.plot_voxel_texture_slice` is **2D-slice only** (matplotlib
``[viz]`` extra):

* ``mode="side_by_side"`` — anatomy | feature (default)
* ``mode="overlay"`` — translucent feature on greyscale anatomy
* ``mode="feature_only"`` — feature map alone

For 3D volumes, omit ``axis`` to get three orthogonal panels through the
densest ROI slice. There is no built-in 3D volume renderer for texture maps;
use ITK-SNAP / 3D Slicer / napari if you need full volumetric browsing.

Also see
--------

* Examples gallery: :doc:`../examples/voxel_texture`
* Kernel: :func:`~habit.kernels.local_entropy_map`
* Habitat graph figures (different product): :doc:`graph_features`
