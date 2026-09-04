:mod:`habit.voxel_features`: per-voxel descriptors
==================================================

.. automodule:: habit.voxel_features
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.voxel_features

**User guide:** Habitat Guide :doc:`../auto_examples/02_voxel/plot_01_feature_routes`
· :doc:`../auto_examples/02_voxel/plot_03_voxel_texture` ·
:doc:`domain_habitat`. Component names:
:doc:`../how_to/habitat_components`.

``raw`` / ``local_entropy`` / ``voxel_radiomics`` describe each ROI voxel;
``concat`` / ``expression`` / ``kinetic`` compose those families.
:func:`~habit.voxel_features.extract_voxel_texture` is the same
``voxel_radiomics`` pass on one ``ImageVolume`` + mask (no ``Subject``).

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   VoxelFeatureExtractor
   RawVoxelFeatures
   LocalEntropyVoxelFeatures
   VoxelRadiomicsFeatures
   ConcatVoxelFeatures
   ExpressionVoxelFeatures
   KineticVoxelFeatures
   VoxelFeatureTree
   VoxelFeatureExtractorRegistry

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   extract_voxel_texture
   build_voxel_extractor
   build_voxel_field
   aligned_image
   roi_voxels
   load_cached_voxel_field
   save_cached_voxel_field
   voxel_radiomics_cache_key
   voxel_volume_fingerprint
