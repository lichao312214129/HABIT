Habitat feature routes (raw, concat, radiomics, SLIC)
=====================================================

Before preprocessing chains run, voxels must be **described**. v1 selects
the route through ``HabitatSpec.voxel_feature_extractor`` (and optionally
``supervoxel_feature_extractor``):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Route
     - Role
   * - ``raw``
     - Concatenate modality intensities inside the ROI (fastest; synthetic demos)
   * - ``concat``
     - Join heterogeneous families side-by-side (e.g. ``raw(T1)`` + ``raw(T2)``)
   * - ``expression``
     - Restricted arithmetic over modalities (ratios, powers, ``square`` / ``log``);
       see :doc:`custom_voxel_features`
   * - ``voxel_radiomics``
     - Per-voxel PyRadiomics texture (needs ``demo_data/`` + PyRadiomics)
   * - ``supervoxel_radiomics``
     - Texture of each supervoxel region (two-step; ``supervoxel_feature_extractor``)
   * - ``slic`` (supervoxelizer)
     - Spatially coherent supervoxels instead of k-means over features

Every route supports **batch** (``recipes.two_step(cohort, spec)``) and
**atomic** (``SubjectPipeline.units(subject)``) inspection.

Script
------

.. literalinclude:: scripts/habitat_feature_routes_demo.py
   :language: python

Output (abbreviated)
--------------------

::

   === raw(modalities) ===
     atomic n_features: 3
     batch: 2 maps, 3 habitats

   === concat(raw, raw) per modality ===
     atomic n_features: 2
     batch: 2 maps

   === supervoxelizer: slic ===
     batch: 2 maps, 3 habitats

   === voxel_radiomics (demo_data, may take ~30s) ===
     atomic n_features: 21
     batch (1 subject): 3 habitats

What to read next
-----------------

* :doc:`habitat_preprocessing` — winsorize / zscore / binning chains
* :doc:`two_step_habitat` — end-to-end two-step workflow
* ``config/habitat/config_habitat_two_step_voxel_radiomics_*.yaml`` — YAML twins
