DIY voxel features (expression + custom plugin)
===============================================

When ``raw`` / ``concat`` are not enough — for example
``square(LAP / PVP^3)``, or a neighbourhood / embedding feature — v1 offers
two complementary routes:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Route
     - When to use it
   * - ``expression``
     - Restricted arithmetic over modality intensities (ratios, powers,
       ``square`` / ``log`` / ...). Safe AST evaluation; no arbitrary Python.
   * - Custom plugin
     - Any :class:`~habit._protocols.VoxelFeatureExtractor`. Register
       with ``@VoxelFeatureExtractorRegistry.register("name")`` in-process,
       or via the ``habit.voxel_feature_extractor`` entry-point group in a
       third-party package (then ``load_plugins()``).

Both routes plug into ``HabitatSpec.voxel_feature_extractor`` and run through
the same recipes (``two_step`` / ``one_step`` / ``direct_pooling``).

Script
------

.. literalinclude:: scripts/custom_voxel_feature_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Draw the figures
----------------

Paste this after the Script block (it uses ``subject`` and
``custom_result``). Writes ``out/custom_voxel_overlay.png``.

.. literalinclude:: scripts/custom_voxel_feature_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

The script writes ``out/custom_voxel_overlay.png``.

Output
------

::

   === expression: square(LAP / (PVP^3 + eps)) ===
     atomic features: ['lap_over_pvp_sq']
   === custom plugin: t1_t2_contrast ===
     batch: 2 maps, 3 habitats
   Wrote out/custom_voxel_overlay.png

Figures
-------

Custom extractors still produce ordinary habitat maps.

.. figure:: ../_static/images/examples/custom_voxel_overlay.png
   :alt: Habitat overlay after a custom voxel feature extractor
   :width: 420

   Habitats after ``Study(...).fit_predict`` with a DIY extractor
   (:func:`~habit.viz.plot_habitat_overlay`).

Trees and combiners
-------------------

The same node abstraction (leaf vs combiner) is worked as a tree on
:doc:`feature_composition` (same ``Spec`` fingerprint in Python and YAML).

See also
--------

* :doc:`habitat_feature_routes` — ``raw`` / ``concat`` intensities
* :doc:`voxel_texture` — local entropy / voxel radiomics
* :doc:`../customization/index` — v1 registry + entry-point DIY guide
* :doc:`../api/plugins` — ``list_plugins("voxel_feature_extractor")``
