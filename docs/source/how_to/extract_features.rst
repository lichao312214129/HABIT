Feature extraction
==================

Goal: habitat-level feature CSVs from images + ``*_habitats.nrrd``.

Need habitat maps first (:doc:`segment_habitat`).

Run the demo
------------

::

   habit check-config --config config/feature_extraction/config_extract_features_demo.yaml
   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

Default ``feature_types``: light families (``volume``, ``msi``, ``ith_score``,
``non_radiomics``). Heavy radiomics lines are commented in that YAML — uncomment
when needed.

**Also extract graph topology:** add ``graph`` to ``feature_types`` (built-in
light family, peer to ``volume`` / ``msi`` / ``ith_score``). Dedicated how-to:
:doc:`graph_features`. Gallery with demo_data figures:
:doc:`../examples/graph_features`. Column reference:
:doc:`../reference/features/graph`.

Your data
---------

★ Edit ``raw_img_folder``, ``habitats_map_folder``, ``out_dir``. Then
``habit check-config`` + ``habit extract``.

Success: CSVs under ``out_dir``.

Volume / MSI / ITH from the same maps (Python)::

   from habit.viz import plot_habitat_volume_fractions

   fig = plot_habitat_volume_fractions(fractions)

.. figure:: ../_static/images/examples/feature_extract_volume_fractions.png
   :alt: Habitat volume fractions
   :width: 420

   Volume fractions from the feature-extraction gallery
   (:func:`~habit.viz.plot_habitat_volume_fractions`).

Next: :doc:`graph_features` (optional topology) or :doc:`train_model`.
