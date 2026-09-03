Traditional radiomics
=====================

.. note::

   Whole-ROI radiomics without habitat maps. Guide:
   :doc:`../examples/features_radiomics_api`. Features **on** habitat
   maps: :doc:`../examples/feature_extraction`.

Goal: whole-ROI PyRadiomics **without** habitat maps. For habitat features use
:doc:`extract_features`. ROI-level radiomics, voxel-level radiomics, and
3-D shape match PyRadiomics ``execute()``; the digital-phantom table and
alignment notes are on :doc:`../reference/features/traditional`.

Run the demo
------------

::

   habit check-config --config config/radiomics/config_traditional_radiomics.yaml
   habit radiomics --config config/radiomics/config_traditional_radiomics.yaml

Your data
---------

★ Edit ``paths.images_folder`` (folder with ``images/`` + ``masks/``),
``paths.out_dir``, and ``processing.process_image_types`` (modality names).

Success: feature tables under ``paths.out_dir``.

Habitat-wise tables (when you do have maps) overlay the same anatomy.
The figure is **not** from ``habit radiomics`` above (whole-ROI has no
labels). It is written by the feature-extraction gallery
(:doc:`../examples/feature_extraction`). Reproduce it::

   python docs/source/examples/scripts/feature_extraction_demo.py

The plot call in that script (``ROI = "LAP"``)::

   from habit.viz import plot_habitat_overlay

   fig = plot_habitat_overlay(subject.image(ROI), habitat_map, title="habitats")

.. figure:: ../_static/images/examples/feature_extract_overlay.png
   :alt: Habitat overlay used before feature tables
   :width: 420

   Same file the gallery script writes to ``out/feature_extract_overlay.png``.
