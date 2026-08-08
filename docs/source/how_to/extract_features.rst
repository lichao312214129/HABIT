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

Your data
---------

★ Edit ``raw_img_folder``, ``habitats_map_folder``, ``out_dir``. Then
``habit check-config`` + ``habit extract``.

Success: CSVs under ``out_dir``.

Next: :doc:`train_model`.
