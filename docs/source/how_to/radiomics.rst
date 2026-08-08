Traditional radiomics
=====================

Goal: whole-ROI PyRadiomics **without** habitat maps. For habitat features use
:doc:`extract_features`.

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
