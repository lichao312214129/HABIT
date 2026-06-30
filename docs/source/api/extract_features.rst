Feature extraction
==================

.. code-block:: bash

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

**Input**: ``raw_img_folder`` + ``habitats_map_folder`` (``*_habitats.nrrd`` )

**Output**: CSV files under ``out_dir`` (controlled by ``feature_types`` ).

**Formulas**: :doc:`../reference/features/index`

**Configuration**: :doc:`../configuration/feature_extraction`
