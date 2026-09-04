:orphan:

Feature extraction (CLI / YAML)
===============================

Bookmark for ``habit extract``. Need maps first (:doc:`segment_habitat`).
Python walk-through: :doc:`../examples/feature_extraction`. Graph gallery:
:doc:`../examples/graph_features`. Column definitions:
:doc:`../reference/features/index`. Habitat-vs-habitat contrast:
:doc:`../examples/habitat_feature_compare`.

Demo commands::

   habit check-config --config config/feature_extraction/config_extract_features_demo.yaml
   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

Default ``feature_types``: ``volume``, ``msi``, ``ith_score``,
``non_radiomics``, ``graph``. Heavy radiomics lines stay commented in that
YAML — uncomment when needed. Graph YAML knobs: :doc:`graph_features`.
Schema: :doc:`../configuration/feature_extraction`.

★ Edit ``raw_img_folder``, ``habitats_map_folder``, ``out_dir``. Then
``habit check-config`` + ``habit extract``.

Success: CSVs under ``out_dir``. Guide:
:doc:`../examples/feature_extraction`.
