After: habitat features for statistics / ML
===========================================

Habitat analysis ends with a **feature table**: one row per patient,
columns such as habitat volume fractions, MSI, ITH score and graph
metrics. That table is the hand-off to your statistics or machine
learning tool; HABIT does not need to own the model.

From Python, the table is ``result.features.frame`` (a pandas
``DataFrame``); write it with ``to_csv`` and join it with your outcome
table by ``subject``. The full train / save / held-out workflow that
ends in that CSV is
:doc:`/auto_examples/05_validation_and_reuse/plot_01_train_save_predict`.
Column definitions: :doc:`../reference/features/index`. HABIT's own
tabular tools are a bookmark only: :doc:`../api/table_ml`.

From the CLI, ``habit get-habitat`` already writes
``habitat_features.csv`` next to the habitat maps. ``habit extract``
recomputes features from saved maps (need maps first:
:doc:`segment_habitat`)::

   habit check-config --config config/feature_extraction/config_extract_features_demo.yaml
   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

Default ``feature_types``: ``volume``, ``msi``, ``ith_score``,
``non_radiomics``, ``graph``. Heavy radiomics lines stay commented in that
YAML — uncomment when needed. Graph YAML knobs: :doc:`graph_features`.
Schema: :doc:`../configuration/feature_extraction`.

★ Edit ``raw_img_folder``, ``habitats_map_folder``, ``out_dir``. Then
``habit check-config`` + ``habit extract``.

Success: CSVs under ``out_dir``. Python walk-through of each feature
family: :doc:`/auto_examples/04_quantifying_habitats/index`.
