How-to guides
=============

YAML + CLI operator path. Pattern on every page:

1. Copy a shipped config under ``config/``
2. Edit only ★ fields (usually ``data_dir`` / ``out_dir``)
3. ``habit check-config -c …`` then the run command

Start: :doc:`before_you_start` → :doc:`prepare_data` → pipeline pages below.

.. toctree::
   :maxdepth: 1

   before_you_start
   prepare_data
   preprocess
   segment_habitat
   extract_features
   radiomics
   train_model
   compare_models
   auxiliary_tools
