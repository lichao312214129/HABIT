Demo tutorial
=============

End-to-end pipeline: preprocessing → habitat segmentation → feature extraction → machine learning → model comparison.

Prerequisites: :doc:`installation` .

Prepare data
------------

.. note::

   ``D:\habit-cpu`` is an example path — use your portable or project root.

1. Download and extract to the project root (same level as ``python.exe`` or repo root):

   ``demo_data.rar`` (required)

   - `Download demo_data.rar <|demo_data_link|>`_
   - Code: ``|demo_data_code|``

   ``config.rar`` (portable users; source users already have ``config/`` )

   - `Download config.rar <|config_pack_link|>`_
   - Code: ``|config_pack_code|``

2. Verify ``habit --version`` .

Run (5 steps)
-------------

Demo includes preprocessed data — **start at step 2** on first run.

.. code-block:: bash

   cd /d D:\habit-cpu

   habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml
   habit get-habitat --config config/habitat/config_habitat_two_step.yaml
   habit extract --config config/feature_extraction/config_extract_features_demo.yaml
   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train
   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

Outputs under ``demo_data/results/`` . Your own data → :doc:`../how_to/index` or :doc:`../gui/index` .
