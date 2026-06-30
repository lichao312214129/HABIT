Preprocessing
=============

.. code-block:: bash

   habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml

DICOM sort only (no NIfTI conversion):

.. code-block:: bash

   habit sort-dicom --config config/dicom_sort/config_sort_dicom.yaml

**Output**: ``out_dir/processed_images/`` ; log ``<out_dir>/processing.log`` .

**Configuration**: :doc:`../configuration/preprocessing`
