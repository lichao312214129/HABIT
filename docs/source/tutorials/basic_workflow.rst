Basic Workflow Tutorial
=======================

This tutorial walks you through a complete habitat analysis workflow.

Step 1: Prepare Data
---------------------

Organize your medical images and masks in a structured directory:

.. code-block:: text

   data/
   ├── subject1/
   │   ├── image1.nrrd
   │   ├── image2.nrrd
   │   └── mask.nrrd
   └── subject2/
       ├── image1.nrrd
       ├── image2.nrrd
       └── mask.nrrd

Step 2: Create Configuration
------------------------------

Create ``config.yaml`` with your analysis settings.

Step 3: Run Training
---------------------

.. code-block:: bash

   habit get-habitat --config config.yaml --mode train

Step 4: Run Prediction
----------------------

.. code-block:: bash

   habit get-habitat --config config.yaml --mode predict --pipeline results/habitat_pipeline.pkl

Step 5: Review Results
----------------------

Check the output directory for:
- ``habitats.csv``: Results with habitat labels
- ``*_habitats.nrrd``: Habitat maps
- ``habitat_pipeline.pkl``: Trained pipeline
