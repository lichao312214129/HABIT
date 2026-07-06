CLI Commands API
================

Command Line Interface
----------------------

HABIT provides a command-line interface through the ``habit`` command.

Main Commands
-------------

.. code-block:: bash

   habit preprocess       # Preprocess images
   habit sort-dicom       # DICOM rename/sort only (dcm2niix -r y)
   habit get-habitat      # Habitat segmentation (train / predict)
   habit extract          # Extract habitat features
   habit model            # Train or predict ML models (--mode train|predict)
   habit cv               # K-fold cross-validation
   habit compare          # Model comparison plots and DeLong test
   habit radiomics        # Traditional radiomics (standalone)
   habit icc              # ICC analysis
   habit retest           # Test-retest reproducibility
   habit dice             # Dice between two mask batches
   habit dicom-info       # Inspect DICOM metadata
   habit merge-csv        # Merge CSV/Excel files by index column
   habit gui              # Web GUI (under development)

For detailed CLI documentation, see the command help:

.. code-block:: bash

   habit --help
   habit get-habitat --help

See also :doc:`../reference/cli`.
