Auxiliary tools
===============

Standalone utilities. Full flag lists: ``habit <command> --help`` and
:doc:`../reference/auxiliary`.

ICC (feature reproducibility)
-----------------------------

.. code-block:: bash

   habit icc --config config/auxiliary/config_icc_demo.yaml

**Configuration**: :doc:`../configuration/auxiliary`.

Test-retest habitat mapping
---------------------------

.. code-block:: bash

   habit retest --config config/auxiliary/config_test_retest.yaml

**Configuration**: :doc:`../configuration/auxiliary`.

DICOM metadata export
---------------------

.. code-block:: bash

   habit dicom-info -i ./demo_data/dicom -o info.csv

No YAML config; all options are CLI flags.

Merge tables
------------

.. code-block:: bash

   habit merge-csv file1.csv file2.csv -o merged.csv --index-col PatientID

Dice between mask batches
-------------------------

.. code-block:: bash

   habit dice --input1 dir1 --input2 dir2 --output dice_results.csv
