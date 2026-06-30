Auxiliary commands
==================

Standalone tools. Run ``habit <command> --help`` for all flags.

habit dicom-info
----------------

Export DICOM metadata to CSV / Excel.

.. code-block:: bash

   habit dicom-info -i ./demo_data/dicom -o info.csv

habit icc
---------

Intraclass correlation (ICC) for feature reproducibility.

.. code-block:: bash

   habit icc --config config/auxiliary/config_icc_demo.yaml

See :doc:`../configuration/auxiliary` .

habit retest
------------

Test-retest habitat mapping analysis.

.. code-block:: bash

   habit retest --config config/auxiliary/config_test_retest.yaml

habit merge-csv
---------------

Merge CSV / Excel files on an index column.

.. code-block:: bash

   habit merge-csv file1.csv file2.csv -o merged.csv --index-col PatientID

habit dice
----------

Dice coefficient between two mask sets.

.. code-block:: bash

   habit dice --input1 dir1 --input2 dir2 --output dice_results.csv
