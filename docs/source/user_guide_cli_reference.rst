CLI Reference
=============

Habitat Analysis Command
------------------------

.. code-block:: bash

    habit habitat --config <config_file>

Options:

* ``--config``: Configuration YAML file
* ``--debug``: Enable debug mode

Machine Learning Command
------------------------

.. code-block:: bash

    habit ml --config <config_file> --mode train

Options:

* ``--config``: Configuration YAML file
* ``--mode``: Operation mode (train/predict)
* ``--model``: Path to model file (for prediction)

Model Comparison Command
------------------------

.. code-block:: bash

    habit compare --config <config_file>

Options:

* ``--config``: Configuration YAML file

Image Preprocessing Command
---------------------------

.. code-block:: bash

    habit preprocess --config <config_file>

Options:

* ``--config``: Configuration YAML file

Feature Extraction Command
--------------------------

.. code-block:: bash

    habit extract-features --config <config_file>

Options:

* ``--config``: Configuration YAML file

ICC Analysis Command
--------------------

.. code-block:: bash

    habit icc --config <config_file>

Options:

* ``--config``: Configuration YAML file
