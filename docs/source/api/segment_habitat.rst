Habitat segmentation
====================

.. code-block:: bash

   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

Other strategies (swap config file):

- One-step: ``config/habitat/config_habitat_one_step_raw_concat_train.yaml``
- Direct pooling: ``config/habitat/config_habitat_direct_pooling.yaml``

Options: ``--mode train|predict`` , ``--resume`` .

**Output**: ``*_habitats.nrrd`` ; overlay in ITK-SNAP / 3D Slicer.

**Strategy choice**: :doc:`../explanation/concepts`

**Configuration**: :doc:`../configuration/habitat`
