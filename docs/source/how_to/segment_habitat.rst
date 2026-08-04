Habitat segmentation
====================

Python API
----------

For in-memory runs without the CLI, see the **Common workflows** section in
:doc:`../api/python_api` (synthetic cohort, three habitat designs, YAML
translation, and ``StudyResult.save``). The CLI examples below use the same
recipes internally.

CLI
---

.. code-block:: bash

   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

Other strategies (swap config file):

- One-step: ``config/habitat/config_habitat_one_step_raw_concat_train.yaml``
- Direct pooling: ``config/habitat/config_habitat_direct_pooling.yaml``

Options: ``--mode train|predict`` , ``--pipeline`` (predict: override saved pipeline path) , ``--debug`` , ``--resume`` .

**Output**: ``*_habitats.nrrd`` ; overlay in ITK-SNAP / 3D Slicer.

**Strategy choice**: :doc:`../explanation/concepts`

**Configuration**: :doc:`../configuration/habitat`
