Habitat segmentation
====================

Goal: get ``*_habitats.nrrd`` maps, then view them.

Before you start
----------------

* :doc:`before_you_start` (terminal at project root)
* Data ready (:doc:`prepare_data`). Demo::

     demo_data/preprocessed/

Run the demo
------------

::

   habit check-config --config config/habitat/config_habitat_two_step.yaml
   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

   habit view demo_data/preprocessed/images/subj001/LAP/WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd demo_data/results/habitat_two_step/subj001_habitats.nrrd

For fuller 3D inspection, load the source image and ``*_habitats.nrrd``
together in **ITK-SNAP**, **3D Slicer**, or a **SimpleITK**-based viewer
(label overlay / segmentation).

Other strategies (same command, different YAML)::

   habit get-habitat --config config/habitat/config_habitat_one_step_raw_concat_train.yaml
   habit get-habitat --config config/habitat/config_habitat_direct_pooling.yaml

Use your own data
-----------------

Copy ``config/habitat/config_habitat_two_step.yaml`` (or
``config_habitat_two_step_minimal.yaml``) and edit:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - ★ Field
     - What to put
   * - ``data_dir`` / ``data.source``
     - Path-list YAML or folder root (:doc:`prepare_data`)
   * - ``out_dir``
     - Output folder
   * - modality names in ``feature_construction``
     - Must match your data keys (``T1`` / ``T2`` / …)

Then::

   habit check-config --config path/to/your_habitat.yaml
   habit get-habitat --config path/to/your_habitat.yaml
   habit view path/to/image.nii.gz path/to/subj001_habitats.nrrd

Success: ``*_habitats.nrrd`` under ``out_dir``.

Next: :doc:`extract_features`. Config details: :doc:`../configuration/habitat`.
