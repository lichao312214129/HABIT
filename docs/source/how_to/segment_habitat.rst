:orphan:

Habitat segmentation (CLI / YAML)
=================================

Bookmark for ``habit get-habitat``. Python walk-throughs:
:doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`, :doc:`/auto_examples/00_introductory_tutorials/plot_02_inside_each_subject`,
:doc:`/auto_examples/00_introductory_tutorials/plot_03_pooled_voxels`. Atomic ``op(subject)``:
:doc:`/auto_examples/01_building_habitat_maps/plot_02_atomic_two_step`. Strategy table:
:doc:`/auto_examples/00_introductory_tutorials/index`.

Before you start: :doc:`before_you_start`. Data layout:
:doc:`/auto_examples/06_manipulating_images/index`. Demo tree::

   demo_data/preprocessed/

Demo commands (one physical line each)::

   habit check-config --config config/habitat/config_habitat_two_step.yaml
   habit get-habitat --config config/habitat/config_habitat_two_step.yaml
   habit view demo_data/preprocessed/images/subj001/LAP/WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd demo_data/results/habitat_two_step/subj001_habitats.nrrd

Other shipped YAMLs (same command)::

   habit get-habitat --config config/habitat/config_habitat_one_step_raw_concat_train.yaml
   habit get-habitat --config config/habitat/config_habitat_direct_pooling.yaml

★ fields when you copy a YAML: ``data_dir`` / ``data.source``, ``out_dir``,
and modality names in ``feature_construction``. Schema:
:doc:`../configuration/habitat`. ``Spec`` names:
:doc:`habitat_components`.

Success: ``*_habitats.nrrd`` under ``out_dir``. Next CLI step:
:doc:`extract_features`.
