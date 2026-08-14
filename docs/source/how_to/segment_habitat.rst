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

The figure below is **not** from the CLI YAML above (that config is the
full demo Spec). It is written by the two-step gallery
(:doc:`../examples/two_step_habitat`). Reproduce it::

   python docs/source/examples/scripts/two_step_habitat_quickstart.py

The plot call in that script (``ROI = "LAP"``)::

   from habit.viz import plot_habitat_overlay

   fig = plot_habitat_overlay(subject.image(ROI), habitat_map, title="habitats")

Orthogonal 2D panels (omit ``axis``) use ``display_convention="radiological"``
by default. Pass the ``ImageVolume`` and ``HabitatMap`` (not ``.data`` /
``label_array``, and not ``direction=volume.direction``). Demo image and ROI
headers can disagree on ``+z``; the habitat map keeps the mask direction so
coronal/sagittal stay superior-up. ``display_convention="native"`` skips
flips. See :mod:`habit.viz.orientation`.

.. figure:: ../_static/images/examples/two_step_overlay.png
   :alt: Two-step habitat overlay
   :width: 420

   Same file the gallery script writes to ``out/two_step_overlay.png``.

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

Do not skip clustering-time voxel scaling on two-step / direct-pooling
runs. On the demo pack (5 subjects, ``LAP``, auto-K 2–10, seed 0) an
empty ``voxel_feature_preprocessors`` chain still fitted ``model_k=4``
but used a mean of **2.8** labels per map; ``winsorize`` then ``minmax``
restored 4/4. Image-level ``zscore_normalization`` is a different step
(:func:`~habit.preprocess_subject`) and is not a substitute. Numbers
and the recommended recipe: :doc:`../examples/habitat_preprocessing`.

Which ``Spec("...")`` names exist, and what each parameter means:
:doc:`habitat_components` (Python + YAML, generated from ``params_model``).

Next: :doc:`extract_features`. Config details: :doc:`../configuration/habitat`.
