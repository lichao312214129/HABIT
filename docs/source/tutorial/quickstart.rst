Quickstart: YAML + CLI (no programming)
=======================================

This is the clinician / no-programming quickstart: you edit a YAML file and
run ``habit`` commands; you never write Python. (In v1.0 the CLI is a thin
shell over the Python API — same specs, same results. Developers should use
the parallel :doc:`quickstart_python` instead.)

End-to-end pipeline: preprocessing → habitat segmentation → feature
extraction → machine learning → model comparison.

Prerequisites: :doc:`installation` .

Prepare data
------------

.. note::

   ``D:\habit`` is an example project root — use your clone or working directory.

1. Obtain demo images (pick **one** path):

   **A. Packaged DCE demo** (Baidu Netdisk; matches the YAML commands below)::

      demo_data.rar → extract at the project root (folder that contains ``config/``)

   - |download_demo_data|
   - Code: |demo_data_code|

   Modalities in that pack are ``delay2`` / ``delay3`` / ``delay5``.

   **B. Public MSD BrainTumour mini-demo** (international HTTPS; no Baidu)::

      python scripts/download_msd_brain_demo.py --n 5

   This pulls Medical Segmentation Decathlon Task01 (BraTS-like) cases via
   plain HTTPS, splits the 4D volumes with SimpleITK, and writes HABIT's
   ``images/`` + ``masks/`` layout under
   ``demo_data/preprocessed/processed_images/``. Modalities are
   ``t1ce`` / ``t1`` / ``t2`` / ``flair``; ROI folder is ``tumor``.
   Then run the matching config instead of the delay* demo::

      habit check-config -c config/habitat/config_habitat_msd_demo.yaml
      habit get-habitat  -c config/habitat/config_habitat_msd_demo.yaml

   Cite MSD / BraTS when publishing; data license CC-BY-SA 4.0
   (see ``medicaldecathlon.com``). Details:
   :doc:`../how_to/prepare_data`.

   ``config/`` is already included in the source checkout.

   ``tests.zip`` (optional)

   - |download_tests_pack|
   - Code: |tests_pack_code|

2. Verify ``habit --version`` prints a ``1.0.x`` version.

Run the demo
------------

Demo includes preprocessed data — **start at step 2** on first run.

0. *(Recommended)* Validate a config before running it — this catches YAML
   typos and unknown keys without starting any computation:

   .. code-block:: bash

      habit check-config --config config/habitat/config_habitat_two_step.yaml

1. Run the pipeline (each command reads one YAML and writes under
   ``demo_data/results/``):

   .. code-block:: bash

      cd /d D:\habit-cpu

      habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml
      habit get-habitat --config config/habitat/config_habitat_two_step.yaml
      habit extract --config config/feature_extraction/config_extract_features_demo.yaml
      habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train
      habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
      habit compare --config config/model_comparison/config_model_comparison_demo.yaml

Outputs under ``demo_data/results/`` . Your own data → :doc:`../how_to/index` .

The two config formats (v0.1 and v1)
------------------------------------

``config/`` ships both generations of YAML:

* **v0.1** (most files, e.g. ``config_habitat_two_step.yaml``) — the
  long-standing layout; the CLI translates it internally before running.
* **v1** (``*_v1.yaml``, e.g. ``config_habitat_two_step_v1.yaml``) — the
  native v1.0 layout whose ``spec:`` section mirrors the Python
  :class:`~habit.spec.HabitatSpec` field for field.

Both run through the same ``habit`` commands. To upgrade your own v0.1
config to v1:

.. code-block:: bash

   habit migrate-config --config config/habitat/config_habitat_two_step.yaml

See :doc:`../configuration/index` for the full field reference and
:doc:`../api/spec` for how the two formats relate.

Where to go next
----------------

* :doc:`../how_to/index` — step-by-step guides with your own data
* :doc:`../reference/cli` — all 16 subcommands
* :doc:`../configuration/index` — every YAML field, by workflow
* :doc:`quickstart_python` — the same analyses from Python
