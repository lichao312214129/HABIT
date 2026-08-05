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

   ``D:\habit-cpu`` is an example path — use your portable or project root.

1. Download and extract to the project root (same level as ``python.exe`` or repo root):

   ``demo_data.rar`` (required)

   - |download_demo_data|
   - Code: |demo_data_code|

   ``config/`` is already included in both the portable ZIP and source checkout.

   ``tests.zip`` (optional)

   - |download_tests_pack|
   - Code: |tests_pack_code|

2. Verify ``habit --version`` prints ``1.0.0``.

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
