Configuration Reference
========================

HABIT uses **YAML** to control each pipeline step. Example templates live under the repository ``config/`` directory (index in ``config/README_CONFIG.md``).

Usage: copy a template → edit ``data_dir`` / ``out_dir`` in the ``#%%====`` blocks → run the corresponding ``habit`` command.

Omitted keys use program defaults (listed on each page below). Example values in templates are for reference only.

.. note::

   If you only ``pip install habit`` without the ``config/`` directory, obtain the full source tree from `GitHub <https://github.com/lichao312214129/habit_project_v1>`_.

.. toctree::
   :maxdepth: 2

   recipe_catalog
   preprocessing
   habitat
   feature_extraction
   machine_learning
   auxiliary
