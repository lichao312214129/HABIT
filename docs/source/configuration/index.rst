Configuration Reference
========================

HABIT uses **YAML** to control each pipeline step. Example templates live under
the repository ``config/`` directory (catalog: :doc:`recipe_catalog`).

Usage: copy a template → edit ``data_dir`` / ``out_dir`` in the ``#%%====`` blocks → run the corresponding ``habit`` command.

Omitted keys use program defaults (listed on each page below). Example values in templates are for reference only.

.. note::

   If you only ``pip install habit`` without the ``config/`` directory, obtain the full source tree from |link_github_repo|.

.. toctree::
   :maxdepth: 2

   recipe_catalog
   preprocessing
   dicom_sort
   habitat
   feature_extraction
   radiomics
   machine_learning
   model_comparison
   auxiliary
