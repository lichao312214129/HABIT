Prepare data
============

Project root
------------

Run commands from the **project root** (folder containing ``config/`` ):

- **Portable pack**: extract directory (same level as ``python.exe`` )
- **Source**: GitHub repo root

Paths in YAML are resolved **relative to the YAML file location** unless absolute.

Folder layout
-------------

.. code-block:: text

   data_root/
   ├── images/
   │   └── subject001/
   │       ├── T1/T1.nii.gz
   │       └── T2/T2.nii.gz
   └── masks/
       └── subject001/
           ├── T1/mask.nii.gz
           └── T2/mask.nii.gz

See ``demo_data/preprocessed/processed_images/`` after extracting the demo.

Data specification
------------------

1. **Folder**: ``data_dir`` points to a root with ``images/`` and ``masks/``
2. **Manifest YAML** (recommended): ``data_dir`` points to e.g. ``config/habitat/file_habitat_demo.yaml``

Draw ROIs in **ITK-SNAP** or **3D Slicer** ; export NIfTI masks.

Templates: ``config/`` — see ``config/README_CONFIG.md`` .
