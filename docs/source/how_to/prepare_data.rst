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

Public MSD BrainTumour alternative
----------------------------------

If you cannot download the Baidu ``demo_data.rar`` pack, fetch 5–10 public
MSD Task01 BrainTumour (BraTS-like) cases with stdlib + SimpleITK only
(no MONAI / torch)::

   python scripts/download_msd_brain_demo.py --n 5

Layout written under ``demo_data/preprocessed/processed_images/``::

   images/<subj>/{flair,t1,t1ce,t2}/<subj>_<mod>.nii.gz
   masks/<subj>/tumor/<subj>_tumor.nii.gz

MSD stores modalities as one 4D volume in the order **flair, t1, t1ce, t2**.
The script also copies the whole-tumor mask (``label > 0``) under each
modality folder so ``habit get-habitat`` can load the tree with the current
CLI ROI rule (first modality key in the feature expression).

Matching habitat config (replaces ``delay2`` / ``delay3`` / ``delay5``)::

   habit check-config -c config/habitat/config_habitat_msd_demo.yaml
   habit get-habitat  -c config/habitat/config_habitat_msd_demo.yaml

Acknowledgement: Medical Segmentation Decathlon Task01 / BraTS 2016–2017;
please cite MSD and BraTS when using these cases. License: CC-BY-SA 4.0
(``https://medicaldecathlon.com/``). The large tarball is cached under
``demo_data/_msd_cache/`` (gitignored).

Data specification
------------------

1. **Folder**: ``data_dir`` points to a root with ``images/`` and ``masks/``
2. **Manifest YAML** (recommended): ``data_dir`` points to e.g. ``config/habitat/file_habitat_demo.yaml``

Draw ROIs in **ITK-SNAP** or **3D Slicer** ; export NIfTI masks.

Templates: ``config/`` — see :doc:`../configuration/recipe_catalog`.
