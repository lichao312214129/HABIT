Prepare data
============

Goal: tell HABIT where your images and ROI masks are. Modality names in the
data must match the names used later in the habitat / radiomics config.

Before you start
----------------

* HABIT installed; you can run ``habit --version``
  (:doc:`before_you_start`).
* Terminal open at the **project root** (folder that contains ``config/``).
* Prefer **NIfTI** (``.nii`` / ``.nii.gz``). Draw ROIs in ITK-SNAP or
  3D Slicer. DICOM → NIfTI: see :doc:`preprocess`.

Two ways to provide data
------------------------

**Recommended:** list every image / mask path in a small YAML file, then set
that file as ``data_dir`` or ``data.source`` in your workflow config.

**Alternative:** point ``data_dir`` / ``data.source`` at a folder that already
follows the HABIT tree (shown below).

Option A — path list YAML (recommended)
---------------------------------------

1. Copy the template::

      config/habitat/file_habitat_registered_single_roi.yaml

2. Edit **subject IDs**, **modality keys**, and **file paths** (fields marked
   ★ in that file). Keep modality keys identical to your habitat config
   (e.g. ``T1``, ``T2``).

3. In the workflow config, point at this file (not at a single ``.nii.gz``)::

      # habit get-habitat / preprocess YAML
      data_dir: ./file_habitat_registered_single_roi.yaml

      # native habitat YAML
      data:
        source: config/habitat/file_habitat_registered_single_roi.yaml

Example path list (absolute paths; relative paths are OK and resolve from
**this YAML file's directory**)::

   auto_select_first_file: false

   images:
     subj001:                                    # ★ subject ID
       T1: D:/study/registered/subj001/T1.nii.gz # ★ image path
       T2: D:/study/registered/subj001/T2.nii.gz # ★ image path
     subj002:
       T1: D:/study/registered/subj002/T1.nii.gz
       T2: D:/study/registered/subj002/T2.nii.gz

   masks:
     # One ROI per subject is enough when modalities are already co-registered.
     # Use the first modality key from your feature expression (here T1).
     subj001:
       T1: D:/study/registered/subj001/roi.nii.gz  # ★ mask path
     subj002:
       T1: D:/study/registered/subj002/roi.nii.gz

Notes:

* Direct ``.nii.gz`` paths → keep ``auto_select_first_file: false``.
* Paths that point to a **folder** containing one NIfTI → set
  ``auto_select_first_file: true`` (see ``file_habitat_demo.yaml``).
* Mask grid must match the registered image (same size / spacing).

Check syntax::

   habit check-config --config config/habitat/file_habitat_registered_single_roi.yaml --syntax-only

Option B — folder tree
----------------------

Point ``data_dir`` / ``data.source`` at a directory with this layout::

   data_root/
   ├── images/
   │   └── subj001/
   │       ├── T1/
   │       │   └── T1.nii.gz
   │       └── T2/
   │           └── T2.nii.gz
   └── masks/
       └── subj001/
           └── T1/                 # first modality used in the config
               └── roi.nii.gz

Demo pack (``preprocessed.zip`` — |download_demo_data|, code
|demo_data_code|). After extract at the project root (no nested
``processed_images``)::

   demo_data/preprocessed/
   ├── images/subj001/pre_contrast|LAP|PVP|delay_3min/...
   └── masks/subj001/pre_contrast|LAP|PVP|delay_3min/...

If the zip top level is ``preprocessed/``, extract into ``demo_data/``; if
it is ``images/`` + ``masks/``, put them under ``demo_data/preprocessed/``.

Subjects in the pack: ``subj001`` … ``subj005``. Files are NRRD under each
modality folder.

Then in the workflow YAML::

   data_dir: ../../demo_data/preprocessed

   # or
   data:
     source: demo_data/preprocessed

What success looks like
-----------------------

* ``data_dir`` / ``data.source`` is either a **path-list YAML** or a **folder
  root** — never a bare ``.nii.gz``.
* Subject IDs and modality keys match your habitat config.
* Each subject has a usable ROI when the workflow needs masks.
* ``habit check-config`` on the path-list YAML (``--syntax-only``) succeeds.

Next: :doc:`preprocess` (DICOM / raw) or :doc:`segment_habitat`.

Templates: ``config/`` — catalogue in :doc:`../configuration/recipe_catalog`.
