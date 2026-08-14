How-to guides
=============

YAML + CLI operator path. Pattern on every page:

1. Copy a shipped config under ``config/``
2. Edit only ★ fields (usually ``data_dir`` / ``out_dir``)
3. ``habit check-config -c …`` then the run command

Start: :doc:`before_you_start` → :doc:`prepare_data` → pipeline pages below.

**Which Spec to put in each habitat stage** (one series first, then
combiners such as ``concat(raw("T1"), voxel_radiomics("T2"))``):
:doc:`habitat_components`.

**Features from habitat maps** (after habitats exist): :doc:`extract_features`
and topology :doc:`graph_features` (built-in ``graph`` family; now in the
default extract list). Galleries: :doc:`../examples/graph_features` and
habitat-vs-habitat contrast :doc:`../examples/habitat_feature_compare`.

**Voxel texture maps** (local entropy / GLCM via ``voxel_radiomics``):
:doc:`voxel_texture`. Gallery: :doc:`../examples/voxel_texture`.

**Precise voxel features** (morphology-aware screen before clustering):
:doc:`../tutorial/precise_screening` (Gaussian noise, 0.5-voxel
translation, 0.5° rotation; optional MONAI B-spline / elastic FFD of the
ROI contour — not MIRP ROI grow/shrink). Gallery:
:doc:`../examples/precise_features`.

.. toctree::
   :maxdepth: 1

   before_you_start
   prepare_data
   preprocess
   segment_habitat
   habitat_components
   extract_features
   graph_features
   voxel_texture
   radiomics
   train_model
   compare_models
   auxiliary_tools
