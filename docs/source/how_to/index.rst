How-to guides
=============

Habitat analysis is the core. Image preprocessing, whole-ROI radiomics,
and tabular ML are supporting pages at the bottom.

**Concept and embedding:** :doc:`../tutorial/habitat_analysis` ·
:doc:`../examples/habitat_atomic_ops`. **Parallel / fault tolerance:**
:doc:`../tutorial/execution`.

YAML + CLI operator path on the pages below:

1. Copy a shipped config under ``config/``
2. Edit only ★ fields (usually ``data_dir`` / ``out_dir``)
3. ``habit check-config -c …`` then the run command

Start: :doc:`before_you_start` → :doc:`prepare_data` →
:doc:`segment_habitat`.

**Which Spec to put in each habitat stage** (one series first, then
combiners such as ``concat(raw("T1"), voxel_radiomics("T2"))``):
:doc:`habitat_components`.

**Features from habitat maps** (after habitats exist): :doc:`extract_features`
and topology :doc:`graph_features`. Galleries:
:doc:`../examples/graph_features` and
:doc:`../examples/habitat_feature_compare`.

**Voxel texture maps** (inputs to clustering, via ``voxel_radiomics``):
:doc:`voxel_texture`. Precise-feature screen:
:doc:`../tutorial/precise_screening`.

.. toctree::
   :maxdepth: 1

   before_you_start
   prepare_data
   segment_habitat
   habitat_components
   extract_features
   graph_features
   voxel_texture
   ../tutorial/precise_screening
   preprocess
   radiomics
   train_model
   compare_models
   auxiliary_tools
