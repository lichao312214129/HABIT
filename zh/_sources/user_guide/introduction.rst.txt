Introduction
============

What habitat analysis asks
--------------------------

A whole-tumour summary collapses the ROI into one average. Inside many
lesions, subregions enhance, necrose, or sit in different locations.
**Habitat analysis** partitions the ROI into a few spatially coherent
subregions (**habitats**) whose voxels behave alike across the input
images, then describes each patient by how much of each habitat they
have and how those habitats are arranged in space.

Practical challenges
--------------------

Doing this well—and so that a second lab can rerun it—is harder than
drawing a mask and clustering once.

* A whole-tumour feature table hides those subregions; two patients can
  share a similar average and still differ where habitats sit.
* Published pipelines disagree on design: cluster inside each patient,
  pool voxels across the cohort, or build supervoxels then fit a cohort
  model. Results are hard to compare or reproduce when the design is
  not named clearly.
* MRI signal is not a quantitative unit across patients and scans. A
  model fit on raw intensities can paint a new patient as a single
  habitat. Intensities need subject-level scaling before cross-patient
  clustering; the concrete step is in Examples and
  :doc:`building_habitat_maps`.
* Habitat IDs swap between fits, so colours and column names do not
  match until labels are aligned.
* A saved model applied to a new cohort is wrong if the feature space
  differs, and the failure can look like a plausible map.
* Readers need a path from images to habitat maps, to numbers (volume,
  spatial interaction, heterogeneity, graph layout), and to a methods
  paragraph they can put in a paper.

What HABIT is for
-----------------

HABIT is a Python library for that path. The Python API is the product:
the same analysis can be declared as a ``HabitatSpec``, assembled from
atomic components, or driven by a short YAML / CLI recipe.

* Three habitat designs share one vocabulary: **two-step**
  (supervoxels, then a cohort model), **inside each subject**, and
  **pooled voxels**.
* You can embed it: arrays in, habitat map out, without HABIT's folder
  layout.
* Fit once, save a ``HabitatModel``, and apply it to new patients.
* Quantify habitats: volume fractions, MSI, ITH, per-habitat and
  whole-map radiomics, and graph-network features that describe how
  habitats sit in space.
* Radiomics on habitat maps follow IBSI / PyRadiomics definitions, with
  an IBSI-1 Phase 1 digital-phantom comparison on
  :doc:`/reference/features/traditional`.
* Screen features for reproducibility (**precise features**) and match
  habitat labels across fits.
* Run a cohort with a simple serial backend; scale with a process
  backend, failure policy, and resume when the cohort is large.
* ``describe_methods()`` drafts the methods paragraph from the run
  itself.

How to use this documentation
-----------------------------

* **Quickstart** — :doc:`/auto_quickstart/plot_quickstart_python`
  (YAML twin: :doc:`/auto_quickstart/plot_quickstart_yaml`).
* **Examples** — end-to-end code under
  :doc:`/auto_examples/index`, starting with the introductory
  tutorials.
* **User guide** — short *what* / *why* pages; copy runnable code from
  Examples. Next:
  :doc:`building_habitat_maps`,
  :doc:`features`,
  :doc:`quantifying_habitats`.
