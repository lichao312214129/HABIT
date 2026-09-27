Glossary
========

* **habitat** — sub-region of the ROI with similar voxel features.
* **ROI / mask** — region analysed; background is 0.
* **subject-level preprocessing** — transform using one patient's
  statistics only (default: winsorize 1% + z-score).
* **cohort-level preprocessing** — transform learned on pooled training
  rows and stored in the model (not the default after subject z-score).
* **supervoxel** — small patch of similar neighbouring voxels.
* **pool** — stack per-subject units into one matrix.
* **.habitatmodel** — saved habitat definition for reuse.
