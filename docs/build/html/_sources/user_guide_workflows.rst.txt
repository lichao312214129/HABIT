Workflows
=========

Two-Step Clustering (Default)
-----------------------------

The two-step strategy is the default method:

1. Extract voxel features for each subject
2. Cluster voxels to form supervoxels (per-subject)
3. Calculate supervoxel-level feature means
4. Cluster supervoxels to identify habitats (population-level)

One-Step Clustering
-------------------

The one-step strategy directly clusters voxels to habitats:

1. Extract voxel features for each subject
2. Cluster voxels to identify habitats (per-subject, independent)

Direct Pooling Strategy
-----------------------

The direct pooling strategy:

1. Extract voxel features for all subjects
2. Pool all voxels together
3. Cluster to identify habitats (population-level)

Choosing a Strategy
-------------------

* **Two-Step** (Default): Cohort studies, cross-patient habitat pattern identification
* **One-Step**: Individual heterogeneity analysis, small sample studies
* **Direct Pooling**: Moderate data size, unified labels without supervoxel step
