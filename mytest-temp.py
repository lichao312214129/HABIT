from pathlib import Path

from habit import make_synthetic_cohort
from habit.domain import (
    RawVoxelFeatures,
    KMeansSupervoxelizer,
    SupervoxelRadiomicsFeatures,
)

# Need pyradiomics installed. Prefer real/demo Subject; synthetic works for API smoke.
subject = make_synthetic_cohort(n_subjects=1, shape=(24, 24, 24), rng=0)[0]

# 1) Voxel features (used only to build the partition)
field = RawVoxelFeatures(modalities=["T1"])(subject)

# 2) Supervoxel labels
partition = KMeansSupervoxelizer(n_supervoxels=8, n_init=5)(field)

# 3) Per-supervoxel texture (reads original images inside each supervoxel ROI)
params = Path("habit/resources/radiomics/params_supervoxel_radiomics.yaml")
tex = SupervoxelRadiomicsFeatures(
    modalities=["T1"],                 # or modality="T1"
    params_file=str(params),           # omit → bundled preset; or pass params={...}
    use_torch_radiomics=False,         # CPU / PyRadiomics path
)(subject, partition)

# Result: DataFrame, index = supervoxel id, columns = texture features (*-T1)
df = tex.features
print(df.shape, list(df.columns)[:8])
# label map unchanged: tex.label_array


tex = SupervoxelRadiomicsFeatures(
    modalities=["T1", "T2"],
    params_file=str(params),
)(subject, partition)
# columns look like featureName-T1, featureName-T2