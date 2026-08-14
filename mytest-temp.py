from habit import HabitatSpec, Spec, Stage, cohort_from_directory
import habit.recipes as recipes

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {len(cohort)} subjects")

# What can go in Spec("...")? See docs/source/how_to/habitat_components.rst
# or: list_plugins("voxel_feature_extractor") / get_param_schema("kmeans", "habitat_model_fitter")
spec = HabitatSpec(
    name="habitat_one_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 10,
                    "validation": "elbow",
                    "n_init": 5,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
    ),
    random_seed=42,
)

result = recipes.Study(spec=spec).fit_predict(cohort)
print(f"Cohort-level habitat_model: {result.habitat_model}")
print(f"Per-subject models: {len(result.subject_models)}")
for subject_id, model in sorted(result.subject_models.items()):
    print(f"  {subject_id}: {model.n_habitats} habitats")
print(f"Habitat maps: {len(result.habitat_maps)}")