#!/usr/bin/env python
# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
End-to-end two-step habitat analysis on the HABIT demo data (v1.0 Python API).

What this script demonstrates, in order:

1. Preflight -- record the software stack (``habit.show_versions``) and check
   that every component the analysis declares is registered
   (``habit.check_component``), so a typo fails before any compute starts.
2. Cohort assembly -- read the demo cohort from the conventional directory
   layout (``<root>/images/<subject>/<modality>/`` plus
   ``<root>/masks/<subject>/<roi>/``) with ``habit.cohort_from_directory``.
3. Specification -- declare the whole analysis as a ``HabitatSpec``: a frozen,
   fingerprintable value object that knows nothing about files or execution.
4. Training -- run the ``two_step`` recipe (supervoxels per subject, habitats
   learned across the cohort) and inspect the in-memory ``StudyResult``.
5. Persistence -- write habitat maps, feature tables, the model archive and
   the run manifest with ``StudyResult.save()``; writing to disk is a
   separate, explicit act in v1.0.
6. Prediction -- reload the ``.habitatmodel`` archive and project the SAME
   habitat definition onto the cohort with ``apply_habitat_model``. On a
   real study this second cohort would be new data; reusing the training
   cohort here doubles as a determinism check (labels must match exactly).

Run from the repository root inside the py310 conda environment::

    E:\\conda\\mconda\\envs\\py310\\python.exe examples/habitat_v1_two_step_demo.py

Add ``--dry-run`` to validate imports, the spec and cohort discovery without
running the computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import habit
import habit.recipes as recipes
from habit import Cohort, HabitatModel, HabitatSpec, Spec, StudyResult

# --- Paths and analysis constants -----------------------------------------

#: Repository root: examples/habitat_v1_two_step_demo.py -> parents[1]
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

#: Demo imaging root holding images/ and masks/ in the conventional layout.
DATA_ROOT: Path = REPO_ROOT / "demo_data" / "preprocessed" / "processed_images"

#: Destination for every artefact this script writes.
OUT_DIR: Path = REPO_ROOT / "demo_data" / "results" / "examples" / "habitat_v1_two_step_demo"

#: DCE-MRI modality keys under DATA_ROOT/images/<subject>/.
MODALITIES: tuple[str, ...] = ("delay2", "delay3", "delay5")

#: Mask key: the demo stores one ROI mask per subject under masks/<subject>/delay2/.
ROI: str = "delay2"

#: Seed applied to every stochastic component; part of the spec fingerprint.
RANDOM_SEED: int = 42

#: (component name, registry domain) pairs checked before any compute starts.
PREFLIGHT_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("raw", "voxel_feature_extractor"),
    ("kmeans", "supervoxelizer"),
    ("kmeans", "habitat_model_fitter"),
    ("nearest_centroid", "habitat_assigner"),
    ("volume", "habitat_feature_extractor"),
    ("msi", "habitat_feature_extractor"),
    ("ith_score", "habitat_feature_extractor"),
    ("winsorize", "feature_preprocessing_method"),
    ("minmax", "feature_preprocessing_method"),
    ("binning", "feature_preprocessing_method"),
)


def build_spec() -> HabitatSpec:
    """
    Declare the two-step habitat analysis as a frozen spec.

    The parameters mirror the shipped ``config/habitat/config_habitat_two_step.yaml``
    demo (and therefore the golden baseline), plus three habitat feature
    families. Feature families are computed AFTER habitat assignment, so they
    enrich the feature table without changing the habitat maps or the model.

    Returns:
        The fully wired habitat specification.
    """
    return HabitatSpec(
        name="habitat_two_step",
        # Voxel level: concatenate the raw intensities of the three DCE phases.
        voxel_feature_extractor=Spec(name="raw", params={"modalities": list(MODALITIES)}),
        # Supervoxel level: k-means over per-subject voxel features.
        supervoxelizer=Spec(
            name="kmeans",
            params={"n_supervoxels": 50, "max_iter": 300, "n_init": 10},
        ),
        # Cohort level: k-means over pooled supervoxels, habitat count selected
        # automatically in [2, 10] by the elbow criterion.
        habitat_model_fitter=Spec(
            name="kmeans",
            params={
                "min_habitats": 2,
                "max_habitats": 10,
                "validation": "elbow",
                "max_iter": 300,
                "n_init": 10,
            },
        ),
        # Assignment: each supervoxel takes the habitat of its nearest centroid.
        habitat_assigner=Spec(name="nearest_centroid"),
        # Habitat feature families: per-subject descriptors of the label map.
        habitat_features=(
            Spec(name="volume"),     # voxel counts and volume fractions per habitat
            Spec(name="msi"),        # multiregional spatial interaction features
            Spec(name="ith_score"),  # ITH score and per-habitat fragmentation
        ),
        # Per-subject chain, applied BEFORE supervoxelization: clip extremes,
        # then rescale each subject's features to a common range.
        voxel_feature_preprocessors=(
            Spec(name="winsorize", params={"winsor_limits": (0.05, 0.05), "across_features": False}),
            Spec(name="minmax", params={"across_features": False}),
        ),
        # Cohort-level chain, fitted once on the pooled TRAINING units; its
        # fitted state travels inside the saved HabitatModel.
        cohort_feature_preprocessors=(
            Spec(name="binning", params={"n_bins": 10, "bin_strategy": "uniform", "across_features": False}),
        ),
        random_seed=RANDOM_SEED,
    )


def preflight(spec: HabitatSpec) -> None:
    """
    Record the software stack and verify every declared component resolves.

    Args:
        spec: The specification whose components are checked.
    """
    for package, version in habit.show_versions().items():
        print(f"  {package}: {version}")
    for name, domain in PREFLIGHT_COMPONENTS:
        if not habit.check_component(name, domain=domain):
            raise RuntimeError(f"Component {name!r} is not registered in domain {domain!r}.")
    print(f"Preflight OK: {len(PREFLIGHT_COMPONENTS)} components registered.")
    print(f"Spec fingerprint: {spec.fingerprint()}")


def load_cohort() -> Cohort:
    """
    Build the demo cohort from the conventional directory layout.

    Returns:
        The cohort in reproducible (sorted subject id) order, holding lazy
        file references -- no image is loaded until a stage needs it.
    """
    cohort: Cohort = habit.cohort_from_directory(
        DATA_ROOT,
        modalities=MODALITIES,
        roi=ROI,
        name="demo_dce",
    )
    print(f"Loaded cohort: {len(cohort)} subjects -> {cohort.subject_ids}")
    return cohort


def train(cohort: Cohort, spec: HabitatSpec) -> StudyResult:
    """
    Fit the cohort-level habitat definition and label every subject.

    Args:
        cohort: Subjects to fit the habitat definition on.
        spec: The analysis to run.

    Returns:
        The in-memory study result (model, maps, features, manifest).
    """
    result: StudyResult = recipes.two_step(cohort, spec)

    assert result.habitat_model is not None  # two_step always fits a cohort model
    print("\n--- Fitted habitat model ---")
    print(result.habitat_model.summary())

    print(f"\nFeature table: {result.features.frame.shape[0]} subjects x "
          f"{len(result.features.feature_columns)} features")

    # Auto-generated methods paragraph, stating only steps that executed.
    print("\n--- Methods paragraph (from the run manifest) ---")
    print(result.manifest.describe_methods())
    return result


def predict_with_saved_model(cohort: Cohort, spec: HabitatSpec, train_result: StudyResult) -> int:
    """
    Reload the saved model archive and project it onto the cohort.

    Args:
        cohort: Subjects to label (new data in a real study).
        spec: Spec whose upstream stages match the fitted model.
        train_result: The training result, used for the determinism check.

    Returns:
        Number of subjects whose predicted labels differ from training;
        ``0`` confirms the save/load/apply round-trip is exact.
    """
    archive: Path = OUT_DIR / "habitat_model.habitatmodel"
    model: HabitatModel = HabitatModel.load(archive)
    print(f"\nReloaded model {model.model_id} ({model.n_habitats} habitats) from {archive}")

    predict_result: StudyResult = recipes.apply_habitat_model(cohort, spec, model)
    predict_dir: Path = predict_result.save(OUT_DIR / "predict_on_train")
    print(f"Predict artefacts written to: {predict_dir}")

    mismatches: int = sum(
        1
        for train_map, predict_map in zip(train_result.habitat_maps, predict_result.habitat_maps)
        if not np.array_equal(train_map.label_array, predict_map.label_array)
    )
    print(f"Label mismatches vs training: {mismatches} / {len(train_result.habitat_maps)}")
    return mismatches


def main() -> None:
    """Run the demo: preflight, train, save, reload, predict, verify."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate spec and cohort discovery without running the analysis.",
    )
    args = parser.parse_args()

    spec: HabitatSpec = build_spec()
    preflight(spec)
    cohort: Cohort = load_cohort()
    if args.dry_run:
        print("Dry-run OK: spec, components and cohort validated.")
        return

    train_result: StudyResult = train(cohort, spec)

    # Persist everything: NRRD maps, parquet/csv tables, model, manifest, plots.
    saved_dir: Path = train_result.save(
        OUT_DIR,
        write_cluster_plots=True,
        write_cluster_plots_3d=True,
        write_interactive_cluster_plots=True,  # skipped gracefully without plotly
    )
    print(f"\nTraining artefacts written to: {saved_dir}")

    mismatches: int = predict_with_saved_model(cohort, spec, train_result)
    if mismatches:
        raise RuntimeError(f"Predict round-trip mismatch on {mismatches} subject(s).")
    print("Done.")


if __name__ == "__main__":
    main()
