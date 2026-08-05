#!/usr/bin/env python
"""
Cohort assembly, plugins, auxiliary workflows, and config tooling.

Covers:

* :func:`~habit.cohort_from_directory` (batch) and single-subject slice
* :func:`~habit.api.plugins.list_plugins`
* :func:`~habit.recipes.dice`, :func:`~habit.recipes.dicom_info`,
  :func:`~habit.recipes.merge_tables`
* :func:`~habit.recipes.icc_analysis`, :func:`~habit.recipes.test_retest_analysis`
  when ``demo_data/ml_data`` is present
* programmatic :func:`~habit.commands.cmd_check_config.run_check_config` and
  :func:`~habit.commands.cmd_migrate_config.run_migrate_config`

This script accompanies ``docs/source/examples/cohort_plugins_auxiliary.rst``.

Run from the repository root::

    python docs/source/examples/scripts/cohort_plugins_aux_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit import cohort_from_directory, list_plugins, make_synthetic_cohort
from habit.commands.cmd_check_config import run_check_config
from habit.commands.cmd_migrate_config import run_migrate_config
import habit.recipes as recipes

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
IMAGING_ROOT: Path = REPO_ROOT / "demo_data" / "preprocessed" / "processed_images"
ML_DATA: Path = REPO_ROOT / "demo_data" / "ml_data"
V0_HABITAT_YAML: Path = REPO_ROOT / "config" / "habitat" / "config_habitat_two_step.yaml"
HABITAT_MAPS: Path = REPO_ROOT / "demo_data" / "results" / "api" / "02_habitat_two_step"

# --- Cohort: batch load + atomic single-subject slice -------------------------
if IMAGING_ROOT.is_dir():
    cohort = cohort_from_directory(
        IMAGING_ROOT,
        modalities=("delay2", "delay3", "delay5"),
        roi="delay2",
    )
    print(f"cohort_from_directory (batch): {len(cohort)} subjects from demo_data")
    one = cohort[0]
    print(f"atomic slice cohort[0]: {one.subject_id}, modalities={list(one.images.keys())}")
else:
    cohort = make_synthetic_cohort(n_subjects=3, rng=1)
    print(f"demo_data absent — synthetic cohort: {len(cohort)} subjects")

# --- Plugin inventory (subset) ------------------------------------------------
for domain in ("voxel_feature_extractor", "habitat_model_fitter", "classifier"):
    names: Sequence[str] = tuple(info.name for info in list_plugins(domain=domain))
    preview = ", ".join(names[:5])
    suffix = "..." if len(names) > 5 else ""
    print(f"list_plugins({domain!r}): {len(names)} registered — {preview}{suffix}")

work_dir = Path(tempfile.mkdtemp(prefix="habit_aux_demo_"))

# --- dice: pairwise mask comparison -------------------------------------------
for batch, label in [(1, "batch_a"), (2, "batch_b")]:
    root = work_dir / label
    for subject in ("P001", "P002"):
        folder = root / "masks" / subject / "tumor"
        folder.mkdir(parents=True)
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        if batch == 2 and subject == "P002":
            mask[3:7, 3:7, 3:7] = 1
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(folder / f"{subject}_mask.nrrd"))

dice_df = recipes.dice(str(work_dir / "batch_a"), str(work_dir / "batch_b"))
print(f"\ndice(): {len(dice_df)} pairwise rows, mean Dice={dice_df['Dice'].mean():.3f}")

# --- merge_tables -------------------------------------------------------------
left = pd.DataFrame({"subject_id": ["P001", "P002"], "feat_a": [1.0, 2.0]})
right = pd.DataFrame({"subject_id": ["P001", "P002"], "feat_b": [3.0, 4.0]})
left_csv = work_dir / "left.csv"
right_csv = work_dir / "right.csv"
left.to_csv(left_csv, index=False)
right.to_csv(right_csv, index=False)
merged = recipes.merge_tables([str(left_csv), str(right_csv)], index_cols=["subject_id"])
print(f"merge_tables: {merged.shape[1]} columns")

# --- dicom_info (synthetic stand-in when no DICOM tree) -----------------------
print("\ndicom_info: requires a DICOM directory; use recipes.sort_dicom first.")
print("  CLI: habit sort-dicom / habit dicom-info — see config/sort_dicom/")

# --- icc_analysis (demo radiomics retest CSVs) --------------------------------
RADIOMICS_CSV = ML_DATA / "breast_cancer_dataset.csv"
RETEST_CSV = ML_DATA / "breast_cancer_dataset_retest_simulated.csv"
if RADIOMICS_CSV.is_file() and RETEST_CSV.is_file():
    icc_config: Dict[str, Any] = {
        "files": [[str(RADIOMICS_CSV), str(RETEST_CSV)]],
        "metrics": ["icc2", "icc3"],
        "out_dir": str(work_dir / "icc"),
    }
    icc_result = recipes.icc_analysis(icc_config)
    print(f"\nicc_analysis: {icc_result.output_dir}")
    if icc_result.artifacts:
        print(f"  icc_result: {icc_result.artifacts.get('icc_result', 'n/a')}")
else:
    print("\nicc_analysis: skipped (need demo_data/ml_data/*.csv)")

# --- test_retest_analysis (habitat maps from API coverage when present) -------
if HABITAT_MAPS.is_dir() and list(HABITAT_MAPS.glob("*_habitats.nrrd")):
    retest_config: Dict[str, Any] = {
        "habitats_map_folder": str(HABITAT_MAPS),
        "out_dir": str(work_dir / "test_retest"),
        "habitat_pattern": "*_habitats.nrrd",
        "feature_columns": ["count", "delay2", "delay3"],
    }
    try:
        retest_result = recipes.test_retest_analysis(retest_config)
        print(f"test_retest_analysis: {retest_result.output_dir}")
    except Exception as exc:  # noqa: BLE001 - report and continue in demo
        print(f"test_retest_analysis: {exc}")
else:
    print("test_retest_analysis: skipped (run demo_data/results/api/02_habitat_two_step first)")

# --- check-config / migrate-config --------------------------------------------
if V0_HABITAT_YAML.is_file():
    print(f"\ncheck-config: {V0_HABITAT_YAML.name}")
    run_check_config(str(V0_HABITAT_YAML), workflow="habitat", syntax_only=False)
    migrated = work_dir / "habitat_two_step.v1.yaml"
    run_migrate_config(str(V0_HABITAT_YAML), output_path=str(migrated), dry_run=False)
    print(f"migrate-config wrote: {migrated.name} ({migrated.stat().st_size} bytes)")
else:
    print(f"\nSkipping check-config (missing {V0_HABITAT_YAML})")

print("\nsort_dicom: batch DICOM reorganisation — recipes.sort_dicom(config)")
print("  requires a DICOM tree; see config/sort_dicom/ and tests/examples/")
