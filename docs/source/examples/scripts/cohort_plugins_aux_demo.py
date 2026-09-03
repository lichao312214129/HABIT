#!/usr/bin/env python
"""
Cohort assembly, plugins, auxiliary workflows, and config tooling.

Covers:

* :func:`~habit.contracts.cohort_from_directory` (batch) and single-subject slice
* :func:`~habit.api.plugins.list_plugins`
* :func:`~habit.recipes.dice`, :func:`~habit.recipes.dicom_info`,
  :func:`~habit.recipes.merge_tables`
* :func:`~habit.recipes.icc_analysis`, :func:`~habit.recipes.dice`,
  :func:`~habit.recipes.merge_tables` when ``demo_data/ml_data`` is present
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

# BEGIN example
from habit.contracts import cohort_from_directory
from habit.api.plugins import list_plugins
from habit.datasets import make_synthetic_cohort
from habit.commands.cmd_check_config import run_check_config
from habit.commands.cmd_migrate_config import run_migrate_config
import habit.recipes as recipes

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
IMAGING_ROOT: Path = REPO_ROOT / "demo_data" / "preprocessed"
ML_DATA: Path = REPO_ROOT / "demo_data" / "ml_data"
V0_HABITAT_YAML: Path = REPO_ROOT / "config" / "habitat" / "config_habitat_two_step.yaml"
HABITAT_MAPS: Path = REPO_ROOT / "demo_data" / "results" / "api" / "02_habitat_two_step"

# --- Cohort: batch load + atomic single-subject slice -------------------------
if IMAGING_ROOT.is_dir():
    cohort = cohort_from_directory(
        IMAGING_ROOT,
        modalities=("pre_contrast", "LAP", "PVP", "delay_3min"),
        roi="LAP",
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
    icc_out = work_dir / "icc"
    icc_out.mkdir(parents=True, exist_ok=True)
    icc_config: Dict[str, Any] = {
        "input": {
            "type": "files",
            "file_groups": [[str(RADIOMICS_CSV), str(RETEST_CSV)]],
        },
        "output": {"path": str(icc_out / "icc_results.json")},
        "metrics": ["icc2", "icc3"],
    }
    try:
        icc_result = recipes.icc_analysis(icc_config)
        print(f"\nicc_analysis: {icc_result.output_dir}")
        if icc_result.artifacts:
            print(f"  icc_result: {icc_result.artifacts.get('icc_result', 'n/a')}")
    except Exception as exc:  # noqa: BLE001 - demo continues if CSV ICC is heavy
        print(f"\nicc_analysis: {exc}")
else:
    print("\nicc_analysis: skipped (need demo_data/ml_data/*.csv)")

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
# END example

# BEGIN figures
# Paste after the Script block. Uses dice_df and work_dir from the Dice demo.
from habit.viz import plot_habitat_overlay
import matplotlib.pyplot as plt

id_col = next(c for c in dice_df.columns if c.lower() != "dice")
fig_dice, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(dice_df[id_col].astype(str), dice_df["Dice"], color="#4C78A8")
ax.set_ylim(0.0, 1.05)
ax.set_ylabel("Dice")
ax.set_xlabel(id_col)
ax.set_title("Pairwise mask Dice")
fig_dice.tight_layout()
Path("out").mkdir(exist_ok=True)
fig_dice.savefig("out/cohort_plugins_dice.png", dpi=150, bbox_inches="tight")

mask_a = sitk.GetArrayFromImage(
    sitk.ReadImage(str(work_dir / "batch_a" / "masks" / "P001" / "tumor" / "P001_mask.nrrd"))
)
fig_overlay = plot_habitat_overlay(
    np.ones_like(mask_a, dtype=np.float32),
    mask_a.astype(np.int32),
    axis=0,
    title="Dice demo: batch_a ROI",
)
fig_overlay.savefig("out/cohort_plugins_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/cohort_plugins_dice.png and out/cohort_plugins_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure

    save_example_figure(fig_dice, "cohort_plugins_dice.png")
    save_example_figure(fig_overlay, "cohort_plugins_overlay.png")
