#!/usr/bin/env python
"""
Run a v1 YAML configuration document with recipes.run_from_yaml.

This script accompanies docs/source/examples/run_from_yaml.rst. To keep the
example self-contained it first writes a tiny imaging dataset in HABIT's
conventional directory layout (four subjects, two modalities, one ROI) and a
v1 YAML document, then executes the document exactly as the CLI would --
including persisting outputs.

Run from the repository root:

    python docs/source/examples/scripts/run_from_yaml_demo.py
"""

import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import habit.recipes as recipes

work_dir = Path(tempfile.mkdtemp(prefix="habit_yaml_demo_"))

# 1. A tiny dataset in the conventional layout:
#      <root>/images/<subject>/<modality>/<file>.nrrd
#      <root>/masks/<subject>/<first modality>/<file>.nrrd
rng = np.random.default_rng(42)
data_root = work_dir / "dataset"
for index in range(4):
    subject_id = f"P{index:03d}"
    for modality in ("T1", "T2"):
        folder = data_root / "images" / subject_id / modality
        folder.mkdir(parents=True)
        array = rng.normal(100, 15, size=(16, 16, 16)).astype(np.float32)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image, str(folder / f"{subject_id}_{modality}.nrrd"))
    folder = data_root / "masks" / subject_id / "T1"
    folder.mkdir(parents=True)
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[4:12, 4:12, 4:12] = 1
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(mask_image, str(folder / f"{subject_id}_mask.nrrd"))
print(f"Wrote 4 synthetic subjects under {data_root}")

# 2. The v1 document: version + workflow + mode + spec + data + output.
#    The spec section mirrors habit.HabitatSpec field for field.
yaml_path = work_dir / "analysis.yaml"
yaml_path.write_text(
    f"""\
version: '1.0'
workflow: habitat
mode: train

spec:
  name: habitat_two_step
  voxel_feature_extractor:
    name: raw
    params:
      modalities: [T1, T2]
  supervoxelizer:
    name: kmeans
    params:
      n_supervoxels: 6
      n_init: 5
  habitat_model_fitter:
    name: kmeans
    params:
      min_habitats: 2
      max_habitats: 3
      validation: elbow
      n_init: 5
  habitat_assigner:
    name: nearest_centroid
    params: {{}}
  habitat_features:
    - name: volume
      params: {{}}
    - name: msi
      params: {{}}
    - name: ith_score
      params: {{}}
    - name: non_radiomics
      params: {{}}
    # Heavy PyRadiomics families (opt-in; require pyradiomics):
    # - name: traditional
    #   params: {{}}
    # - name: whole_habitat
    #   params: {{}}
    # - name: each_habitat
    #   params: {{}}
  random_seed: 42

data:
  source: {data_root}

output:
  out_dir: {work_dir / 'out'}
  habitats_results_format: csv
""",
    encoding="utf-8",
)
print(f"Wrote v1 document {yaml_path}")

# 3. Execute it. save=True writes the same artefacts the CLI would: NRRD
#    habitat maps, the feature table, the .habitatmodel archive and the run
#    manifest under output.out_dir.
result = recipes.run_from_yaml(yaml_path, workflow="habitat", save=True)

print(f"\nResult type: {type(result).__name__}")
print(f"Habitats: {result.habitat_model.n_habitats}")
print(f"Habitat maps: {len(result.habitat_maps)}")

out_dir = work_dir / "out"
print(f"\nArtefacts under {out_dir}:")
for path in sorted(out_dir.rglob("*")):
    if path.is_file() and not path.name.endswith(".pkl"):  # checkpoints omitted
        print(f"  {path.relative_to(out_dir)}")

# Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from habit.adapters import DirectoryDataSource
from _habitat_eye_check import eye_check_study
cohort = DirectoryDataSource(data_root, modalities=("T1", "T2"), roi="T1").load()
eye_check_study(cohort, result)
