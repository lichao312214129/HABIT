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
Coverage matrix: habitat analysis on the synthetic tree.

Runs every clustering mode the v1.0.0 recipes expose, through both the
``habit get-habitat`` CLI (v0.1 YAML) and the v1 ``run_from_yaml`` entry
point, asserting the documented artefacts appear:

- ``two_step`` train (module-scoped, reused by predict/test-retest/extract
  suites) and predict through the saved ``habitat_model.habitatmodel``;
- ``one_step`` train with elbow selection;
- ``direct_pooling`` train with k-means and with GMM/AIC;
- the native v1 five-section document via ``habit.recipes.run_from_yaml``.

The synthetic images plant 2-3 compact intensity subregions inside each
tumour mask, so habitat clustering has real structure to recover.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from click.testing import CliRunner

from tests.cloud_coverage.conftest import RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree

#: Habitat label maps every train/predict run must produce per subject.
HABITAT_MAP_GLOB = "*_habitats.nrrd"


@pytest.fixture(scope="module")
def two_step_train_out(
    synthetic_tree: SyntheticTree, render_config
) -> Path:
    """
    Run the two-step train once per module and return its output dir.

    Args:
        synthetic_tree: Session synthetic dataset.
        render_config: Session config renderer.

    Returns:
        Output directory holding the fitted model and habitat maps.
    """
    rendered: RenderedConfig = render_config(
        "habitat_two_step_train.yaml", "habitat_two_step", synthetic_tree
    )
    run_cli(CliRunner(), ["get-habitat", "-c", str(rendered.path)])
    return rendered.out_dir


@pytest.mark.integration
def test_two_step_train_cli(two_step_train_out: Path, synthetic_tree: SyntheticTree) -> None:
    """two_step train writes the model, the unit table and the label maps."""
    out_dir = two_step_train_out
    assert (out_dir / "habitat_model.habitatmodel").is_file()
    assert list(out_dir.glob("habitats.*")), f"no habitats table under {out_dir}"
    maps = list(out_dir.glob(HABITAT_MAP_GLOB))
    assert len(maps) == len(synthetic_tree.subjects), (
        f"expected {len(synthetic_tree.subjects)} habitat maps, got {[p.name for p in maps]}"
    )
    supervoxel_maps = list(out_dir.glob("*_supervoxel.nrrd"))
    assert supervoxel_maps, "two_step must export per-subject supervoxel maps"
    # plot_curves: true in the config -> at least one figure artefact.
    figures = list(out_dir.glob("**/*.png")) + list(out_dir.glob("**/*.html"))
    assert figures, "expected clustering curve figures under out_dir"


@pytest.mark.integration
def test_two_step_train_habitat_table(two_step_train_out: Path, synthetic_tree: SyntheticTree) -> None:
    """The habitats unit table covers every subject and recovers structure."""
    out_dir = two_step_train_out
    parquet = out_dir / "habitats.parquet"
    csv = out_dir / "habitats.csv"
    if parquet.is_file():
        table = pd.read_parquet(parquet)
    else:
        assert csv.is_file(), f"no habitats table under {out_dir}"
        table = pd.read_csv(csv)
    # The unit table has one row per (subject, supervoxel) pair.
    assert set(table["subject"]) == set(synthetic_tree.subjects)
    label_cols = [c for c in table.columns if "habitat" in c.lower()]
    assert label_cols, f"no habitat column in {list(table.columns)}"
    # The planted subregions must yield at least two distinct habitats.
    assert table["habitats"].nunique() >= 2
    habitat_map = sitk.GetArrayFromImage(
        sitk.ReadImage(str(next(out_dir.glob(HABITAT_MAP_GLOB))))
    )
    labels = np.unique(habitat_map[habitat_map > 0])
    assert labels.size >= 2, "planted subregions should yield >= 2 habitats"


@pytest.mark.integration
def test_two_step_predict_cli(
    two_step_train_out: Path, synthetic_tree: SyntheticTree, render_config
) -> None:
    """two_step predict reproduces habitat maps from the saved pipeline."""
    pipeline = two_step_train_out / "habitat_model.habitatmodel"
    rendered: RenderedConfig = render_config(
        "habitat_two_step_predict.yaml",
        "habitat_two_step_predict",
        synthetic_tree,
        {"@PIPELINE_PATH@": pipeline.as_posix()},
    )
    run_cli(CliRunner(), ["get-habitat", "-c", str(rendered.path)])
    maps = list(rendered.out_dir.glob(HABITAT_MAP_GLOB))
    assert len(maps) == len(synthetic_tree.subjects)
    assert list(rendered.out_dir.glob("habitats.*")), "predict must write a habitats table"


@pytest.mark.integration
def test_one_step_train_cli(synthetic_tree: SyntheticTree, render_config) -> None:
    """one_step train with elbow selection writes maps and a habitats table."""
    rendered: RenderedConfig = render_config(
        "habitat_one_step_train.yaml", "habitat_one_step", synthetic_tree
    )
    run_cli(CliRunner(), ["get-habitat", "-c", str(rendered.path)])
    maps = list(rendered.out_dir.glob(HABITAT_MAP_GLOB))
    assert len(maps) == len(synthetic_tree.subjects)
    assert list(rendered.out_dir.glob("habitats.*"))


@pytest.mark.integration
def test_direct_pooling_train_cli(synthetic_tree: SyntheticTree, render_config) -> None:
    """direct_pooling k-means train writes maps and a habitats table."""
    rendered: RenderedConfig = render_config(
        "habitat_direct_pooling_train.yaml", "habitat_direct_pooling", synthetic_tree
    )
    run_cli(CliRunner(), ["get-habitat", "-c", str(rendered.path)])
    maps = list(rendered.out_dir.glob(HABITAT_MAP_GLOB))
    assert len(maps) == len(synthetic_tree.subjects)
    assert list(rendered.out_dir.glob("habitats.*"))


@pytest.mark.integration
def test_direct_pooling_gmm_train_cli(synthetic_tree: SyntheticTree, render_config) -> None:
    """direct_pooling GMM/AIC train writes maps and a habitats table."""
    rendered: RenderedConfig = render_config(
        "habitat_direct_pooling_gmm_train.yaml",
        "habitat_direct_pooling_gmm",
        synthetic_tree,
    )
    run_cli(CliRunner(), ["get-habitat", "-c", str(rendered.path)])
    maps = list(rendered.out_dir.glob(HABITAT_MAP_GLOB))
    assert len(maps) == len(synthetic_tree.subjects)
    assert list(rendered.out_dir.glob("habitats.*"))


@pytest.mark.integration
def test_v1_two_step_run_from_yaml(synthetic_tree: SyntheticTree, render_config) -> None:
    """The native v1 document runs through habit.recipes.run_from_yaml."""
    from habit.recipes import run_from_yaml

    rendered: RenderedConfig = render_config(
        "habitat_two_step_v1.yaml", "habitat_two_step_v1", synthetic_tree
    )
    result = run_from_yaml(rendered.path, save=True)
    assert result is not None
    assert list(rendered.out_dir.glob("habitats.*")), "v1 run must persist a habitats table"


@pytest.mark.unit
def test_synthetic_tree_is_deterministic(tmp_path: Path) -> None:
    """Two generations with the same seed produce byte-identical artefacts."""
    from tests.fixtures.synthetic_data import make_synthetic_tree

    first = make_synthetic_tree(tmp_path / "run1", seed=42)
    second = make_synthetic_tree(tmp_path / "run2", seed=42)
    image1 = (first.root / "images/subj001/delay2/image.nrrd").read_bytes()
    image2 = (second.root / "images/subj001/delay2/image.nrrd").read_bytes()
    assert image1 == image2
    assert first.clinical_csv.read_bytes() == second.clinical_csv.read_bytes()
    assert first.radiomics_csv.read_bytes() == second.radiomics_csv.read_bytes()


@pytest.mark.unit
def test_synthetic_tree_layout_and_content(synthetic_tree: SyntheticTree) -> None:
    """The tree matches the demo-data layout and carries planted structure."""
    root = synthetic_tree.root
    for subject in synthetic_tree.subjects:
        for modality in synthetic_tree.modalities:
            image = sitk.ReadImage(
                str(root / "images" / subject / modality / "image.nrrd")
            )
            assert image.GetSize() == (64, 64, 16)
            assert image.GetSpacing() == (1.0, 1.0, 2.0)
        for modality in synthetic_tree.modalities:
            mask_path = root / "masks" / subject / modality / "mask.nrrd"
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
            assert 500 < int(mask.sum()) < 20000
    clinical = pd.read_csv(synthetic_tree.clinical_csv)
    assert set(clinical.columns) == {"subject_id", "outcome", "age", "noise_score"}
    assert clinical["outcome"].nunique() == 2
    radiomics = pd.read_csv(synthetic_tree.radiomics_csv)
    feature_cols: List[str] = [c for c in radiomics.columns if c.startswith("feature_")]
    assert len(feature_cols) == 20
    assert radiomics["label"].nunique() == 2
