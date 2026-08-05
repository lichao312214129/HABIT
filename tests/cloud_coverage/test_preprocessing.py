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
Coverage matrix: image preprocessing on the synthetic tree.

- resample-only and resample+z-score variants through ``habit preprocess``;
- the elastix-registration and dcm2nii variants are exercised for config
  validation but skipped for execution because the elastix/dcm2niix
  binaries ship as Windows executables only (tools/bin/*.exe) and are on
  the do-not-install list for this environment.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from click.testing import CliRunner

from tests.cloud_coverage.conftest import REPO_ROOT, RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree

def _output_images(out_dir: Path) -> list:
    """Return every preprocessed image volume (NIfTI) below the out dir."""
    return sorted((out_dir / "processed_images" / "images").glob("**/*.nii.gz"))


@pytest.mark.integration
def test_preprocess_resample_only(synthetic_tree: SyntheticTree, render_config) -> None:
    """Resampling rewrites every modality at the 3x3x3 mm target spacing."""
    rendered: RenderedConfig = render_config(
        "preprocess_resample_only.yaml", "preprocess_resample", synthetic_tree
    )
    run_cli(CliRunner(), ["preprocess", "-c", str(rendered.path)])
    outputs = _output_images(rendered.out_dir)
    # 4 subjects x 3 modalities.
    assert len(outputs) == 12, (
        f"expected 12 resampled images, got {[str(p) for p in outputs]}"
    )
    for path in outputs:
        spacing = sitk.ReadImage(str(path)).GetSpacing()
        assert spacing == pytest.approx((3.0, 3.0, 3.0)), f"{path.name}: {spacing}"


@pytest.mark.integration
def test_preprocess_resample_zscore(synthetic_tree: SyntheticTree, render_config) -> None:
    """Resample+z-score rewrites images with ~zero mean and unit std."""
    rendered: RenderedConfig = render_config(
        "preprocess_resample_zscore.yaml", "preprocess_zscore", synthetic_tree
    )
    run_cli(CliRunner(), ["preprocess", "-c", str(rendered.path)])
    outputs = _output_images(rendered.out_dir)
    assert len(outputs) == 12
    for path in outputs:
        image = sitk.ReadImage(str(path))
        assert image.GetSpacing() == pytest.approx((3.0, 3.0, 3.0))
        array = sitk.GetArrayFromImage(image)
        # z-score over the whole image with clip to [-3, 3].
        assert abs(float(array.mean())) < 0.2, f"{path.name} mean {array.mean():.3f}"
        assert 0.5 < float(array.std()) < 1.5, f"{path.name} std {array.std():.3f}"
        assert float(array.min()) >= -3.0 - 1e-3
        assert float(array.max()) <= 3.0 + 1e-3


@pytest.mark.integration
@pytest.mark.skipif(
    (shutil.which("elastix") is None
     and not Path("/workspace/tools/bin/elastix").exists())
    or not (REPO_ROOT / "demo_data" / "Par0040affine.txt").is_file(),
    reason="elastix binary or demo_data/Par0040affine.txt parameter file unavailable",
)
def test_preprocess_elastix_registration(synthetic_tree: SyntheticTree, render_config) -> None:
    """Elastix registration variant (runs only when an elastix binary exists)."""
    rendered: RenderedConfig = render_config(
        "preprocess_registration.yaml",
        "preprocess_registration",
        synthetic_tree,
        {"@PAR_FILE@": (REPO_ROOT / "demo_data" / "Par0040affine.txt").as_posix()},
    )
    run_cli(CliRunner(), ["preprocess", "-c", str(rendered.path)])
    assert _output_images(rendered.out_dir)


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("dcm2niix") is None
    and not Path("/workspace/tools/bin/dcm2niix").exists(),
    reason="dcm2niix binary unavailable on this Linux image (only tools/bin/dcm2niix.exe ships; dcm2niix is on the do-not-install list)",
)
def test_preprocess_dcm2nii(synthetic_tree: SyntheticTree, render_config, tmp_path: Path) -> None:
    """dcm2nii conversion variant (runs only when a dcm2niix binary exists)."""
    rendered: RenderedConfig = render_config(
        "preprocess_dcm2nii.yaml", "preprocess_dcm2nii", synthetic_tree
    )
    run_cli(CliRunner(), ["preprocess", "-c", str(rendered.path)])
    assert _output_images(rendered.out_dir)
