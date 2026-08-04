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
"""Tests for stable image geometry and low-level radiomics contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from habit.api.exceptions import GeometryError
from habit.image import (
    GeometryPolicy,
    ImageMaskPair,
    ImageVolume,
    MaskVolume,
    align_image_mask,
    validate_geometry,
)
from habit.radiomics import extract_features


@pytest.mark.unit
def test_validate_geometry_reports_physical_metadata_mismatch() -> None:
    """Geometry validation must distinguish compatible arrays from compatible space."""
    image = ImageVolume.from_array(
        np.ones((2, 2, 2), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
    )
    mask = MaskVolume.from_array(
        np.ones((2, 2, 2), dtype=np.uint8),
        spacing=(1.0, 1.0, 2.0),
    )

    report = validate_geometry(image, mask)

    assert not report.compatible
    assert report.mismatches == ("spacing",)


@pytest.mark.unit
def test_align_image_mask_strictly_rejects_mismatch() -> None:
    """Strict alignment must never silently rewrite mask spatial metadata."""
    image = ImageVolume.from_array(np.ones((2, 2), dtype=np.float32))
    mask = MaskVolume.from_array(
        np.ones((3, 3), dtype=np.uint8),
        spacing=(2.0, 2.0),
    )

    with pytest.raises(GeometryError, match="shape, spacing"):
        align_image_mask(ImageMaskPair(image, mask))


@pytest.mark.unit
def test_low_level_radiomics_returns_features_and_provenance() -> None:
    """The public component API separates scalar features from diagnostics."""
    pytest.importorskip("SimpleITK")

    class FakeExtractor:
        """Minimal PyRadiomics-compatible test double."""

        def execute(self, **_: Any) -> dict[str, Any]:
            return {
                "original_firstorder_Mean": np.float64(2.5),
                "diagnostics_Versions_PyRadiomics": "test-version",
            }

    image = ImageVolume.from_array(np.ones((2, 2), dtype=np.float32))
    mask = MaskVolume.from_array(np.ones((2, 2), dtype=np.uint8))
    with patch(
        "habit.api.radiomics._create_pyradiomics_extractor",
        return_value=FakeExtractor(),
    ):
        result = extract_features(image, mask, params={"setting": {"binWidth": 25}})

    assert result.values == {"original_firstorder_Mean": 2.5}
    assert result.provenance["diagnostics_Versions_PyRadiomics"] == "test-version"
    assert result.geometry_report.compatible
    assert result.resolved_params == {"setting": {"binWidth": 25}}


@pytest.mark.unit
def test_warn_geometry_policy_records_incompatibility() -> None:
    """Warn mode is explicit and leaves the original physical metadata intact."""
    image = ImageVolume.from_array(np.ones((2, 2), dtype=np.float32))
    mask = MaskVolume.from_array(
        np.ones((2, 2), dtype=np.uint8),
        origin=(5.0, 0.0),
    )

    with pytest.warns(RuntimeWarning):
        pair = align_image_mask(
            ImageMaskPair(image, mask),
            policy=GeometryPolicy.WARN,
        )

    assert pair.mask.origin == (5.0, 0.0)
    assert pair.geometry_report is not None
    assert pair.geometry_report.action == "warn"


@pytest.mark.unit
def test_voxel_feature_selection_uses_mask_not_feature_threshold() -> None:
    """Zero and negative feature values inside an ROI must remain in the table."""
    from habit.kernels.radiomics.voxel_maps import feature_values_in_mask

    feature_map = np.array([[0.0, -2.0], [3.5, 99.0]], dtype=np.float32)
    mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)

    values = feature_values_in_mask(feature_map, mask)

    np.testing.assert_array_equal(values, np.array([0.0, -2.0, 3.5]))


@pytest.mark.unit
def test_cropped_voxel_feature_map_uses_physical_mask_alignment() -> None:
    """A cropped feature map must select the requested label on its physical grid."""
    sitk = pytest.importorskip("SimpleITK")
    from habit.kernels.radiomics.voxel_maps import (
        feature_values_in_mask,
        mask_array_for_feature_map,
    )

    mask_data = np.zeros((11, 11, 11), dtype=np.uint8)
    mask_data[4:7, 3:8, 2:6] = 2
    mask_data[9, 9, 9] = 3
    mask = sitk.GetImageFromArray(mask_data)
    mask.SetSpacing((0.8, 1.2, 2.5))
    mask.SetOrigin((12.0, -7.0, 30.0))
    mask.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))

    # RegionOfInterest reproduces the physical metadata behavior of the crop
    # returned by PyRadiomics while keeping this unit test fast and deterministic.
    cropped_reference = sitk.RegionOfInterest(mask, size=(6, 7, 5), index=(1, 2, 3))
    feature_data = np.arange(
        np.prod(cropped_reference.GetSize()),
        dtype=np.float32,
    ).reshape(tuple(reversed(cropped_reference.GetSize())))

    aligned_mask = mask_array_for_feature_map(mask, cropped_reference, label=2)
    values = feature_values_in_mask(feature_data, aligned_mask)

    assert aligned_mask.shape == feature_data.shape
    assert int(aligned_mask.sum()) == 60
    assert values.shape == (60,)


@pytest.mark.integration
def test_voxel_radiomics_accepts_real_pyradiomics_cropped_maps(
    tmp_path: Path,
) -> None:
    """Real PyRadiomics ROI crops must yield one feature row per mask voxel."""
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("radiomics")
    from habit.core.habitat_analysis.clustering_features.voxel_radiomics_extractor import (
        VoxelRadiomicsExtractor,
    )

    params_file = tmp_path / "voxel_firstorder.yaml"
    params_file.write_text(
        "\n".join(
            (
                "imageType:",
                "  Original: {}",
                "featureClass:",
                "  firstorder:",
                "    - Mean",
                "setting:",
                "  binWidth: 12",
            )
        ),
        encoding="utf-8",
    )

    image_data = np.arange(11**3, dtype=np.float32).reshape((11, 11, 11)) - 700.0
    mask_data = np.zeros((11, 11, 11), dtype=np.uint8)
    mask_data[4:7, 3:8, 2:6] = 1
    image = sitk.GetImageFromArray(image_data)
    mask = sitk.GetImageFromArray(mask_data)

    extractor = VoxelRadiomicsExtractor(params_file=str(params_file))
    features = extractor.extract_features(
        image,
        mask,
        image="test",
        subject="cropped-map-regression",
        kernel_radius=1,
        voxel_batch=1000,
        use_torch_radiomics=False,
    )

    assert features.shape == (60, 1)
    assert not features.isna().any().any()
    assert (features.iloc[:, 0] < 0).any()


@pytest.mark.unit
def test_plugin_entry_point_loader_invokes_registration_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Third-party plugins can self-register through a standard entry point."""
    import habit.api.plugins as plugins

    called: list[bool] = []

    class FakeEntryPoint:
        """Minimal importlib.metadata entry point used without package installation."""

        name = "demo"
        value = "demo_package:register"

        @staticmethod
        def load() -> Any:
            return lambda: called.append(True)

    monkeypatch.setattr(plugins, "_ENTRY_POINT_GROUPS", {"models": "habit.models"})
    monkeypatch.setattr(
        plugins,
        "_entry_points_for",
        lambda group: (FakeEntryPoint(),),
    )
    plugins._LOADED_ENTRY_POINTS.clear()

    report = plugins.load_plugins()

    assert called == [True]
    assert report.loaded == ("models:demo",)
    assert report.failures == {}


@pytest.mark.unit
def test_run_manifest_has_deterministic_config_hash_and_persists(tmp_path) -> None:
    """A workflow manifest must preserve resolved configuration and version context."""
    from habit.api.provenance import create_run_manifest, write_run_manifest

    config = {"out_dir": tmp_path / "results", "random_state": 42}
    first = create_run_manifest("radiomics", config)
    second = create_run_manifest("radiomics", config)
    manifest_path = write_run_manifest(first, tmp_path)

    assert first.config_hash == second.config_hash
    assert first.run_id != second.run_id
    assert manifest_path.is_file()
    assert "resolved_config" in manifest_path.read_text(encoding="utf-8")
