"""Tests for stable image geometry and low-level radiomics contracts."""

from __future__ import annotations

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
    pytest.importorskip("SimpleITK")
    pytest.importorskip("radiomics")
    from habit.core.habitat_analysis.clustering_features.voxel_radiomics_extractor import (
        _feature_values_in_mask,
    )

    feature_map = np.array([[0.0, -2.0], [3.5, 99.0]], dtype=np.float32)
    mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)

    values = _feature_values_in_mask(feature_map, mask)

    np.testing.assert_array_equal(values, np.array([0.0, -2.0, 3.5]))


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
