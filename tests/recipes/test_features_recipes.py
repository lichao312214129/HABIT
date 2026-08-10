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

"""Fast tests for L4 feature-extraction recipes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from habit.exceptions import HABITAPIError
from habit.recipes import extract_habitat_features, traditional_radiomics
from habit.schemas.workflows.habitat import FeatureExtractionConfig


def _feature_config(feature_types: List[str]) -> FeatureExtractionConfig:
    """Build a minimal validated extract config for unit tests."""
    return FeatureExtractionConfig.model_construct(
        raw_img_folder="/tmp/images",
        habitats_map_folder="/tmp/habitats",
        out_dir="/tmp/out",
        feature_types=feature_types,
        n_processes=1,
        habitat_pattern="*_habitats.nrrd",
        n_habitats=2,
        debug=False,
    )


def _synthetic_extract_dataset(tmp_path: Path) -> Dict[str, Any]:
    """
    Create a one-subject synthetic extract dataset (raw image + habitat map).

    Args:
        tmp_path: Test-private temporary directory.

    Returns:
        Mapping accepted by ``FeatureExtractionConfig.model_validate``.
    """
    import numpy as np
    import SimpleITK as sitk

    raw_root = tmp_path / "raw"
    habitats_dir = tmp_path / "habitats"
    image_dir = raw_root / "images" / "sub001" / "delay2"
    image_dir.mkdir(parents=True)
    habitats_dir.mkdir(parents=True)

    intensity = np.linspace(10, 100, num=125, dtype=np.float32).reshape((5, 5, 5))
    sitk.WriteImage(
        sitk.GetImageFromArray(intensity), str(image_dir / "delay2.nii.gz")
    )

    labels = np.zeros((5, 5, 5), dtype=np.uint32)
    labels[1:3, 1:3, 1:3] = 1
    labels[3:5, 1:3, 1:3] = 2
    sitk.WriteImage(
        sitk.GetImageFromArray(labels), str(habitats_dir / "sub001_habitats.nrrd")
    )

    return {
        "raw_img_folder": str(raw_root),
        "habitats_map_folder": str(habitats_dir),
        "out_dir": str(tmp_path / "features_out"),
        "n_processes": 1,
        "habitat_pattern": "*_habitats.nrrd",
        "feature_types": ["graph"],
        "n_habitats": 2,
        "debug": False,
    }


class _BlockedImportFinder:
    """
    Meta-path finder that fails specific module imports.

    Raising inside ``find_spec`` propagates as the import error, which is
    exactly what a regression importing the blocked module would hit.
    """

    def __init__(self, blocked: Tuple[str, ...]) -> None:
        self._blocked = blocked

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname in self._blocked:
            raise ImportError(f"blocked import for test: {fullname}")
        return None


@pytest.fixture
def no_compat_graph_loader_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[str, ...]:
    """
    Make the deprecated compat graph loaders un-importable for one test.

    The modules are also evicted from ``sys.modules`` so a prior import by
    another test cannot mask a fresh import on the domain path.

    Yields:
        The blocked module names.
    """
    blocked = (
        "habit.compat.feature_extraction_loader",
        "habit.compat.graph_plugin",
    )
    for name in blocked:
        monkeypatch.delitem(sys.modules, name, raising=False)
    finder = _BlockedImportFinder(blocked)
    sys.meta_path.insert(0, finder)
    try:
        yield blocked
    finally:
        sys.meta_path.remove(finder)


@pytest.mark.unit
def test_extract_habitat_features_uses_domain_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Built-in feature_types run through the domain extract helper."""
    from unittest.mock import MagicMock

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock(run_id="test-run", metadata={"engine": "domain"})

    monkeypatch.setattr(
        "habit.recipes.features._run_domain_extract", _spy
    )

    config = _feature_config(["msi", "ith_score", "non_radiomics"])
    logger = MagicMock()

    result = extract_habitat_features(config, logger=logger)

    assert len(calls) == 1
    assert calls[0]["args"][0] is config
    assert calls[0]["kwargs"]["logger"] is logger
    assert result.run_id == "test-run"


@pytest.mark.unit
def test_extract_habitat_features_falls_back_for_legacy_only_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Names known only to the v0.1 factory keep the compat analyzer path."""
    from unittest.mock import MagicMock

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock(run_id="compat-run", metadata={"engine": "compat"})

    monkeypatch.setattr(
        "habit.recipes.features._run_compat_extract", _spy
    )
    # Simulate a legacy-only plugin: absent from the domain registry but
    # provided by the v0.1 HabitatFeatureFactory.
    monkeypatch.setattr(
        "habit.recipes.features._legacy_feature_type_names",
        lambda: frozenset({"future_plugin"}),
    )

    config = _feature_config(["msi", "future_plugin"])
    plugins = {"future_plugin": {"enabled": True}}
    logger = MagicMock()

    result = extract_habitat_features(
        config, plugin_configs=plugins, logger=logger
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["plugin_configs"] == plugins
    assert result.run_id == "compat-run"


@pytest.mark.unit
def test_extract_habitat_features_rejects_unknown_feature_types() -> None:
    """Names registered nowhere fail fast with a precise error."""
    config = _feature_config(["msi", "not_a_feature_family"])

    with pytest.raises(HABITAPIError, match="not_a_feature_family"):
        extract_habitat_features(config)


@pytest.mark.unit
def test_extract_habitat_features_routes_domain_registered_plugins_to_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Third-party names in the domain registry dispatch like built-ins."""
    from unittest.mock import MagicMock

    from habit.domain.habitat_features import HabitatFeatureExtractorRegistry

    class _FakeDomainExtractor:
        def __call__(self, subject: Any, habitat_map: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock(run_id="test-run", metadata={"engine": "domain"})

    monkeypatch.setattr(
        "habit.recipes.features._run_domain_extract", _spy
    )

    HabitatFeatureExtractorRegistry.register("fake_domain_family")(
        _FakeDomainExtractor
    )
    try:
        config = _feature_config(["fake_domain_family"])
        result = extract_habitat_features(config, logger=MagicMock())
    finally:
        HabitatFeatureExtractorRegistry._registry.pop("fake_domain_family", None)

    assert len(calls) == 1
    assert result.run_id == "test-run"


@pytest.mark.unit
def test_extract_habitat_features_routes_graph_to_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The built-in ``graph`` family runs through the domain extract helper."""
    from unittest.mock import MagicMock

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock(run_id="test-run", metadata={"engine": "domain"})

    monkeypatch.setattr(
        "habit.recipes.features._run_domain_extract", _spy
    )

    config = _feature_config(["msi", "graph"])
    plugins = {"graph": {"distance_threshold": 8.0}}
    logger = MagicMock()

    result = extract_habitat_features(
        config, plugin_configs=plugins, logger=logger
    )

    assert len(calls) == 1
    assert calls[0]["args"][0] is config
    assert calls[0]["kwargs"]["plugin_configs"] == plugins
    assert result.run_id == "test-run"


@pytest.mark.unit
def test_build_domain_extractors_applies_graph_plugin_params() -> None:
    """Graph plugin settings reach the domain extractor constructor."""
    from habit.domain.habitat_features import GraphHabitatFeatures
    from habit.recipes.features import _build_domain_extractors

    extractors = _build_domain_extractors(
        ["graph"],
        params_file_of_non_habitat=None,
        params_file_of_habitat=None,
        plugin_configs={"graph": {"distance_threshold": 8.0, "erosion_radius": 0}},
    )

    extractor = extractors["graph"]
    assert isinstance(extractor, GraphHabitatFeatures)
    assert extractor.spec.params["distance_threshold"] == 8.0
    assert extractor.spec.params["erosion_radius"] == 0


@pytest.mark.unit
def test_graph_params_from_plugin_configs_drops_visualization_keys() -> None:
    """Visualization/legacy block keys never reach the extractor constructor."""
    from habit.recipes.features import _graph_params_from_plugin_configs
    from habit.schemas.workflows.habitat import GraphFeatureBlock

    block = GraphFeatureBlock.model_validate(
        {
            "distance_threshold": 8.0,
            "visualize": True,
            "visualization_format": "png",
            "visualization_dpi": 300,
            "visualization_show_background": False,
            "visualization_save_3d": False,
            "enabled": True,
            "n_workers": 4,
        }
    )
    params = _graph_params_from_plugin_configs({"graph": block})

    assert params["distance_threshold"] == 8.0
    assert "visualize" not in params
    assert "n_workers" not in params
    assert "enabled" not in params
    assert not any(key.startswith("visualization_") for key in params)

    # Plain mappings from direct API callers are filtered the same way.
    mapping_params = _graph_params_from_plugin_configs(
        {"graph": {"distance_threshold": 6.0, "visualize": True}}
    )
    assert mapping_params == {"distance_threshold": 6.0}

    assert _graph_params_from_plugin_configs(None) == {}
    assert _graph_params_from_plugin_configs({}) == {}


@pytest.mark.unit
def test_graph_block_from_plugin_configs_coercion() -> None:
    """The figure hook's block view accepts blocks, mappings, and shims."""
    from habit.domain.habitat_features.graph import GraphHabitatFeaturesParams
    from habit.recipes.features import _graph_block_from_plugin_configs
    from habit.schemas.workflows.habitat import GraphFeatureBlock

    block = GraphFeatureBlock(visualize=True, visualization_format="png")
    assert _graph_block_from_plugin_configs({"graph": block}) is block

    coerced = _graph_block_from_plugin_configs({"graph": {"visualize": True}})
    assert isinstance(coerced, GraphFeatureBlock)
    assert coerced.visualize is True

    # The deprecated compat shim passes the extraction-only params model; it
    # carries no visualization fields, so the figure hook stays off.
    legacy = GraphHabitatFeaturesParams(distance_threshold=8.0)
    shimmed = _graph_block_from_plugin_configs({"graph": legacy})
    assert isinstance(shimmed, GraphFeatureBlock)
    assert shimmed.visualize is False
    assert shimmed.distance_threshold == 8.0

    assert _graph_block_from_plugin_configs(None) is None
    assert _graph_block_from_plugin_configs({}) is None


@pytest.mark.unit
def test_graph_feature_block_mirrors_domain_params() -> None:
    """The YAML block's extraction fields mirror the domain params model."""
    from habit.domain.habitat_features.graph import GraphHabitatFeaturesParams
    from habit.schemas.workflows.habitat import GraphFeatureBlock

    extraction = set(GraphHabitatFeaturesParams.model_fields)
    block_fields = set(GraphFeatureBlock.model_fields)
    assert extraction <= block_fields
    for field in extraction:
        assert (
            GraphFeatureBlock.model_fields[field].default
            == GraphHabitatFeaturesParams.model_fields[field].default
        ), f"default drift on {field}"


@pytest.mark.integration
def test_domain_extract_graph_writes_visualizations(tmp_path: Path) -> None:
    """``graph.visualize: true`` renders per-subject figures after the CSV."""
    pytest.importorskip("matplotlib")

    import logging

    config_dict = _synthetic_extract_dataset(tmp_path)
    config_dict["graph"] = {
        # The 2x2x2 synthetic regions would not survive the default erosion;
        # disabling it keeps graph nodes (and therefore 3D renders) non-empty.
        "erosion_radius": 0,
        "visualize": True,
        "visualization_format": "png",
        "visualization_dpi": 72,
        "visualization_save_3d": True,
    }

    result = extract_habitat_features(
        config_dict, logger=logging.getLogger("test.graph_viz")
    )

    out_dir = Path(result.output_dir)
    assert (out_dir / "habitat_graph_features.csv").is_file()

    figure_dir = out_dir / "visualizations" / "graph"
    slice_png = figure_dir / "sub001_graph_slice.png"
    network_png = figure_dir / "sub001_graph_network_2d.png"
    assert slice_png.is_file() and slice_png.stat().st_size > 0
    assert network_png.is_file() and network_png.stat().st_size > 0

    try:
        import pyvista  # noqa: F401
        import skimage  # noqa: F401
    except ImportError:
        # 3D rendering is optional: missing backends skip with a warning.
        assert not (figure_dir / "sub001_graph_surface_3d.png").exists()
    else:
        surface_png = figure_dir / "sub001_graph_surface_3d.png"
        network_3d_png = figure_dir / "sub001_graph_network_3d.png"
        assert surface_png.is_file() and surface_png.stat().st_size > 0
        assert network_3d_png.is_file() and network_3d_png.stat().st_size > 0


@pytest.mark.integration
def test_domain_extract_graph_without_visualize_writes_no_figures(
    tmp_path: Path,
) -> None:
    """Default settings keep the run figure-free (v0.1 visualize=false)."""
    config_dict = _synthetic_extract_dataset(tmp_path)

    result = extract_habitat_features(config_dict)

    out_dir = Path(result.output_dir)
    assert (out_dir / "habitat_graph_features.csv").is_file()
    assert not (out_dir / "visualizations").exists()


@pytest.mark.integration
def test_graph_extract_full_path_never_imports_compat(
    tmp_path: Path,
    no_compat_graph_loader_imports: Tuple[str, ...],
) -> None:
    """The graph YAML->CSV->figure path works with compat loaders blocked."""
    pytest.importorskip("matplotlib")

    import logging

    config_dict = _synthetic_extract_dataset(tmp_path)
    config_dict["graph"] = {
        "distance_threshold": 8.0,
        "visualize": True,
        "visualization_format": "png",
        "visualization_dpi": 72,
        "visualization_save_3d": False,
    }

    result = extract_habitat_features(
        config_dict, logger=logging.getLogger("test.graph_no_compat")
    )

    out_dir = Path(result.output_dir)
    assert (out_dir / "habitat_graph_features.csv").is_file()
    assert (out_dir / "visualizations" / "graph" / "sub001_graph_slice.png").is_file()
    for name in no_compat_graph_loader_imports:
        assert name not in sys.modules


@pytest.mark.unit
def test_traditional_radiomics_delegates_to_public_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The radiomics recipe forwards config to habit.api.habitat.run_radiomics."""
    from unittest.mock import MagicMock

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock(run_id="radiomics-run")

    monkeypatch.setattr("habit.api.habitat.run_radiomics", _spy)

    config = MagicMock(out_dir="/tmp/radiomics")
    logger = MagicMock()

    result = traditional_radiomics(config, logger=logger)

    assert len(calls) == 1
    assert calls[0]["args"][0] is config
    assert calls[0]["kwargs"]["logger"] is logger
    assert result.run_id == "radiomics-run"


@pytest.mark.unit
def test_recipes_module_exports_feature_symbols() -> None:
    """New recipes are registered on habit.recipes for CLI and notebooks."""
    from habit import recipes

    assert callable(recipes.extract_habitat_features)
    assert callable(recipes.traditional_radiomics)
