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

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

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


@pytest.mark.unit
def test_extract_habitat_features_uses_domain_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Built-in feature_types run through the domain extract helper."""
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
def test_extract_habitat_features_falls_back_for_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional plugin feature types keep the compat analyzer path."""
    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock(run_id="compat-run", metadata={"engine": "compat"})

    monkeypatch.setattr(
        "habit.recipes.features._run_compat_extract", _spy
    )

    config = _feature_config(["msi", "graph"])
    plugins = {"graph": {"enabled": True}}
    logger = MagicMock()

    result = extract_habitat_features(
        config, plugin_configs=plugins, logger=logger
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["plugin_configs"] == plugins
    assert result.run_id == "compat-run"


@pytest.mark.unit
def test_traditional_radiomics_delegates_to_public_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The radiomics recipe forwards config to habit.api.habitat.run_radiomics."""
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
