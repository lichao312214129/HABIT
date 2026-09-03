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
"""Contract tests for the v1.0 ComponentRegistry surface."""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel, Field

from habit.api.exceptions import ComponentNotFoundError, ConfigurationError
from habit.exceptions import HABITAPIError
from habit.registry.base import ClassRegistry, _BaseRegistry
from habit.registry.core import ComponentRegistry


class _DummyParams(BaseModel):
    """Params schema for the throwaway test component."""

    threshold: float = Field(default=0.5, ge=0.0)
    mode: str = "fast"


class _DemoRegistry(ComponentRegistry):
    """Throwaway registry; storage is isolated per subclass."""

    domain = "demo_component"
    kind = "demo component"


class _OtherRegistry(ComponentRegistry):
    """Second throwaway registry used to prove storage isolation."""

    domain = "other_component"
    kind = "other component"


@_DemoRegistry.register("alpha", params_model=_DummyParams)
class _AlphaComponent:
    """Minimal component honouring the params constructor convention."""

    def __init__(self, threshold: float = 0.5, mode: str = "fast") -> None:
        self.threshold = threshold
        self.mode = mode


@_DemoRegistry.register("beta")
class _BetaComponent:
    """Component registered without a params schema."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

@pytest.mark.unit
def test_component_registry_subclasses_shared_base() -> None:
    """ComponentRegistry builds on the uniform v0.1 registry core."""
    assert issubclass(ComponentRegistry, ClassRegistry)
    assert issubclass(_DemoRegistry, _BaseRegistry)


@pytest.mark.unit
def test_create_validates_and_constructs() -> None:
    """Params pass through the Pydantic schema before construction."""
    component = _DemoRegistry.create("alpha", threshold="0.25", mode="slow")
    assert component.threshold == pytest.approx(0.25)
    assert component.mode == "slow"
    defaulted = _DemoRegistry.create("alpha")
    assert defaulted.threshold == pytest.approx(0.5)


@pytest.mark.unit
def test_create_unknown_name_raises_component_not_found() -> None:
    """Unknown names fail with the public, documented error type."""
    with pytest.raises(ComponentNotFoundError):
        _DemoRegistry.create("gamma")


@pytest.mark.unit
def test_create_invalid_params_raise_configuration_error() -> None:
    """Schema violations surface as ConfigurationError at the call site."""
    with pytest.raises(ConfigurationError):
        _DemoRegistry.create("alpha", threshold=-1.0)
    with pytest.raises(ConfigurationError):
        _DemoRegistry.create("alpha", threshold="not-a-number")


@pytest.mark.unit
def test_create_without_params_model_passes_kwargs_through() -> None:
    """Components without a schema receive the raw kwargs (arbitrary objects)."""
    marker = object()
    component = _DemoRegistry.create("beta", payload=marker)
    assert component.kwargs["payload"] is marker


@pytest.mark.unit
def test_constructor_signature_is_registry_parameter_contract() -> None:
    """Registry inspection and create() share one Python constructor signature."""
    signature = _DemoRegistry.constructor_signature("alpha")
    assert isinstance(signature, inspect.Signature)
    assert tuple(signature.parameters) == ("threshold", "mode")
    with pytest.raises(ConfigurationError, match="unexpected keyword"):
        _DemoRegistry.create("alpha", unsupported=True)


@pytest.mark.unit
def test_available_is_sorted_and_params_model_lookup() -> None:
    """Introspection returns deterministic names and the registered schema."""
    assert _DemoRegistry.available() == ("alpha", "beta")
    assert _DemoRegistry.params_model("alpha") is _DummyParams
    assert _AlphaComponent.__habit_params_model__ is _DummyParams
    assert _DemoRegistry.params_model("beta") is None
    assert _DemoRegistry.params_model("missing") is None


@pytest.mark.unit
def test_entry_point_group_follows_domain() -> None:
    """The third-party registration group is ``habit.<domain>``."""
    assert _DemoRegistry.entry_point_group() == "habit.demo_component"


@pytest.mark.unit
def test_registries_do_not_share_storage() -> None:
    """Sibling ComponentRegistry subclasses own independent mappings."""
    assert _OtherRegistry.available() == ()
    assert _OtherRegistry._registry is not _DemoRegistry._registry


@pytest.mark.unit
def test_builtin_domains_use_snake_case_protocol_names() -> None:
    """The five domain registries obey the singular snake_case convention."""
    from habit.habitat_model.assignment import HabitatAssignerRegistry
    from habit.habitat_features import HabitatFeatureExtractorRegistry
    from habit.habitat_model import HabitatModelFitterRegistry
    from habit.supervoxel import SupervoxelizerRegistry
    from habit.voxel_features import VoxelFeatureExtractorRegistry

    assert VoxelFeatureExtractorRegistry.domain == "voxel_feature_extractor"
    assert SupervoxelizerRegistry.domain == "supervoxelizer"
    assert HabitatModelFitterRegistry.domain == "habitat_model_fitter"
    assert HabitatAssignerRegistry.domain == "habitat_assigner"
    assert HabitatFeatureExtractorRegistry.domain == "habitat_feature_extractor"


@pytest.mark.unit
def test_builtin_registries_create_with_validation() -> None:
    """Built-in components construct through their registry with coercion."""
    from habit.supervoxel import SupervoxelizerRegistry

    slic = SupervoxelizerRegistry.create("slic", n_supervoxels=8)
    assert slic.n_supervoxels == 8
    with pytest.raises(HABITAPIError, match="n_supervoxels"):
        SupervoxelizerRegistry.create("slic", n_supervoxels="8")
    with pytest.raises(HABITAPIError):
        SupervoxelizerRegistry.create("slic", n_supervoxels=-1)
    with pytest.raises(ComponentNotFoundError):
        SupervoxelizerRegistry.create("watershed")
