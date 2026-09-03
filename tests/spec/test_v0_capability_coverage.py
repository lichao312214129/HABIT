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
"""Does v1 still do everything v0.1 did?

Structural tests written against the v1 implementation are self-consistent by
construction: they can pass in full while an entire v0.1 capability has no v1
home at all. That is exactly how per-supervoxel radiomics went missing --
``Supervoxelizer`` hard-coded mean aggregation, the legacy adapter emitted the
name ``supervoxel_radiomics``, and nothing ever checked that a component of
that name could be built.

These tests close that gap by working from the OUTSIDE IN: take the shipped
v0.1 configuration templates, translate each one, and require that every
component name the translation produces is actually registered and
constructible. A capability that v1 cannot express fails here, loudly, with
the config that needs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest
import yaml

from habit.spec.legacy import LegacyConfigAdapter
from habit.spec.specs import HabitatSpec

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "habitat"


def _registry_for(domain: str) -> Any:
    """
    Return the component registry serving one v1 domain.

    Args:
        domain: Registry domain name.

    Returns:
        The registry class.
    """
    from habit.api.plugins import _registry_for_domain

    return _registry_for_domain(domain)


#: Domains a translated habitat spec can name, in pipeline order.
_SPEC_DOMAINS: Tuple[str, ...] = (
    "voxel_feature_extractor",
    "supervoxelizer",
    "supervoxel_feature_extractor",
    "habitat_model_fitter",
    "habitat_assigner",
)

#: v0.1 capabilities with no v1 component yet, each tied to the configs that
#: need it. Entries are asserted to STILL be missing, so removing one is
#: mandatory when the component lands -- the list can only shrink.
_KNOWN_GAPS: Dict[Tuple[str, str], str] = {}


def _habitat_configs() -> List[Path]:
    """Return every shipped v0.1 habitat configuration template."""
    return sorted(_CONFIG_DIR.glob("*.yaml"))


def _translated_spec(path: Path) -> Optional[Mapping[str, Any]]:
    """
    Translate one v0.1 config and return its spec section.

    Args:
        path: Configuration file to translate.

    Returns:
        The spec payload, or ``None`` when the config declares none (predict
        stubs take their definition from the fitted model artefact).
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return LegacyConfigAdapter().translate(payload, "habitat").document.get("spec")


@pytest.mark.unit
def test_config_templates_are_discoverable() -> None:
    """Guard the guard: an empty glob would make every test below vacuous."""
    assert len(_habitat_configs()) >= 20


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_path", _habitat_configs(), ids=lambda p: p.stem
)
def test_translated_component_names_are_registered(config_path: Path) -> None:
    """Every component a translated v0.1 config names must be constructible."""
    spec = _translated_spec(config_path)
    if spec is None:
        pytest.skip(f"{config_path.name} declares no spec section.")

    missing: List[str] = []
    for domain in _SPEC_DOMAINS:
        entry = spec.get(domain)
        if entry is None:
            continue
        name = str(entry["name"])
        if (domain, name) in _KNOWN_GAPS:
            continue
        available = _registry_for(domain).available()
        if name not in available:
            missing.append(
                f"{domain}={name!r} is not registered (available: {available})"
            )
    assert not missing, (
        f"{config_path.name} translates to components v1 cannot build: {missing}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_path", _habitat_configs(), ids=lambda p: p.stem
)
def test_translated_spec_rebuilds_as_a_habitat_spec(config_path: Path) -> None:
    """The translation must be a valid HabitatSpec, not just a dict."""
    spec = _translated_spec(config_path)
    if spec is None:
        pytest.skip(f"{config_path.name} declares no spec section.")
    rebuilt = HabitatSpec.from_dict(spec).to_dict()
    for domain in ("supervoxelizer", "supervoxel_feature_extractor"):
        original = spec.get(domain)
        if original is None:
            assert rebuilt[domain] is None
            continue
        # ``to_dict`` fills in the schema version the translation omits;
        # compare the parts the translation is responsible for.
        assert rebuilt[domain]["name"] == original["name"]
        assert rebuilt[domain]["params"] == original.get("params", {})


@pytest.mark.unit
@pytest.mark.parametrize(
    "gap", sorted(_KNOWN_GAPS), ids=lambda g: f"{g[0]}.{g[1]}"
)
def test_known_gaps_are_still_gaps(gap: Tuple[str, str]) -> None:
    """A closed gap must be deleted from the list, not left to rot.

    Without this, ``_KNOWN_GAPS`` would silently keep excusing components
    that already exist, and the coverage test above would stop checking them.
    """
    domain, name = gap
    assert name not in _registry_for(domain).available(), (
        f"{domain}.{name} is now registered: remove it from _KNOWN_GAPS so "
        "the coverage test starts enforcing it."
    )


@pytest.mark.unit
def test_two_step_configs_translate_both_supervoxel_axes() -> None:
    """Growing supervoxels and describing them are separate v0.1 choices.

    ``habitat_segmentation.supervoxel`` (algorithm) and
    ``feature_construction.supervoxel_level`` (features) are independent
    blocks; fusing them into one spec loses whichever one does not win the
    name. This asserts they land in two different domains.
    """
    radiomics_config = _CONFIG_DIR / "config_habitat_two_step_supervoxel_radiomics_train.yaml"
    spec = _translated_spec(radiomics_config)
    assert spec is not None
    # The algorithm block chose kmeans; the feature block chose radiomics.
    assert spec["supervoxelizer"]["name"] == "kmeans"
    assert spec["supervoxel_feature_extractor"]["name"] == "supervoxel_radiomics"


@pytest.mark.unit
def test_gmm_supervoxel_config_keeps_its_algorithm() -> None:
    """A GMM supervoxel config must not be silently rewritten to kmeans."""
    spec = _translated_spec(
        _CONFIG_DIR / "config_habitat_two_step_supervoxel_gmm_train.yaml"
    )
    assert spec is not None
    assert spec["supervoxelizer"]["name"] == "gmm"
    # Its supervoxel_level is the default mean, which needs no extra step.
    assert spec["supervoxel_feature_extractor"] is None


@pytest.mark.unit
def test_one_step_configs_declare_no_supervoxel_stage() -> None:
    """One-step clusters voxels directly: both supervoxel slots stay empty."""
    spec = _translated_spec(
        _CONFIG_DIR / "config_habitat_one_step_elbow_train.yaml"
    )
    assert spec is not None
    assert spec["supervoxelizer"] is None
    assert spec["supervoxel_feature_extractor"] is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_path", _habitat_configs(), ids=lambda p: p.stem
)
def test_translated_preprocessing_chains_are_constructible(
    config_path: Path,
) -> None:
    """Every preprocessing step a config names must build into a real chain.

    The failure this prevents is subtler than a missing component: a spec field
    can be populated correctly by the translator, serialised faithfully, and
    still have no consumer, in which case the analysis silently skips
    normalisation and produces plausible-looking habitats from unnormalised
    features. Building the chains here proves the names resolve; the pipeline
    tests prove they are then applied.
    """
    spec = _translated_spec(config_path)
    if spec is None:
        pytest.skip(f"{config_path.name} declares no spec section.")

    from habit.feature_preprocessing import (
        CohortPreprocessingChain,
        SubjectPreprocessingChain,
        build_methods,
    )
    from habit.spec.specs import Spec

    for field, chain_type in (
        ("voxel_feature_preprocessors", SubjectPreprocessingChain),
        ("supervoxel_feature_preprocessors", SubjectPreprocessingChain),
        ("cohort_feature_preprocessors", CohortPreprocessingChain),
    ):
        steps = spec.get(field) or []
        if not steps:
            continue
        chain = chain_type(build_methods([Spec.from_dict(step) for step in steps]))
        # Imputation is prepended, so the chain is one step longer than the
        # configuration -- and must carry every configured step in order.
        configured = [str(step["name"]) for step in steps]
        actual = [method.spec.name for method in chain.methods]
        assert actual[-len(configured):] == configured, (
            f"{config_path.name}: {field} lost or reordered steps "
            f"({configured} -> {actual})."
        )


@pytest.mark.unit
def test_v01_preprocessing_blocks_reach_their_v1_chains() -> None:
    """v0.1's two preprocessing blocks must land in the right v1 chains.

    Order matters scientifically and the two blocks are not interchangeable:
    the subject-level block must stay stateless (it removes between-subject
    variation) and the group-level block must stay stateful (it makes subjects
    comparable). Swapping them would still run and still produce habitats.
    """
    payload = yaml.safe_load(
        (_CONFIG_DIR / "config_habitat_two_step.yaml").read_text(encoding="utf-8")
    )
    v0_construction = payload["feature_construction"]
    v0_subject = [
        entry["method"]
        for entry in (v0_construction["preprocessing_for_subject_level"]["methods"])
    ]
    v0_group = [
        entry["method"]
        for entry in (v0_construction["preprocessing_for_group_level"]["methods"])
    ]

    spec = LegacyConfigAdapter().translate(payload, "habitat").document["spec"]
    assert [s["name"] for s in spec["voxel_feature_preprocessors"]] == v0_subject
    assert [s["name"] for s in spec["cohort_feature_preprocessors"]] == v0_group


@pytest.mark.unit
def test_subject_pipeline_exposes_a_slot_for_every_preprocessing_chain() -> None:
    """A spec field with no pipeline slot is a silent disconnection.

    This test exists because the three chains were once specified, translated
    and serialised while ``SubjectPipeline`` had no parameter that could
    receive them.
    """
    import inspect

    from habit.pipeline import SubjectPipeline

    parameters = set(inspect.signature(SubjectPipeline.__init__).parameters)
    for spec_field, pipeline_slot in (
        ("voxel_feature_preprocessors", "voxel_feature_preprocessor"),
        ("supervoxel_feature_preprocessors", "supervoxel_feature_preprocessor"),
        ("cohort_feature_preprocessors", "cohort_feature_preprocessor"),
    ):
        assert pipeline_slot in parameters, (
            f"HabitatSpec.{spec_field} has no SubjectPipeline slot; a "
            "configured chain would be recorded and never applied."
        )
