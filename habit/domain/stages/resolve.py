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
"""Resolve Stage roles from position + registry domain membership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from habit.domain.assignment.registry import HabitatAssignerRegistry
from habit.domain.feature_preprocessing.registry import (
    FeaturePreprocessingMethodRegistry,
)
from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.domain.habitat_model.registry import HabitatModelFitterRegistry
from habit.domain.pooling_marker.registry import PoolingRegistry
from habit.domain.postprocess import ConnectedComponentPostprocess
from habit.domain.supervoxel.registry import SupervoxelizerRegistry
from habit.domain.supervoxel_features.registry import (
    SupervoxelFeatureExtractorRegistry,
)
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.exceptions import HABITAPIError
from habit.spec.specs import (
    POOL_COMPONENT_NAME,
    ROLE_ASSIGN,
    ROLE_EXTRACT_SUPERVOXEL_FEATURES,
    ROLE_EXTRACT_VOXEL_FEATURES,
    ROLE_FIT,
    ROLE_PARTITION,
    ROLE_POOL,
    ROLE_POSTPROCESS_HABITAT,
    ROLE_POSTPROCESS_SUPERVOXEL,
    ROLE_PREPROCESS,
    ROLE_QUANTIFY,
    HabitatSpec,
    Stage,
)

__all__ = ["ResolvedStage", "resolve_habitat_stages", "design_from_stages"]

#: Domain -> default role when the name is unambiguous.
_DOMAIN_ROLE: Dict[str, str] = {
    "voxel_feature_extractor": ROLE_EXTRACT_VOXEL_FEATURES,
    "feature_preprocessing_method": ROLE_PREPROCESS,
    "supervoxelizer": ROLE_PARTITION,
    "supervoxel_feature_extractor": ROLE_EXTRACT_SUPERVOXEL_FEATURES,
    "pooling": ROLE_POOL,
    "habitat_model_fitter": ROLE_FIT,
    "habitat_assigner": ROLE_ASSIGN,
    "habitat_feature_extractor": ROLE_QUANTIFY,
}


@dataclass(frozen=True)
class ResolvedStage:
    """A stage whose scientific role has been determined."""

    name: str
    component: Stage  # original stage (keeps Spec)
    role: str
    domain: str


def _domains_for_name(name: str) -> List[str]:
    """Return registry domains that currently register ``name``."""
    # Ensure built-ins are discovered.
    import habit.domain.pooling_marker.pool  # noqa: F401
    import habit.domain.voxel_features  # noqa: F401
    import habit.domain.supervoxel  # noqa: F401
    import habit.domain.supervoxel_features  # noqa: F401
    import habit.domain.feature_preprocessing  # noqa: F401
    import habit.domain.habitat_model  # noqa: F401
    import habit.domain.assignment  # noqa: F401
    import habit.domain.habitat_features  # noqa: F401

    checks: Tuple[Tuple[str, type], ...] = (
        ("voxel_feature_extractor", VoxelFeatureExtractorRegistry),
        ("feature_preprocessing_method", FeaturePreprocessingMethodRegistry),
        ("supervoxelizer", SupervoxelizerRegistry),
        ("supervoxel_feature_extractor", SupervoxelFeatureExtractorRegistry),
        ("pooling", PoolingRegistry),
        ("habitat_model_fitter", HabitatModelFitterRegistry),
        ("habitat_assigner", HabitatAssignerRegistry),
        ("habitat_feature_extractor", HabitatFeatureExtractorRegistry),
    )
    found: List[str] = []
    for domain, registry in checks:
        if registry.get(name) is not None:
            found.append(domain)
    # Connected-component postprocess is not a plugin domain; recognise by
    # Spec name used in HabitatSpec sugar.
    if name in ("connected_component", "cc_postprocess"):
        found.append("postprocess")
    return found


def _looks_like_postprocess(stage: Stage) -> bool:
    """Return True when the stage is a connected-component postprocess Spec."""
    if stage.role in (ROLE_POSTPROCESS_SUPERVOXEL, ROLE_POSTPROCESS_HABITAT):
        return True
    return stage.component.name in (
        "connected_component",
        ConnectedComponentPostprocess.__name__,
    )


def _disambiguate(
    name: str,
    domains: Sequence[str],
    *,
    index: int,
    stages: Sequence[Stage],
    roles_so_far: Sequence[str],
) -> Tuple[str, str]:
    """
    Resolve a multi-domain component name using position rules.

    Dual-domain ``kmeans`` / ``gmm``:
    * before an upcoming supervoxel-feature stage (or before pool, no
      partition yet) → partition (supervoxelizer);
    * otherwise, before assign / after pool → fitter.
    """
    domain_set = set(domains)
    if domain_set <= {"supervoxelizer", "habitat_model_fitter"} and domain_set == {
        "supervoxelizer",
        "habitat_model_fitter",
    }:
        later_names = [s.component.name for s in stages[index + 1 :]]
        later_roles = [
            s.role for s in stages[index + 1 :] if s.role is not None
        ]
        has_svx_feat_later = (
            ROLE_EXTRACT_SUPERVOXEL_FEATURES in later_roles
            or any(
                SupervoxelFeatureExtractorRegistry.get(n) is not None
                for n in later_names
            )
        )
        seen_partition = ROLE_PARTITION in roles_so_far
        seen_pool = ROLE_POOL in roles_so_far
        if not seen_partition and not seen_pool and (
            has_svx_feat_later or ROLE_POOL in later_roles or POOL_COMPONENT_NAME in later_names
        ):
            return "supervoxelizer", ROLE_PARTITION
        if seen_pool or seen_partition or ROLE_ASSIGN in later_roles:
            return "habitat_model_fitter", ROLE_FIT
        # Default: fitter when assign follows closely; else complain.
        if any(
            HabitatAssignerRegistry.get(n) is not None for n in later_names
        ):
            return "habitat_model_fitter", ROLE_FIT
        raise HABITAPIError(
            f"Component {name!r} is registered as both supervoxelizer and "
            "habitat_model_fitter, and its position is ambiguous. Place it "
            "before extract_supervoxel_features (partition) or immediately "
            "before assign after pool (fitter)."
        )
    if len(domains) == 1:
        domain = domains[0]
        if domain == "postprocess":
            # Position: after partition / before pool → supervoxel CC;
            # after assign → habitat CC.
            if ROLE_ASSIGN in roles_so_far:
                return "postprocess", ROLE_POSTPROCESS_HABITAT
            return "postprocess", ROLE_POSTPROCESS_SUPERVOXEL
        role = _DOMAIN_ROLE.get(domain)
        if role is None:
            raise HABITAPIError(
                f"No stage role mapping for domain {domain!r} "
                f"(component {name!r})."
            )
        return domain, role
    raise HABITAPIError(
        f"Component {name!r} matches multiple plugin domains {list(domains)} "
        "and no disambiguation rule applies. Rename one plugin or place the "
        "stage where its role is unique (see HabitatSpec.validate_dataflow / "
        "docs for kmeans partition-vs-fitter rules)."
    )


def resolve_habitat_stages(spec: HabitatSpec) -> Tuple[ResolvedStage, ...]:
    """
    Attach a role + domain to every stage in ``spec``.

    Args:
        spec: Habitat specification (sugar or explicit stages).

    Returns:
        Resolved stages in order.

    Raises:
        HABITAPIError: On unknown components, ambiguity, or illegal sequences.
    """
    stages = list(spec.resolved_stages())
    if not stages:
        raise HABITAPIError("HabitatSpec.stages is empty; nothing to resolve.")

    resolved: List[ResolvedStage] = []
    roles_so_far: List[str] = []
    for index, stage in enumerate(stages):
        if stage.role is not None:
            # Honour sugar / authored roles; still verify the component exists
            # in a matching domain when possible.
            domain_hint = {
                ROLE_EXTRACT_VOXEL_FEATURES: "voxel_feature_extractor",
                ROLE_PREPROCESS: "feature_preprocessing_method",
                ROLE_PARTITION: "supervoxelizer",
                ROLE_EXTRACT_SUPERVOXEL_FEATURES: "supervoxel_feature_extractor",
                ROLE_POOL: "pooling",
                ROLE_FIT: "habitat_model_fitter",
                ROLE_ASSIGN: "habitat_assigner",
                ROLE_QUANTIFY: "habitat_feature_extractor",
                ROLE_POSTPROCESS_SUPERVOXEL: "postprocess",
                ROLE_POSTPROCESS_HABITAT: "postprocess",
            }.get(stage.role, "unknown")
            resolved.append(
                ResolvedStage(
                    name=stage.name,
                    component=stage,
                    role=stage.role,
                    domain=domain_hint,
                )
            )
            roles_so_far.append(stage.role)
            continue

        if _looks_like_postprocess(stage):
            domain, role = _disambiguate(
                stage.component.name,
                ["postprocess"],
                index=index,
                stages=stages,
                roles_so_far=roles_so_far,
            )
        else:
            domains = _domains_for_name(stage.component.name)
            if not domains:
                raise HABITAPIError(
                    f"Stage {stage.name!r} uses unknown component "
                    f"{stage.component.name!r}. Register it under a habit.* "
                    "entry-point group, or check the name against "
                    "list_plugins()."
                )
            domain, role = _disambiguate(
                stage.component.name,
                domains,
                index=index,
                stages=stages,
                roles_so_far=roles_so_far,
            )
        resolved.append(
            ResolvedStage(
                name=stage.name,
                component=stage,
                role=role,
                domain=domain,
            )
        )
        roles_so_far.append(role)

    _validate_role_sequence(tuple(r.role for r in resolved))
    return tuple(resolved)


def _validate_role_sequence(roles: Sequence[str]) -> None:
    """Reject illegal role sequences with actionable messages."""
    has_partition = ROLE_PARTITION in roles
    has_pool = ROLE_POOL in roles
    if has_partition and not has_pool:
        raise HABITAPIError(
            "Stage sequence includes partition but no pool: per-subject "
            "definition on supervoxels is not supported. Add "
            "Stage('pool', Spec('pool')) after the subject-level prefix, or "
            "remove the partition stage (one_step)."
        )
    if roles.count(ROLE_POOL) > 1:
        raise HABITAPIError(
            "Stage sequence contains more than one pool marker; the "
            "subject↔cohort watershed must appear at most once."
        )
    if ROLE_EXTRACT_VOXEL_FEATURES not in roles:
        raise HABITAPIError(
            "Stage sequence lacks extract_voxel_features "
            "(voxel_feature_extractor)."
        )
    if ROLE_FIT not in roles:
        raise HABITAPIError("Stage sequence lacks fit (habitat_model_fitter).")
    if ROLE_ASSIGN not in roles:
        raise HABITAPIError("Stage sequence lacks assign (habitat_assigner).")
    if has_pool:
        pool_i = roles.index(ROLE_POOL)
        fit_i = roles.index(ROLE_FIT)
        if fit_i < pool_i:
            raise HABITAPIError(
                "fit must come after pool in a cohort-level stage sequence "
                f"(pool at index {pool_i}, fit at index {fit_i})."
            )


def design_from_stages(resolved: Sequence[ResolvedStage]) -> str:
    """
    Infer the legacy design name from a resolved stage sequence.

    Args:
        resolved: Output of :func:`resolve_habitat_stages`.

    Returns:
        One of ``"two_step"``, ``"direct_pooling"``, ``"one_step"``.
    """
    roles = {stage.role for stage in resolved}
    has_partition = ROLE_PARTITION in roles
    has_pool = ROLE_POOL in roles
    if has_partition and has_pool:
        return "two_step"
    if has_pool:
        return "direct_pooling"
    return "one_step"
