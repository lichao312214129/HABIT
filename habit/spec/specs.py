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
"""Algorithm specifications: what to run, never where data lives.

``Spec`` describes ONE pluggable component; ``HabitatSpec`` composes the
specs of a complete habitat analysis. Neither knows about file locations
(a ``DataSource`` concern) or execution policy (a ``RunPolicy`` concern) --
that tripartition is what makes a specification portable between machines
and what gives every result a stable, comparable fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

from habit.exceptions import HABITAPIError

__all__ = [
    "Spec",
    "Stage",
    "HabitatSpec",
    "MLSpec",
    "coerce_spec",
    "ROLE_EXTRACT_VOXEL_FEATURES",
    "ROLE_PREPROCESS",
    "ROLE_PARTITION",
    "ROLE_EXTRACT_SUPERVOXEL_FEATURES",
    "ROLE_POOL",
    "ROLE_FIT",
    "ROLE_ASSIGN",
    "ROLE_QUANTIFY",
    "ROLE_POSTPROCESS_SUPERVOXEL",
    "ROLE_POSTPROCESS_HABITAT",
    "POOL_COMPONENT_NAME",
]

#: Recommended / sugar role tags (documentation + sugar expansion). Names are
#: labels, not keywords -- domain code may also infer roles from position.
ROLE_EXTRACT_VOXEL_FEATURES = "extract_voxel_features"
ROLE_PREPROCESS = "preprocess"
ROLE_PARTITION = "partition"
ROLE_EXTRACT_SUPERVOXEL_FEATURES = "extract_supervoxel_features"
ROLE_POOL = "pool"
ROLE_FIT = "fit"
ROLE_ASSIGN = "assign"
ROLE_QUANTIFY = "quantify"
ROLE_POSTPROCESS_SUPERVOXEL = "postprocess_supervoxel"
ROLE_POSTPROCESS_HABITAT = "postprocess_habitat"

#: Built-in marker component name in the ``pooling`` plugin domain.
POOL_COMPONENT_NAME = "pool"

#: Registry domains recognised inside a HabitatSpec, in canonical order.
#: Field names deliberately match the plugin domains verbatim (see
#: developer/api_upgrade/08_naming_decisions.md §4) so no fourth vocabulary
#: appears between YAML, Python, and the registry layer.
_COMPONENT_DOMAINS: Tuple[str, ...] = (
    "voxel_feature_extractor",
    "supervoxelizer",
    "supervoxel_feature_extractor",
    "habitat_model_fitter",
    "habitat_assigner",
)

#: Component domains a specification may leave unset. Both concern the
#: supervoxel stage, which the one-step and direct-pooling designs skip.
_OPTIONAL_COMPONENT_DOMAINS: Tuple[str, ...] = (
    "supervoxelizer",
    "supervoxel_feature_extractor",
)

#: Methods styles supported by ``HabitatSpec.describe_methods`` (and by the
#: RunManifest counterpart -- same verb, same signature, same vocabulary).
_METHODS_STYLES: Tuple[str, ...] = ("radiology", "nature")

#: Human phrasing for the component domains, in pipeline order.
#: (Deliberately duplicated in ``habit.contracts.manifest``: the spec layer
#: sits at the foundation of the stack and must never import upwards, and
#: the fragment is small.)
_COMPONENT_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("voxel_feature_extractor", "voxel feature extraction"),
    ("supervoxelizer", "supervoxelization"),
    ("postprocess_supervoxel", "supervoxel connected-component postprocessing"),
    ("supervoxel_feature_extractor", "supervoxel feature extraction"),
    ("habitat_model_fitter", "habitat model fitting"),
    ("habitat_assigner", "habitat assignment"),
    ("postprocess_habitat", "habitat connected-component postprocessing"),
)

#: Preprocessing chains, keyed by field name, with the prose each renders as.
#: Ordered as they run. The first two are stateless and per subject, the third
#: is fitted once on the training cohort -- the distinction that decides where
#: leakage is possible, and the reason the field names say WHOSE statistics are
#: used rather than which granularity is processed.
_PREPROCESSING_CHAINS: Tuple[Tuple[str, str], ...] = (
    ("voxel_feature_preprocessors", "per-subject voxel feature preprocessing"),
    (
        "supervoxel_feature_preprocessors",
        "per-subject supervoxel feature preprocessing",
    ),
    ("cohort_feature_preprocessors", "cohort-level feature preprocessing"),
)


def _params_text(params: Any) -> str:
    """Render one component's parameter mapping as prose."""
    if not params:
        return "default parameters"
    if isinstance(params, Mapping):
        return ", ".join(f"{key}={value!r}" for key, value in params.items())
    return str(params)


def _component_phrases(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    """
    Render a HabitatSpec-shaped payload as ordered prose phrases.

    Only keys actually present are rendered: the paragraph states what the
    analysis contains, never what a template might have contained.

    Args:
        payload: Spec payload as produced by ``HabitatSpec.to_dict``.

    Returns:
        One phrase per present component, in pipeline order.
    """
    phrases: list[str] = []
    for key, phrase in _COMPONENT_PHRASES:
        if key not in payload:
            continue
        entry = payload[key]
        if entry is None:
            if key == "supervoxelizer":
                phrases.append("direct voxel clustering (no supervoxelization)")
            # An unset supervoxel_feature_extractor is not a missing step:
            # the supervoxelizer's own feature means describe the regions.
            continue
        if isinstance(entry, Mapping) and "name" in entry:
            phrases.append(
                f"{phrase} with {entry['name']} ({_params_text(entry.get('params'))})"
            )
        else:
            phrases.append(f"{phrase} with {entry}")
    features = payload.get("habitat_features") or []
    if features:
        families = ", ".join(
            f"{entry['name']} ({_params_text(entry.get('params'))})"
            if isinstance(entry, Mapping) and "name" in entry
            else str(entry)
            for entry in features
        )
        phrases.append(f"habitat feature families: {families}")
    for chain_key, chain_phrase in _PREPROCESSING_CHAINS:
        chain = payload.get(chain_key) or []
        if chain:
            steps = ", ".join(
                entry["name"] if isinstance(entry, Mapping) and "name" in entry else str(entry)
                for entry in chain
            )
            phrases.append(f"{chain_phrase}: {steps}")
    return tuple(phrases)


def _canonical_json(value: Any) -> str:
    """
    Serialise a spec payload deterministically.

    Tuples become lists and mappings are key-sorted so that two equal specs
    always produce byte-identical text -- the precondition for a stable
    fingerprint.

    Args:
        value: Spec payload of plain Python / NumPy scalar values.

    Returns:
        Canonical JSON text.
    """
    def _normalise(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): _normalise(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [_normalise(val) for val in item]
        # NumPy scalars and Path-likes degrade to plain JSON values.
        if hasattr(item, "item") and callable(item.item):
            return item.item()
        if isinstance(item, (int, float, str, bool)) or item is None:
            return item
        return str(item)

    return json.dumps(_normalise(value), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Spec:
    """
    Specification of ONE pluggable component.

    Attributes:
        name: Registered component name, e.g. ``"slic"``.
        params: Constructor parameters. Defaults live in the component
            classes themselves so a spec only records deviations.
        version: Specification schema version, reserved for future
            migrations.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Normalise the parameter mapping into an immutable plain dict."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise HABITAPIError("Spec.name must be a non-empty string.")
        object.__setattr__(self, "params", dict(self.params))

    def fingerprint(self) -> str:
        """
        Return a stable hash identifying this exact specification.

        Two runs with equal fingerprints are scientifically comparable;
        caching and provenance key on it.

        Returns:
            Hex digest of the canonical payload.
        """
        return hashlib.sha256(self._payload().encode("utf-8")).hexdigest()

    def _payload(self) -> str:
        """Return the canonical text form used for hashing."""
        return _canonical_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (YAML/JSON isomorphic)."""
        return {
            "name": self.name,
            "params": json.loads(_canonical_json(self.params)),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Spec":
        """
        Rebuild a spec from its dict form.

        Args:
            payload: Mapping with ``name`` and optional ``params`` /
                ``version``.

        Returns:
            The reconstructed spec.

        Raises:
            HABITAPIError: If ``name`` is missing.
        """
        if "name" not in payload:
            raise HABITAPIError(f"Spec payload lacks 'name': {payload!r}.")
        return cls(
            name=str(payload["name"]),
            params=dict(payload.get("params", {})),
            version=str(payload.get("version", "1.0")),
        )


def coerce_spec(entry: Any) -> Optional[Spec]:
    """
    Coerce one component payload into a Spec.

    Accepts the structured mapping form (``name``/``params``) and the
    compact string form (a strict feature-tree expression such as
    ``'concat(raw("T1"), raw("T2"))'``), so YAML documents may spell a
    feature component either way.

    Args:
        entry: The payload, or ``None``.

    Returns:
        The coerced Spec, or ``None`` for an unset component.

    Raises:
        HABITAPIError: On a payload of any other type.
    """
    if entry is None:
        return None
    if isinstance(entry, str):
        # Lazy import: expressions.py itself depends on this module's Spec.
        from habit.spec.expressions import parse_feature_expression

        return parse_feature_expression(entry)
    if isinstance(entry, Mapping):
        return Spec.from_dict(entry)
    raise HABITAPIError(
        f"A component spec must be a mapping or an expression string; "
        f"got {type(entry).__name__}: {entry!r}."
    )


@dataclass(frozen=True)
class Stage:
    """
    One named step in a habitat dataflow.

    Attributes:
        name: Custom label unique within the enclosing HabitatSpec. Defaults
            to the component registry name when built via :meth:`of`.
        component: The pluggable component specification.
        role: Optional role tag filled by sugar expansion. ``None`` for
            user-authored stages until domain role resolution runs.
    """

    name: str
    component: Spec
    role: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate name / component types."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise HABITAPIError("Stage.name must be a non-empty string.")
        if not isinstance(self.component, Spec):
            raise HABITAPIError(
                f"Stage.component must be a Spec; got "
                f"{type(self.component).__name__}."
            )
        if self.role is not None and (
            not isinstance(self.role, str) or not self.role.strip()
        ):
            raise HABITAPIError("Stage.role must be a non-empty string when set.")

    @classmethod
    def of(
        cls,
        component: Union[Spec, Mapping[str, Any]],
        name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> "Stage":
        """
        Build a stage, defaulting ``name`` to the component registry name.

        Args:
            component: Component spec or mapping.
            name: Optional custom label.
            role: Optional sugar role tag.

        Returns:
            The stage.
        """
        spec = component if isinstance(component, Spec) else coerce_spec(component)
        if spec is None:
            raise HABITAPIError("Stage.of requires a component Spec.")
        return cls(name=name or spec.name, component=spec, role=role)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for YAML / fingerprint payloads."""
        payload: Dict[str, Any] = {
            "name": self.name,
            "component": self.component.to_dict(),
        }
        if self.role is not None:
            payload["role"] = self.role
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage":
        """
        Rebuild a stage from its dict form.

        Args:
            payload: Mapping with ``name`` + ``component``, or a bare Spec
                mapping with optional ``stage_name``.

        Returns:
            The reconstructed stage.
        """
        if "component" in payload:
            component = coerce_spec(payload["component"])
            if component is None:
                raise HABITAPIError(
                    f"Stage payload lacks a component Spec: {payload!r}."
                )
            name = str(payload.get("name") or component.name)
            role = payload.get("role")
            return cls(
                name=name,
                component=component,
                role=None if role is None else str(role),
            )
        component = coerce_spec(payload)
        if component is None:
            raise HABITAPIError(f"Cannot parse Stage from payload: {payload!r}.")
        name = str(payload.get("stage_name") or component.name)
        role = payload.get("role")
        return cls(
            name=name,
            component=component,
            role=None if role is None else str(role),
        )


def _expand_stages_from_fields(
    *,
    voxel_feature_extractor: Spec,
    supervoxelizer: Optional[Spec],
    supervoxel_feature_extractor: Optional[Spec],
    habitat_model_fitter: Spec,
    habitat_assigner: Spec,
    habitat_features: Tuple[Spec, ...],
    voxel_feature_preprocessors: Tuple[Spec, ...],
    supervoxel_feature_preprocessors: Tuple[Spec, ...],
    cohort_feature_preprocessors: Tuple[Spec, ...],
    pooling: Optional[str],
    postprocess_supervoxel: Optional[Spec],
    postprocess_habitat: Optional[Spec],
) -> Tuple[Stage, ...]:
    """
    Expand the named-field sugar form into an ordered stage list.

    Preprocess stages are numbered globally (``preprocess1``, …). A ``pool``
    marker is inserted when the dataflow is cohort-level (default / explicit
    ``pooling != "none"``).
    """
    stages: list[Stage] = [
        Stage(
            name=ROLE_EXTRACT_VOXEL_FEATURES,
            component=voxel_feature_extractor,
            role=ROLE_EXTRACT_VOXEL_FEATURES,
        )
    ]
    preprocess_i = 1
    for method in voxel_feature_preprocessors:
        stages.append(
            Stage(
                name=f"{ROLE_PREPROCESS}{preprocess_i}",
                component=method,
                role=ROLE_PREPROCESS,
            )
        )
        preprocess_i += 1
    if supervoxelizer is not None:
        stages.append(
            Stage(
                name=ROLE_PARTITION,
                component=supervoxelizer,
                role=ROLE_PARTITION,
            )
        )
        if postprocess_supervoxel is not None:
            stages.append(
                Stage(
                    name=ROLE_POSTPROCESS_SUPERVOXEL,
                    component=postprocess_supervoxel,
                    role=ROLE_POSTPROCESS_SUPERVOXEL,
                )
            )
        if supervoxel_feature_extractor is not None:
            stages.append(
                Stage(
                    name=ROLE_EXTRACT_SUPERVOXEL_FEATURES,
                    component=supervoxel_feature_extractor,
                    role=ROLE_EXTRACT_SUPERVOXEL_FEATURES,
                )
            )
        for method in supervoxel_feature_preprocessors:
            stages.append(
                Stage(
                    name=f"{ROLE_PREPROCESS}{preprocess_i}",
                    component=method,
                    role=ROLE_PREPROCESS,
                )
            )
            preprocess_i += 1
    include_pool = pooling != "none"
    if include_pool:
        stages.append(
            Stage(
                name=ROLE_POOL,
                component=Spec(POOL_COMPONENT_NAME),
                role=ROLE_POOL,
            )
        )
        for method in cohort_feature_preprocessors:
            stages.append(
                Stage(
                    name=f"{ROLE_PREPROCESS}{preprocess_i}",
                    component=method,
                    role=ROLE_PREPROCESS,
                )
            )
            preprocess_i += 1
    stages.append(
        Stage(
            name=ROLE_FIT,
            component=habitat_model_fitter,
            role=ROLE_FIT,
        )
    )
    stages.append(
        Stage(
            name=ROLE_ASSIGN,
            component=habitat_assigner,
            role=ROLE_ASSIGN,
        )
    )
    if postprocess_habitat is not None:
        stages.append(
            Stage(
                name=ROLE_POSTPROCESS_HABITAT,
                component=postprocess_habitat,
                role=ROLE_POSTPROCESS_HABITAT,
            )
        )
    for index, feature in enumerate(habitat_features):
        name = ROLE_QUANTIFY if index == 0 else f"{ROLE_QUANTIFY}{index + 1}"
        stages.append(Stage(name=name, component=feature, role=ROLE_QUANTIFY))
    return tuple(stages)


def _named_fields_from_stages(
    stages: Sequence[Stage],
) -> Dict[str, Any]:
    """
    Derive named HabitatSpec fields from a stage list that already carries roles.

    Args:
        stages: Stages with ``role`` set (sugar expansion or prior resolution).

    Returns:
        Keyword arguments suitable for constructing / updating HabitatSpec.
    """
    voxel_feature_extractor: Optional[Spec] = None
    supervoxelizer: Optional[Spec] = None
    supervoxel_feature_extractor: Optional[Spec] = None
    habitat_model_fitter: Optional[Spec] = None
    habitat_assigner: Optional[Spec] = None
    habitat_features: list[Spec] = []
    voxel_pre: list[Spec] = []
    supervoxel_pre: list[Spec] = []
    cohort_pre: list[Spec] = []
    postprocess_supervoxel: Optional[Spec] = None
    postprocess_habitat: Optional[Spec] = None
    seen_pool = False
    seen_partition = False
    for stage in stages:
        role = stage.role
        if role is None:
            raise HABITAPIError(
                f"Stage {stage.name!r} has no role; resolve roles before "
                "deriving named HabitatSpec fields."
            )
        if role == ROLE_EXTRACT_VOXEL_FEATURES:
            voxel_feature_extractor = stage.component
        elif role == ROLE_PARTITION:
            supervoxelizer = stage.component
            seen_partition = True
        elif role == ROLE_EXTRACT_SUPERVOXEL_FEATURES:
            supervoxel_feature_extractor = stage.component
        elif role == ROLE_POOL:
            seen_pool = True
        elif role == ROLE_FIT:
            habitat_model_fitter = stage.component
        elif role == ROLE_ASSIGN:
            habitat_assigner = stage.component
        elif role == ROLE_QUANTIFY:
            habitat_features.append(stage.component)
        elif role == ROLE_POSTPROCESS_SUPERVOXEL:
            postprocess_supervoxel = stage.component
        elif role == ROLE_POSTPROCESS_HABITAT:
            postprocess_habitat = stage.component
        elif role == ROLE_PREPROCESS:
            if not seen_pool and not seen_partition:
                voxel_pre.append(stage.component)
            elif not seen_pool and seen_partition:
                supervoxel_pre.append(stage.component)
            else:
                cohort_pre.append(stage.component)
        else:
            raise HABITAPIError(
                f"Unknown stage role {role!r} on stage {stage.name!r}."
            )
    if voxel_feature_extractor is None:
        raise HABITAPIError(
            "stages must include an extract_voxel_features role "
            "(voxel_feature_extractor)."
        )
    if habitat_model_fitter is None:
        raise HABITAPIError(
            "stages must include a fit role (habitat_model_fitter)."
        )
    if habitat_assigner is None:
        raise HABITAPIError(
            "stages must include an assign role (habitat_assigner)."
        )
    pooling = "cohort" if seen_pool else "none"
    return {
        "voxel_feature_extractor": voxel_feature_extractor,
        "supervoxelizer": supervoxelizer,
        "supervoxel_feature_extractor": supervoxel_feature_extractor,
        "habitat_model_fitter": habitat_model_fitter,
        "habitat_assigner": habitat_assigner,
        "habitat_features": tuple(habitat_features),
        "voxel_feature_preprocessors": tuple(voxel_pre),
        "supervoxel_feature_preprocessors": tuple(supervoxel_pre),
        "cohort_feature_preprocessors": tuple(cohort_pre),
        "pooling": pooling,
        "postprocess_supervoxel": postprocess_supervoxel,
        "postprocess_habitat": postprocess_habitat,
    }


@dataclass(frozen=True)
class HabitatSpec:
    """
    Complete specification of a habitat analysis.

    A frozen, fingerprintable value object. ``supervoxelizer=None`` selects
    the direct clustering designs (one-step / direct-pooling), mirroring the
    ``SubjectPipeline`` contract.

    Attributes:
        name: Human-readable specification name.
        voxel_feature_extractor: Spec of the voxel feature step.
        supervoxelizer: Spec of the supervoxel step, or ``None``.
        supervoxel_feature_extractor: Spec of the step describing the
            supervoxels, or ``None`` to keep the supervoxelizer's feature
            means (the v0.1 default).
        habitat_model_fitter: Spec of the cohort-level fitting step.
        habitat_assigner: Spec of the per-subject assignment step.
        habitat_features: Specs of habitat feature families.
        voxel_feature_preprocessors: Ordered method specs of the stateless
            per-subject chain applied to voxel features BEFORE
            supervoxelization (v0.1's ``preprocessing_for_subject_level``).
        supervoxel_feature_preprocessors: Ordered method specs of the
            stateless per-subject chain applied to supervoxel features. Has no
            v0.1 equivalent: that version could only preprocess supervoxel
            features at cohort level, which forced per-supervoxel radiomics
            through a stateful step it did not need.
        cohort_feature_preprocessors: Ordered method specs of the stateful
            chain fitted once on the pooled TRAINING units and replayed
            afterwards (v0.1's ``preprocessing_for_group_level``). Its fitted
            state is stored in ``HabitatModel.preprocessing_state``, because a
            habitat definition is only reproducible together with the feature
            space it was defined in.
        random_seed: Seed applied to every
            :class:`~habit.domain.protocols.Seedable` component. Seeds
            change the scientific result, so they live in the spec (and its
            fingerprint), not in the run policy.
        on_geometry_mismatch: How to handle image/mask voxel-grid disagreements
            before Stage-1 extraction. ``"resample_mask"`` (default)
            nearest-neighbour resamples each ROI onto the reference image
            grid; ``"strict"`` raises :class:`~habit.exceptions.GeometryError`.
            The default is omitted from :meth:`to_dict` so historical
            fingerprints stay stable when the policy is unchanged.
        pooling: Cross-subject pooling declaration of the habitat dataflow
            (sugar / derived view). Prefer :attr:`stages` with a ``pool``
            marker for new code. ``"cohort"`` pools clustering units across
            subjects; ``"none"`` defines habitats inside each subject
            (one-step). ``None`` (default) means undeclared and resolves to
            ``"cohort"`` for sugar forms without an explicit ``pool`` stage;
            both ``None`` and ``"cohort"`` are omitted from :meth:`to_dict`
            so historical fingerprints stay stable, while ``"none"`` is
            always recorded (with the derived :attr:`definition_level`).
        stages: Ordered named stages (source of truth when provided
            explicitly). The named component fields above remain as sugar
            that normalises to the same internal stage list.
        postprocess_supervoxel: Optional Spec for connected-component cleanup
            of supervoxel label maps (two-step). ``None`` skips cleanup and is
            omitted from :meth:`to_dict` so historical fingerprints stay stable.
        postprocess_habitat: Optional Spec for connected-component cleanup of
            final habitat label maps. ``None`` skips cleanup and is omitted
            from :meth:`to_dict`.
        version: Specification schema version.

    Examples:
        A two-step design (supervoxels per subject, habitats across the
        cohort) declared as data:

        >>> from habit import HabitatSpec, Spec
        >>> spec = HabitatSpec(
        ...     name="habitat_two_step",
        ...     voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        ...     supervoxelizer=Spec("kmeans", {"n_supervoxels": 50, "n_init": 10}),
        ...     habitat_model_fitter=Spec(
        ...         "kmeans",
        ...         {"min_habitats": 2, "max_habitats": 10, "validation": "silhouette"},
        ...     ),
        ...     habitat_assigner=Spec("nearest_centroid"),
        ...     habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
        ...     random_seed=42,
        ... )
        >>> spec.fingerprint()  # doctest: +ELLIPSIS
        '...'

        The same document expressed as YAML (``version: '1.0'`` /
        ``workflow: habitat``) loads with
        :func:`~habit.spec.load_habitat_spec`; see
        ``config/habitat/config_habitat_two_step_v1.yaml`` for a complete
        annotated example.
    """

    name: str
    # Named fields are sugar. They are required unless ``stages`` is supplied
    # explicitly (then roles fill them after resolution / sugar roles).
    voxel_feature_extractor: Optional[Spec] = None
    supervoxelizer: Optional[Spec] = None
    habitat_model_fitter: Optional[Spec] = None
    habitat_assigner: Optional[Spec] = None
    supervoxel_feature_extractor: Optional[Spec] = None
    habitat_features: Tuple[Spec, ...] = ()
    voxel_feature_preprocessors: Tuple[Spec, ...] = ()
    supervoxel_feature_preprocessors: Tuple[Spec, ...] = ()
    cohort_feature_preprocessors: Tuple[Spec, ...] = ()
    random_seed: Optional[int] = None
    on_geometry_mismatch: str = "resample_mask"
    pooling: Optional[str] = None
    stages: Optional[Tuple[Stage, ...]] = None
    postprocess_supervoxel: Optional[Spec] = None
    postprocess_habitat: Optional[Spec] = None
    version: str = "1.0"
    #: True when the caller supplied ``stages=`` explicitly (fingerprint
    #: records the ordered stage list). Sugar-only specs keep historical
    #: named-field fingerprints.
    _stages_explicit: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Coerce component payloads into Spec instances and tuples."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise HABITAPIError("HabitatSpec.name must be a non-empty string.")
        # Validate here (not via habit.domain) so habit.spec stays below L3.
        geometry_policy = str(self.on_geometry_mismatch).strip().lower()
        if geometry_policy not in ("resample_mask", "strict"):
            raise HABITAPIError(
                "HabitatSpec.on_geometry_mismatch must be 'resample_mask' or "
                f"'strict'; got {self.on_geometry_mismatch!r}."
            )
        object.__setattr__(self, "on_geometry_mismatch", geometry_policy)
        if self.pooling is not None:
            pooling = str(self.pooling).strip().lower()
            if pooling not in ("cohort", "none"):
                raise HABITAPIError(
                    "HabitatSpec.pooling must be 'cohort' or 'none' (or None "
                    f"to leave the dataflow undeclared); got {self.pooling!r}."
                )
            object.__setattr__(self, "pooling", pooling)

        # ``stages`` stores ONLY caller-authored stages. Sugar expansion is
        # computed by :meth:`resolved_stages` so ``dataclasses.replace`` on
        # named fields (e.g. ``pooling="none"``) never keeps a stale pool
        # stage from a previous expansion.
        stages_explicit = self.stages is not None
        if stages_explicit:
            coerced_stages: List[Stage] = []
            for entry in self.stages or ():
                if isinstance(entry, Stage):
                    coerced_stages.append(entry)
                elif isinstance(entry, Mapping):
                    coerced_stages.append(Stage.from_dict(entry))
                else:
                    raise HABITAPIError(
                        "Every HabitatSpec.stages entry must be a Stage or "
                        f"mapping; got {type(entry).__name__}."
                    )
            object.__setattr__(self, "stages", tuple(coerced_stages))
            if coerced_stages and all(stage.role for stage in coerced_stages):
                derived = _named_fields_from_stages(tuple(coerced_stages))
                for key, value in derived.items():
                    object.__setattr__(self, key, value)

        for field_name in ("postprocess_supervoxel", "postprocess_habitat"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, Spec):
                continue
            coerced = coerce_spec(value)
            if coerced is None or not isinstance(coerced, Spec):
                raise HABITAPIError(
                    f"HabitatSpec.{field_name} must be a Spec; "
                    f"got {type(value).__name__}."
                )
            object.__setattr__(self, field_name, coerced)
        for chain_field in (
            "habitat_features",
            *(key for key, _ in _PREPROCESSING_CHAINS),
        ):
            chain = tuple(getattr(self, chain_field))
            object.__setattr__(self, chain_field, chain)
            for entry in chain:
                if not isinstance(entry, Spec):
                    raise HABITAPIError(
                        f"Every entry of HabitatSpec.{chain_field} must be a Spec."
                    )

        # Sugar path: named fields required when stages were not supplied.
        if not stages_explicit:
            for domain in _COMPONENT_DOMAINS:
                value = getattr(self, domain)
                if value is None:
                    if domain in _OPTIONAL_COMPONENT_DOMAINS:
                        continue
                    raise HABITAPIError(
                        f"HabitatSpec requires a '{domain}' component spec "
                        "(or an explicit stages list that provides it)."
                    )
                if not isinstance(value, Spec):
                    raise HABITAPIError(
                        f"HabitatSpec.{domain} must be a Spec; "
                        f"got {type(value).__name__}."
                    )
        else:
            for domain in _COMPONENT_DOMAINS:
                value = getattr(self, domain)
                if value is None:
                    continue
                if not isinstance(value, Spec):
                    raise HABITAPIError(
                        f"HabitatSpec.{domain} must be a Spec; "
                        f"got {type(value).__name__}."
                    )

        object.__setattr__(self, "_stages_explicit", stages_explicit)
        if self.random_seed is not None:
            object.__setattr__(self, "random_seed", int(self.random_seed))

        stage_names = [stage.name for stage in self.resolved_stages()]
        if len(stage_names) != len(set(stage_names)):
            dupes = sorted(
                {name for name in stage_names if stage_names.count(name) > 1}
            )
            raise HABITAPIError(
                f"HabitatSpec stage names must be unique; duplicates: {dupes}. "
                "Rename the colliding Stage.name labels."
            )

    def component_specs(self) -> Mapping[str, Optional[Spec]]:
        """Return the pipeline component specs keyed by domain name."""
        return {
            "voxel_feature_extractor": self.voxel_feature_extractor,
            "supervoxelizer": self.supervoxelizer,
            "supervoxel_feature_extractor": self.supervoxel_feature_extractor,
            "habitat_model_fitter": self.habitat_model_fitter,
            "habitat_assigner": self.habitat_assigner,
        }

    @property
    def definition_level(self) -> str:
        """
        Level at which the habitat definition is learned, DERIVED from the
        declared dataflow.

        ``"subject"`` when there is no ``pool`` stage / ``pooling="none"``
        (each subject defines its own habitats; the one-step design),
        otherwise ``"cohort"``. This is a read-only view of the spec graph,
        not a free-form field.
        """
        if self.stages and any(
            stage.role == ROLE_POOL
            or (
                stage.role is None
                and stage.component.name == POOL_COMPONENT_NAME
            )
            for stage in self.stages
        ):
            return "cohort"
        return "subject" if self.pooling == "none" else "cohort"

    def resolved_stages(self) -> Tuple[Stage, ...]:
        """
        Return the ordered stages (explicit or sugar-expanded).

        Sugar expansion is computed here (not stored on ``stages``) so
        replacing named fields such as ``pooling`` rebuilds the sequence.

        Returns:
            The effective stage tuple used by the executor and fingerprints
            of explicit-stage specs.
        """
        if self._stages_explicit and self.stages is not None:
            return self.stages
        if self.voxel_feature_extractor is None or self.habitat_model_fitter is None:
            return self.stages or ()
        if self.habitat_assigner is None:
            return self.stages or ()
        return _expand_stages_from_fields(
            voxel_feature_extractor=self.voxel_feature_extractor,
            supervoxelizer=self.supervoxelizer,
            supervoxel_feature_extractor=self.supervoxel_feature_extractor,
            habitat_model_fitter=self.habitat_model_fitter,
            habitat_assigner=self.habitat_assigner,
            habitat_features=self.habitat_features,
            voxel_feature_preprocessors=self.voxel_feature_preprocessors,
            supervoxel_feature_preprocessors=self.supervoxel_feature_preprocessors,
            cohort_feature_preprocessors=self.cohort_feature_preprocessors,
            pooling=self.pooling,
            postprocess_supervoxel=self.postprocess_supervoxel,
            postprocess_habitat=self.postprocess_habitat,
        )

    def validate_dataflow(self) -> None:
        """
        Check cross-field / stage-sequence consistency of the dataflow.

        Construction (:meth:`__post_init__`) only enforces value domains so a
        spec stays a constructible value object; scientifically meaningless
        combinations are rejected here at entry points (recipes /
        ``habit check-config``). Role inference that needs registries runs in
        ``habit.domain.stages`` and is invoked from ``fit_habitat``.

        Raises:
            HABITAPIError: On illegal sugar combinations or structural stage
                errors (duplicate names already rejected at construction;
                partition without pool; subject-level + cohort preprocess).
        """
        stages = self.resolved_stages()
        roles = [stage.role for stage in stages if stage.role is not None]
        has_partition = ROLE_PARTITION in roles or self.supervoxelizer is not None
        has_pool = ROLE_POOL in roles or any(
            stage.component.name == POOL_COMPONENT_NAME for stage in stages
        )
        # Undeclared pooling sugar still means cohort unless pooling='none'.
        if self.pooling == "none":
            has_pool = False
        elif self.pooling == "cohort":
            has_pool = True
        elif not roles and self.pooling is None:
            has_pool = True

        if has_partition and not has_pool:
            raise HABITAPIError(
                "HabitatSpec declares a partition (supervoxelizer) stage but "
                "no pool stage: per-subject definition on supervoxels is not "
                "a supported design. Add a pool stage "
                "(Stage('pool', Spec('pool'))) after the subject-level "
                "prefix, or remove the partition stage for one_step."
            )
        if self.pooling == "none":
            if self.supervoxelizer is not None:
                raise HABITAPIError(
                    "HabitatSpec.pooling='none' (subject-level habitat "
                    "definition) does not support a supervoxelizer: "
                    "per-subject definition on supervoxels is not a supported "
                    "design. Remove the supervoxelizer (one-step) or declare "
                    "pooling='cohort' / add a pool stage (two-step)."
                )
            if self.cohort_feature_preprocessors:
                raise HABITAPIError(
                    "HabitatSpec.pooling='none' (subject-level habitat "
                    "definition) does not support cohort_feature_preprocessors: "
                    "no step pools across subjects, so cohort-level fitted "
                    "statistics would never be used. Move the chain to "
                    "voxel_feature_preprocessors, or declare pooling='cohort' "
                    "/ insert a pool stage before those preprocess steps."
                )

    def describe_methods(self, style: str = "radiology") -> str:
        """
        Render the specification as a manuscript methods paragraph.

        Deliberately the same verb and signature as
        :meth:`habit.contracts.manifest.RunManifest.describe_methods`; the
        difference is completeness, not vocabulary. This describes what was
        INTENDED and can be read before anything runs -- a spec carries no
        software versions, no executed steps and no excluded subjects, so
        none are stated. Every configured step appears with its parameters,
        which is what makes the paragraph useful for preregistration and for
        checking a YAML against the paper draft before the compute starts.

        Args:
            style: Target venue convention. ``"radiology"`` opens with the
                design sentence; ``"nature"`` closes with it. Ordering and
                wording only -- the stated facts are identical.

        Returns:
            English prose describing every configured step and its
            parameters.

        Raises:
            HABITAPIError: On an unknown style.
        """
        if style not in _METHODS_STYLES:
            raise HABITAPIError(
                f"Unknown methods style {style!r}; expected one of "
                f"{_METHODS_STYLES}."
            )
        # Narrate both the ordered stages and the classic component phrases
        # so manuscripts keep recognisable step names while stages stay SoT.
        stage_phrases = [
            (
                f"{stage.name} ({stage.component.name}"
                f"{'' if not stage.component.params else ', ' + _params_text(stage.component.params)})"
            )
            for stage in self.resolved_stages()
        ]
        component_text = "; ".join(_component_phrases(self.to_dict()))
        if stage_phrases:
            body: list[str] = [
                f"The analysis specification {self.name!r} proceeds through "
                f"ordered stages: {'; '.join(stage_phrases)}. "
                f"In component terms it comprises {component_text}."
            ]
        else:
            body = [
                f"The analysis specification {self.name!r} comprises "
                f"{component_text}."
            ]
        if self.definition_level == "subject":
            body.append(
                "Habitats are defined within each subject independently "
                "(no cross-subject pooling), so habitat labels are not "
                "comparable across subjects."
            )
        if self.random_seed is not None:
            body.append(
                f"Random seed {self.random_seed} is fixed for every "
                "stochastic component."
            )
        if self.on_geometry_mismatch == "strict":
            body.append(
                "Image and ROI mask geometries must match exactly; "
                "mismatches raise an error."
            )
        else:
            body.append(
                "When an ROI mask and the reference image disagree on voxel "
                "grid metadata, the mask is aligned onto the image grid "
                "(adopt image geometry when shapes match; otherwise "
                "nearest-neighbour resample)."
            )
        if style == "nature":
            closing = "The analysis was designed with HABIT."
            return " ".join([*body, closing])
        opening = "A habitat imaging analysis was designed with HABIT as follows."
        return " ".join([opening, *body])

    def fingerprint(self) -> str:
        """Return a stable hash identifying this exact specification."""
        return hashlib.sha256(
            _canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (YAML isomorphic)."""
        payload: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }
        # Explicit stages are the fingerprint source of truth. Sugar-only
        # specs keep the historical named-field payload so two_step /
        # direct_pooling fingerprints stay byte-identical.
        if self._stages_explicit:
            payload["stages"] = [stage.to_dict() for stage in self.resolved_stages()]
            payload["random_seed"] = self.random_seed
            if self.on_geometry_mismatch != "resample_mask":
                payload["on_geometry_mismatch"] = self.on_geometry_mismatch
            return payload

        for domain, component in self.component_specs().items():
            payload[domain] = component.to_dict() if component is not None else None
        payload["habitat_features"] = [
            feature.to_dict() for feature in self.habitat_features
        ]
        for chain_key, _ in _PREPROCESSING_CHAINS:
            payload[chain_key] = [
                entry.to_dict() for entry in getattr(self, chain_key)
            ]
        payload["random_seed"] = self.random_seed
        # Omit the default so historical HabitatSpec fingerprints stay stable
        # for analyses that never opted into strict geometry checks.
        if self.on_geometry_mismatch != "resample_mask":
            payload["on_geometry_mismatch"] = self.on_geometry_mismatch
        # Record the dataflow only when it departs from the historical
        # default (cohort-level pooling): ``None`` (undeclared) and
        # ``"cohort"`` are semantically identical and must share one
        # fingerprint, so both are omitted; the subject-level design
        # (one-step) previously went unrecorded and is now always stated,
        # together with the derived definition level.
        if self.pooling == "none":
            payload["pooling"] = "none"
            payload["definition_level"] = self.definition_level
        # Omit unset postprocess slots so analyses that never enable cleanup
        # keep their historical fingerprints.
        if self.postprocess_supervoxel is not None:
            payload["postprocess_supervoxel"] = self.postprocess_supervoxel.to_dict()
        if self.postprocess_habitat is not None:
            payload["postprocess_habitat"] = self.postprocess_habitat.to_dict()
        return payload

    def to_effective_dict(self) -> Dict[str, Any]:
        """
        Serialise with fingerprint-stable defaults expanded for YAML export.

        Unlike :meth:`to_dict`, this always includes ``on_geometry_mismatch``,
        the resolved ``pooling`` / ``stages`` view with its derived
        ``definition_level``, and both postprocess slots (``null`` when unset)
        so a saved document records the full effective analysis, not only
        overridden fields. Fingerprints still use :meth:`to_dict`.
        """
        payload = self.to_dict()
        payload["on_geometry_mismatch"] = self.on_geometry_mismatch
        payload["stages"] = [stage.to_dict() for stage in self.resolved_stages()]
        resolved_pooling = (
            "none"
            if self.definition_level == "subject"
            else ("cohort" if self.pooling is None else self.pooling)
        )
        if self.pooling == "none":
            resolved_pooling = "none"
        elif self.definition_level == "cohort":
            resolved_pooling = "cohort" if self.pooling is None else self.pooling
        payload["pooling"] = resolved_pooling
        payload["definition_level"] = self.definition_level
        payload["postprocess_supervoxel"] = (
            self.postprocess_supervoxel.to_dict()
            if self.postprocess_supervoxel is not None
            else None
        )
        payload["postprocess_habitat"] = (
            self.postprocess_habitat.to_dict()
            if self.postprocess_habitat is not None
            else None
        )
        # Sugar named fields aid human readers even when stages are explicit.
        if self._stages_explicit and self.voxel_feature_extractor is not None:
            for domain, component in self.component_specs().items():
                payload.setdefault(
                    domain, component.to_dict() if component is not None else None
                )
        return payload

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Export the effective specification as YAML text.

        This is the Python→YAML half of the Spec/YAML isomorphism for the
        ``spec:`` section. For a **runnable** document that also carries
        ``data`` / ``policy`` / ``output`` (so CLI and
        :func:`~habit.recipes.run_from_yaml` can replay the run), use
        :func:`~habit.spec.save_habitat_config`.

        Args:
            path: Optional destination file; when set, the YAML is written.

        Returns:
            The YAML text of :meth:`to_effective_dict`.
        """
        # Lazy import keeps ``habit.spec.specs`` free of a hard ``yaml`` edge
        # at module import time (yaml_io owns all YAML I/O).
        from habit.spec.yaml_io import dumps_yaml, _write_yaml

        payload = self.to_effective_dict()
        text = dumps_yaml(payload)
        if path is not None:
            _write_yaml(payload, path)
        return text

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HabitatSpec":
        """
        Rebuild a habitat specification from its dict form.

        Args:
            payload: Mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed specification.

        Raises:
            HABITAPIError: If a required component is missing.
        """
        stages_payload = payload.get("stages")
        if stages_payload is not None and "voxel_feature_extractor" not in payload:
            # Stages-only document.
            stages = tuple(Stage.from_dict(item) for item in stages_payload)
            spec = cls(
                name=str(payload.get("name", "habitat_spec")),
                stages=stages,
                random_seed=payload.get("random_seed"),
                on_geometry_mismatch=str(
                    payload.get("on_geometry_mismatch", "resample_mask")
                ),
                version=str(payload.get("version", "1.0")),
            )
        else:
            components: Dict[str, Optional[Spec]] = {}
            for domain in _COMPONENT_DOMAINS:
                components[domain] = coerce_spec(payload.get(domain))
            features = tuple(
                cast(Spec, coerce_spec(item))
                for item in payload.get("habitat_features", ())
            )
            chains = {
                chain_key: tuple(
                    cast(Spec, coerce_spec(item))
                    for item in payload.get(chain_key, ())
                )
                for chain_key, _ in _PREPROCESSING_CHAINS
            }
            pooling = payload.get("pooling")
            # Named fields present: keep the sugar fingerprint path. An
            # accompanying ``stages`` list from ``to_effective_dict`` is
            # documentation only and must not flip ``_stages_explicit``.
            spec = cls(
                name=str(payload.get("name", "habitat_spec")),
                voxel_feature_extractor=components["voxel_feature_extractor"],
                supervoxelizer=components["supervoxelizer"],
                supervoxel_feature_extractor=components[
                    "supervoxel_feature_extractor"
                ],
                habitat_model_fitter=components["habitat_model_fitter"],
                habitat_assigner=components["habitat_assigner"],
                habitat_features=features,
                random_seed=payload.get("random_seed"),
                on_geometry_mismatch=str(
                    payload.get("on_geometry_mismatch", "resample_mask")
                ),
                pooling=None if pooling is None else str(pooling),
                postprocess_supervoxel=coerce_spec(
                    payload.get("postprocess_supervoxel")
                ),
                postprocess_habitat=coerce_spec(payload.get("postprocess_habitat")),
                version=str(payload.get("version", "1.0")),
                **chains,
            )
        # A document may also carry the derived ``definition_level`` (written
        # by ``to_dict`` / ``to_effective_dict`` / the legacy adapter). It is
        # not a free-form field: reject documents whose stated level
        # contradicts the declared dataflow instead of silently re-deriving.
        stated_level = payload.get("definition_level")
        if stated_level is not None and str(stated_level) != spec.definition_level:
            raise HABITAPIError(
                f"HabitatSpec document declares definition_level="
                f"{stated_level!r} but pooling={spec.pooling!r} derives "
                f"{spec.definition_level!r}; fix the document so the two "
                "agree (definition_level is derived, not settable)."
            )
        return spec


#: DEPRECATED tabular chains of an MLSpec, keyed by field name, with the
#: prose each renders as. Ordered as they run, which is also the order they
#: are concatenated into :attr:`MLSpec.steps`: selection may happen BEFORE
#: preprocessing (the stage v0.1 expressed as ``before_z_score: true`` --
#: scientifically meaningful whenever a selector's statistics are distorted
#: by normalisation, e.g. variance-based selection is vacuous after
#: z-scoring), preprocessing itself, then the ordinary post-preprocessing
#: selection.
#:
#: These three fixed slots express ORDER through STRUCTURE, which caps the
#: expressible orderings at two positions -- before all preprocessing, or
#: after all of it. ``zscore -> variance -> minmax -> lasso`` has no
#: representation here at all. :attr:`MLSpec.steps` replaces them with one
#: ordered list; they are kept as deprecated aliases for all of v1.x.
_ML_CHAINS: Tuple[Tuple[str, str], ...] = (
    ("pre_preprocessing_feature_selectors", "pre-preprocessing feature selection"),
    ("table_preprocessors", "table preprocessing"),
    ("feature_selectors", "feature selection"),
)

#: Field name of the single ordered step list that supersedes ``_ML_CHAINS``.
_ML_STEPS_FIELD = "steps"

#: Prose for the single ordered step list.
_ML_STEPS_PHRASE = "an ordered table pipeline of"


def _ml_phrases(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    """
    Render an MLSpec-shaped payload as ordered prose phrases.

    Handles both payload layouts. A payload carrying the single ordered
    ``steps`` list renders as one phrase, because that is exactly what the
    list is -- one ordered sequence whose positions carry the meaning the
    three deprecated chains used to carry structurally.

    Args:
        payload: Spec payload as produced by ``MLSpec.to_dict``.

    Returns:
        One phrase per present chain or component, in pipeline order.
    """
    phrases: list[str] = []
    ordered_steps = payload.get(_ML_STEPS_FIELD) or []
    if ordered_steps:
        rendered = ", ".join(
            f"{entry['name']} ({_params_text(entry.get('params'))})"
            if isinstance(entry, Mapping) and "name" in entry
            else str(entry)
            for entry in ordered_steps
        )
        phrases.append(f"{_ML_STEPS_PHRASE} {rendered}")
    for chain_key, chain_phrase in _ML_CHAINS:
        chain = payload.get(chain_key) or []
        if chain:
            steps = ", ".join(
                f"{entry['name']} ({_params_text(entry.get('params'))})"
                if isinstance(entry, Mapping) and "name" in entry
                else str(entry)
                for entry in chain
            )
            phrases.append(f"{chain_phrase} with {steps}")
    classifier = payload.get("classifier")
    if isinstance(classifier, Mapping) and "name" in classifier:
        phrases.append(
            f"a {classifier['name']} classifier "
            f"({_params_text(classifier.get('params'))})"
        )
    metrics = payload.get("metrics") or []
    if metrics:
        names = ", ".join(
            entry["name"] if isinstance(entry, Mapping) and "name" in entry else str(entry)
            for entry in metrics
        )
        phrases.append(f"evaluation metrics: {names}")
    return tuple(phrases)


@dataclass(frozen=True)
class MLSpec:
    """
    Complete specification of a tabular machine-learning analysis.

    A frozen, fingerprintable value object describing ONE modelling
    definition: an ordered chain of table steps (preprocessors and feature
    selectors, interleaved however the design calls for), exactly one
    terminal classifier, and the evaluation metric panel. It deliberately
    does NOT describe the validation design (split counts, resampling, id
    files) -- those are choices of the calling recipe, not of the model
    definition.

    **Step order lives in one ordered list.** :attr:`steps` is the pipeline:
    position N of the list is step N of the fit. The three fields
    :attr:`pre_preprocessing_feature_selectors`,
    :attr:`table_preprocessors` and :attr:`feature_selectors` are the
    DEPRECATED predecessor of that list -- they expressed order through
    three fixed slots, which allowed a selector to sit only before all
    preprocessing or after all of it. Declaring any of them still works for
    the whole of v1.x: the three are concatenated in their documented order
    (pre -> preprocessors -> post) into :attr:`steps`, with a
    ``DeprecationWarning``. Declaring both layouts at once is an error --
    which of the two is the pipeline would be a guess.

    **Which layout a spec serialises in.** :meth:`to_dict` emits the three
    deprecated keys for a spec declared with them, and the single ``steps``
    key for a spec declared with ``steps``. That asymmetry is deliberate and
    load-bearing: every provenance record and golden baseline HABIT has ever
    written hashes this payload, so unconditionally adding a ``steps`` key
    would move the fingerprint of every analysis already published. A spec
    with no table steps at all serialises in the deprecated shape for the
    same reason.

    Attributes:
        name: Human-readable specification name.
        classifier: Spec of the terminal classifier.
        pre_preprocessing_feature_selectors: DEPRECATED. Ordered specs of
            the selection chain fitted on the RAW training table, BEFORE
            any preprocessing (v0.1's ``before_z_score: true`` selectors).
            The stage exists because some selection statistics are
            distorted by normalisation -- after z-scoring every feature
            variance is 1.0, so variance-based selection only carries
            information on the raw table. Use :attr:`steps` and put the
            selector before the preprocessor instead.
        table_preprocessors: DEPRECATED. Ordered specs of the stateful
            preprocessing chain fitted on the TRAINING rows and replayed
            afterwards (v0.1's ``normalization``). Use :attr:`steps`.
        feature_selectors: DEPRECATED. Ordered specs of the selection
            chain, fitted after preprocessing (v0.1's
            ``feature_selection_methods`` entries without
            ``before_z_score``). Use :attr:`steps`.
        metrics: Specs of the evaluation metric panel. An empty tuple asks
            the calling recipe for its default panel.
        random_seed: Seed applied to every
            :class:`~habit.domain.protocols.Seedable` component. Seeds
            change the scientific result, so they live in the spec (and its
            fingerprint), not in the run policy.
        version: Specification schema version.
        steps: The ordered table-step chain -- preprocessors and feature
            selectors in the exact order they are fitted. Names are
            resolved across both registries by
            :func:`habit.domain.assembly.build_table_pipeline`; the spec
            layer records the order and stays registry-free. Declared last
            among the fields purely so that existing positional
            construction keeps meaning what it meant.
    """

    name: str
    classifier: Spec
    pre_preprocessing_feature_selectors: Tuple[Spec, ...] = ()
    table_preprocessors: Tuple[Spec, ...] = ()
    feature_selectors: Tuple[Spec, ...] = ()
    metrics: Tuple[Spec, ...] = ()
    random_seed: Optional[int] = None
    version: str = "1.0"
    # Appended after ``version`` rather than inserted next to the chains it
    # replaces: inserting it would silently change the meaning of every
    # positional MLSpec(...) call, which is the kind of breakage that does
    # not raise and lands straight in someone's results.
    steps: Tuple[Spec, ...] = ()

    def __post_init__(self) -> None:
        """
        Coerce payloads into Specs, and fold deprecated chains into ``steps``.

        Raises:
            HABITAPIError: On a missing/mistyped name or classifier, a chain
                entry that is not a :class:`Spec`, or a spec that declares
                both ``steps`` and any deprecated chain.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise HABITAPIError("MLSpec.name must be a non-empty string.")
        if not isinstance(self.classifier, Spec):
            raise HABITAPIError(
                "MLSpec.classifier must be a Spec; got "
                f"{type(self.classifier).__name__}."
            )
        for chain_field, _ in _ML_CHAINS + (
            ("metrics", ""),
            (_ML_STEPS_FIELD, ""),
        ):
            chain = tuple(getattr(self, chain_field))
            object.__setattr__(self, chain_field, chain)
            for entry in chain:
                if not isinstance(entry, Spec):
                    raise HABITAPIError(
                        f"Every entry of MLSpec.{chain_field} must be a Spec."
                    )
        if self.random_seed is not None:
            object.__setattr__(self, "random_seed", int(self.random_seed))
        declared_chains = tuple(
            chain_field
            for chain_field, _ in _ML_CHAINS
            if getattr(self, chain_field)
        )
        folded = tuple(
            entry
            for chain_field, _ in _ML_CHAINS
            for entry in getattr(self, chain_field)
        )
        if declared_chains and self.steps and self.steps != folded:
            # An already-translated spec passes BOTH through
            # ``dataclasses.replace`` (which re-supplies every field), and
            # that is consistent, not contradictory -- ``steps`` is then
            # exactly the fold of the chains. Only a genuine disagreement is
            # rejected, because there the pipeline would have to be guessed.
            raise HABITAPIError(
                "MLSpec declares both 'steps' and the deprecated chain(s) "
                f"{list(declared_chains)}, and they disagree: 'steps' is "
                f"{[entry.name for entry in self.steps]} while the chains "
                f"fold into {[entry.name for entry in folded]}. Move every "
                "step into 'steps' in the order it should run, or keep only "
                "the deprecated chains."
            )
        if declared_chains:
            warnings.warn(
                "MLSpec fields "
                f"{list(chain for chain, _ in _ML_CHAINS)} are deprecated; "
                "declare one ordered 'steps' list instead, where the list "
                "order is the execution order. The deprecated fields are "
                "translated into 'steps' as "
                "pre_preprocessing_feature_selectors + table_preprocessors "
                "+ feature_selectors and will be kept for all of v1.x.",
                DeprecationWarning,
                stacklevel=3,
            )
            object.__setattr__(self, _ML_STEPS_FIELD, folded)

    @property
    def declares_deprecated_chains(self) -> bool:
        """
        Report whether this spec was declared through the deprecated chains.

        Derived from the fields alone (never from hidden construction state)
        so that two equal specs always agree on it -- and therefore always
        serialise identically. A spec with no table steps at all counts as
        deprecated-shaped, which is what keeps its payload byte-identical to
        every one written before ``steps`` existed.

        Returns:
            bool: ``True`` when :meth:`to_dict` emits the three deprecated
            chain keys, ``False`` when it emits the single ``steps`` key.
        """
        return any(
            getattr(self, chain_field) for chain_field, _ in _ML_CHAINS
        ) or not self.steps

    def describe_methods(self, style: str = "radiology") -> str:
        """
        Render the specification as a manuscript methods paragraph.

        Same verb, signature, and vocabulary as
        :meth:`HabitatSpec.describe_methods`; this describes what was
        INTENDED and can be read before anything runs.

        Args:
            style: Target venue convention. ``"radiology"`` opens with the
                design sentence; ``"nature"`` closes with it.

        Returns:
            English prose describing every configured step and its
            parameters.

        Raises:
            HABITAPIError: On an unknown style.
        """
        if style not in _METHODS_STYLES:
            raise HABITAPIError(
                f"Unknown methods style {style!r}; expected one of "
                f"{_METHODS_STYLES}."
            )
        body: list[str] = [
            f"The modelling specification {self.name!r} comprises "
            f"{'; '.join(_ml_phrases(self.to_dict()))}."
        ]
        if self.random_seed is not None:
            body.append(
                f"Random seed {self.random_seed} is fixed for every "
                "stochastic component."
            )
        if style == "nature":
            closing = "The analysis was designed with HABIT."
            return " ".join([*body, closing])
        opening = "A machine-learning analysis was designed with HABIT as follows."
        return " ".join([opening, *body])

    def fingerprint(self) -> str:
        """Return a stable hash identifying this exact specification."""
        return hashlib.sha256(
            _canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a plain dict (YAML isomorphic).

        Emits exactly ONE of the two table-step layouts -- see the class
        docstring for why the choice is asymmetric rather than always
        writing both.

        Returns:
            Dict[str, Any]: The payload, with either the three deprecated
            chain keys or the single ``steps`` key, never both.
        """
        payload: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }
        if self.declares_deprecated_chains:
            for chain_key, _ in _ML_CHAINS:
                payload[chain_key] = [
                    entry.to_dict() for entry in getattr(self, chain_key)
                ]
        else:
            payload[_ML_STEPS_FIELD] = [entry.to_dict() for entry in self.steps]
        payload["classifier"] = self.classifier.to_dict()
        payload["metrics"] = [entry.to_dict() for entry in self.metrics]
        payload["random_seed"] = self.random_seed
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MLSpec":
        """
        Rebuild a machine-learning specification from its dict form.

        Reads whichever table-step layout the payload carries. A payload
        that carries both is rejected rather than resolved by precedence:
        picking one would silently drop half of a hand-written document's
        pipeline.

        Args:
            payload: Mapping as produced by :meth:`to_dict`, or a
                hand-written v1 ``spec`` section.

        Returns:
            The reconstructed specification.

        Raises:
            HABITAPIError: If the classifier component is missing, or the
                payload declares both ``steps`` and a deprecated chain.
        """
        chains = {
            chain_key: tuple(
                Spec.from_dict(item) for item in payload.get(chain_key, ())
            )
            for chain_key, _ in _ML_CHAINS + (("metrics", ""),)
        }
        chains[_ML_STEPS_FIELD] = tuple(
            Spec.from_dict(item) for item in payload.get(_ML_STEPS_FIELD, ())
        )
        classifier = payload.get("classifier")
        return cls(
            name=str(payload.get("name", "ml_spec")),
            classifier=cast(Spec, Spec.from_dict(classifier) if classifier is not None else None),
            random_seed=payload.get("random_seed"),
            version=str(payload.get("version", "1.0")),
            **chains,
        )
