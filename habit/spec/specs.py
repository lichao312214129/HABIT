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
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from habit.api.exceptions import HABITAPIError

__all__ = ["Spec", "HabitatSpec"]

#: Registry domains recognised inside a HabitatSpec, in canonical order.
#: Field names deliberately match the plugin domains verbatim (see
#: developer/api_upgrade/08_naming_decisions.md §4) so no fourth vocabulary
#: appears between YAML, Python, and the registry layer.
_COMPONENT_DOMAINS: Tuple[str, ...] = (
    "voxel_feature_extractor",
    "supervoxelizer",
    "habitat_model_fitter",
    "habitat_assigner",
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
    ("habitat_model_fitter", "habitat model fitting"),
    ("habitat_assigner", "habitat assignment"),
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
    for chain_key, chain_phrase in (
        ("subject_table_preprocessors", "subject-level table preprocessing"),
        ("group_table_preprocessors", "cohort-level table preprocessing"),
    ):
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
        habitat_model_fitter: Spec of the cohort-level fitting step.
        habitat_assigner: Spec of the per-subject assignment step.
        habitat_features: Specs of habitat feature families.
        subject_table_preprocessors: Ordered specs of the per-subject
            feature-table preprocessing chain (domain ``table_preprocessor``),
            applied to each subject's feature table before pooling.
        group_table_preprocessors: Ordered specs of the pooled-table
            preprocessing chain, applied before cohort-level fitting; their
            fitted state is stored in ``HabitatModel.preprocessing_state``.
        random_seed: Seed applied to every
            :class:`~habit.domain.protocols.Seedable` component. Seeds
            change the scientific result, so they live in the spec (and its
            fingerprint), not in the run policy.
        version: Specification schema version.
    """

    name: str
    voxel_feature_extractor: Spec
    supervoxelizer: Optional[Spec]
    habitat_model_fitter: Spec
    habitat_assigner: Spec
    habitat_features: Tuple[Spec, ...] = ()
    subject_table_preprocessors: Tuple[Spec, ...] = ()
    group_table_preprocessors: Tuple[Spec, ...] = ()
    random_seed: Optional[int] = None
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Coerce component payloads into Spec instances and tuples."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise HABITAPIError("HabitatSpec.name must be a non-empty string.")
        for domain in _COMPONENT_DOMAINS:
            value = getattr(self, domain)
            if value is None:
                if domain == "supervoxelizer":
                    continue
                raise HABITAPIError(
                    f"HabitatSpec requires a '{domain}' component spec."
                )
            if not isinstance(value, Spec):
                raise HABITAPIError(
                    f"HabitatSpec.{domain} must be a Spec; got {type(value).__name__}."
                )
        for chain_field in (
            "habitat_features",
            "subject_table_preprocessors",
            "group_table_preprocessors",
        ):
            chain = tuple(getattr(self, chain_field))
            object.__setattr__(self, chain_field, chain)
            for entry in chain:
                if not isinstance(entry, Spec):
                    raise HABITAPIError(
                        f"Every entry of HabitatSpec.{chain_field} must be a Spec."
                    )
        if self.random_seed is not None:
            object.__setattr__(self, "random_seed", int(self.random_seed))

    def component_specs(self) -> Mapping[str, Optional[Spec]]:
        """Return the four pipeline component specs keyed by domain name."""
        return {
            "voxel_feature_extractor": self.voxel_feature_extractor,
            "supervoxelizer": self.supervoxelizer,
            "habitat_model_fitter": self.habitat_model_fitter,
            "habitat_assigner": self.habitat_assigner,
        }

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
        body: list[str] = [
            f"The analysis specification {self.name!r} comprises "
            f"{'; '.join(_component_phrases(self.to_dict()))}."
        ]
        if self.random_seed is not None:
            body.append(
                f"Random seed {self.random_seed} is fixed for every "
                "stochastic component."
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
        for domain, component in self.component_specs().items():
            payload[domain] = component.to_dict() if component is not None else None
        payload["habitat_features"] = [
            feature.to_dict() for feature in self.habitat_features
        ]
        payload["subject_table_preprocessors"] = [
            entry.to_dict() for entry in self.subject_table_preprocessors
        ]
        payload["group_table_preprocessors"] = [
            entry.to_dict() for entry in self.group_table_preprocessors
        ]
        payload["random_seed"] = self.random_seed
        return payload

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
        components: Dict[str, Optional[Spec]] = {}
        for domain in _COMPONENT_DOMAINS:
            raw = payload.get(domain)
            components[domain] = Spec.from_dict(raw) if raw is not None else None
        features = tuple(
            Spec.from_dict(item) for item in payload.get("habitat_features", ())
        )
        subject_chain = tuple(
            Spec.from_dict(item)
            for item in payload.get("subject_table_preprocessors", ())
        )
        group_chain = tuple(
            Spec.from_dict(item)
            for item in payload.get("group_table_preprocessors", ())
        )
        return cls(
            name=str(payload.get("name", "habitat_spec")),
            voxel_feature_extractor=components["voxel_feature_extractor"],
            supervoxelizer=components["supervoxelizer"],
            habitat_model_fitter=components["habitat_model_fitter"],
            habitat_assigner=components["habitat_assigner"],
            habitat_features=features,
            subject_table_preprocessors=subject_chain,
            group_table_preprocessors=group_chain,
            random_seed=payload.get("random_seed"),
            version=str(payload.get("version", "1.0")),
        )
