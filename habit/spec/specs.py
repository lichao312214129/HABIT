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
        version: Specification schema version.
    """

    name: str
    voxel_feature_extractor: Spec
    supervoxelizer: Optional[Spec]
    habitat_model_fitter: Spec
    habitat_assigner: Spec
    habitat_features: Tuple[Spec, ...] = ()
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
        object.__setattr__(self, "habitat_features", tuple(self.habitat_features))
        for feature_spec in self.habitat_features:
            if not isinstance(feature_spec, Spec):
                raise HABITAPIError(
                    "Every entry of HabitatSpec.habitat_features must be a Spec."
                )

    def component_specs(self) -> Mapping[str, Optional[Spec]]:
        """Return the four pipeline component specs keyed by domain name."""
        return {
            "voxel_feature_extractor": self.voxel_feature_extractor,
            "supervoxelizer": self.supervoxelizer,
            "habitat_model_fitter": self.habitat_model_fitter,
            "habitat_assigner": self.habitat_assigner,
        }

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
        return cls(
            name=str(payload.get("name", "habitat_spec")),
            voxel_feature_extractor=components["voxel_feature_extractor"],
            supervoxelizer=components["supervoxelizer"],
            habitat_model_fitter=components["habitat_model_fitter"],
            habitat_assigner=components["habitat_assigner"],
            habitat_features=features,
            version=str(payload.get("version", "1.0")),
        )
