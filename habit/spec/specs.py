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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast

from habit.exceptions import HABITAPIError

__all__ = ["Spec", "HabitatSpec", "MLSpec"]

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
    ("supervoxel_feature_extractor", "supervoxel feature extraction"),
    ("habitat_model_fitter", "habitat model fitting"),
    ("habitat_assigner", "habitat assignment"),
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
    voxel_feature_extractor: Spec
    supervoxelizer: Optional[Spec]
    habitat_model_fitter: Spec
    habitat_assigner: Spec
    supervoxel_feature_extractor: Optional[Spec] = None
    habitat_features: Tuple[Spec, ...] = ()
    voxel_feature_preprocessors: Tuple[Spec, ...] = ()
    supervoxel_feature_preprocessors: Tuple[Spec, ...] = ()
    cohort_feature_preprocessors: Tuple[Spec, ...] = ()
    random_seed: Optional[int] = None
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Coerce component payloads into Spec instances and tuples."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise HABITAPIError("HabitatSpec.name must be a non-empty string.")
        for domain in _COMPONENT_DOMAINS:
            value = getattr(self, domain)
            if value is None:
                if domain in _OPTIONAL_COMPONENT_DOMAINS:
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
            *(key for key, _ in _PREPROCESSING_CHAINS),
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
        """Return the pipeline component specs keyed by domain name."""
        return {
            "voxel_feature_extractor": self.voxel_feature_extractor,
            "supervoxelizer": self.supervoxelizer,
            "supervoxel_feature_extractor": self.supervoxel_feature_extractor,
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
        for chain_key, _ in _PREPROCESSING_CHAINS:
            payload[chain_key] = [
                entry.to_dict() for entry in getattr(self, chain_key)
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
        chains = {
            chain_key: tuple(
                Spec.from_dict(item) for item in payload.get(chain_key, ())
            )
            for chain_key, _ in _PREPROCESSING_CHAINS
        }
        # ``components`` holds Optional[Spec] because a hand-written document
        # may omit a domain; ``__post_init__`` rejects a missing REQUIRED
        # component with HABITAPIError, so the casts below only cross the
        # static gap -- invalid payloads still fail at construction.
        return cls(
            name=str(payload.get("name", "habitat_spec")),
            voxel_feature_extractor=cast(Spec, components["voxel_feature_extractor"]),
            supervoxelizer=components["supervoxelizer"],
            supervoxel_feature_extractor=components["supervoxel_feature_extractor"],
            habitat_model_fitter=cast(Spec, components["habitat_model_fitter"]),
            habitat_assigner=cast(Spec, components["habitat_assigner"]),
            habitat_features=features,
            random_seed=payload.get("random_seed"),
            version=str(payload.get("version", "1.0")),
            **chains,
        )


#: Tabular preprocessing / selection chains of an MLSpec, keyed by field
#: name, with the prose each renders as. Ordered as they run: selection may
#: happen BEFORE preprocessing (the stage v0.1 expressed as
#: ``before_z_score: true`` -- scientifically meaningful whenever a
#: selector's statistics are distorted by normalisation, e.g. variance-based
#: selection is vacuous after z-scoring), preprocessing itself, then the
#: ordinary post-preprocessing selection. A TablePipeline fits steps in
#: chain order, so this tuple order IS the execution order.
_ML_CHAINS: Tuple[Tuple[str, str], ...] = (
    ("pre_preprocessing_feature_selectors", "pre-preprocessing feature selection"),
    ("table_preprocessors", "table preprocessing"),
    ("feature_selectors", "feature selection"),
)


def _ml_phrases(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    """
    Render an MLSpec-shaped payload as ordered prose phrases.

    Args:
        payload: Spec payload as produced by ``MLSpec.to_dict``.

    Returns:
        One phrase per present chain or component, in pipeline order.
    """
    phrases: list[str] = []
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
    definition: an optional pre-preprocessing selection chain, an ordered
    table-preprocessing chain, an ordered post-preprocessing selection
    chain, exactly one terminal classifier, and the evaluation metric
    panel. It deliberately does NOT describe the validation design (split
    counts, resampling, id files) -- those are choices of the calling
    recipe, not of the model definition.

    Attributes:
        name: Human-readable specification name.
        classifier: Spec of the terminal classifier.
        pre_preprocessing_feature_selectors: Ordered specs of the selection
            chain fitted on the RAW training table, BEFORE any
            preprocessing (v0.1's ``before_z_score: true`` selectors). The
            stage exists because some selection statistics are distorted by
            normalisation -- after z-scoring every feature variance is 1.0,
            so variance-based selection only carries information on the raw
            table. The preprocessors then fit on the SELECTED training
            features, exactly as v0.1's two-stage pipeline did.
        table_preprocessors: Ordered specs of the stateful preprocessing
            chain fitted on the TRAINING rows and replayed afterwards
            (v0.1's ``normalization``).
        feature_selectors: Ordered specs of the selection chain, fitted
            after preprocessing (v0.1's ``feature_selection_methods``
            entries without ``before_z_score``).
        metrics: Specs of the evaluation metric panel. An empty tuple asks
            the calling recipe for its default panel.
        random_seed: Seed applied to every
            :class:`~habit.domain.protocols.Seedable` component. Seeds
            change the scientific result, so they live in the spec (and its
            fingerprint), not in the run policy.
        version: Specification schema version.
    """

    name: str
    classifier: Spec
    pre_preprocessing_feature_selectors: Tuple[Spec, ...] = ()
    table_preprocessors: Tuple[Spec, ...] = ()
    feature_selectors: Tuple[Spec, ...] = ()
    metrics: Tuple[Spec, ...] = ()
    random_seed: Optional[int] = None
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Coerce component payloads into Spec instances and tuples."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise HABITAPIError("MLSpec.name must be a non-empty string.")
        if not isinstance(self.classifier, Spec):
            raise HABITAPIError(
                "MLSpec.classifier must be a Spec; got "
                f"{type(self.classifier).__name__}."
            )
        for chain_field, _ in _ML_CHAINS + (("metrics", ""),):
            chain = tuple(getattr(self, chain_field))
            object.__setattr__(self, chain_field, chain)
            for entry in chain:
                if not isinstance(entry, Spec):
                    raise HABITAPIError(
                        f"Every entry of MLSpec.{chain_field} must be a Spec."
                    )
        if self.random_seed is not None:
            object.__setattr__(self, "random_seed", int(self.random_seed))

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
        """Serialise to a plain dict (YAML isomorphic)."""
        payload: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }
        for chain_key, _ in _ML_CHAINS:
            payload[chain_key] = [
                entry.to_dict() for entry in getattr(self, chain_key)
            ]
        payload["classifier"] = self.classifier.to_dict()
        payload["metrics"] = [entry.to_dict() for entry in self.metrics]
        payload["random_seed"] = self.random_seed
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MLSpec":
        """
        Rebuild a machine-learning specification from its dict form.

        Args:
            payload: Mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed specification.

        Raises:
            HABITAPIError: If the classifier component is missing.
        """
        chains = {
            chain_key: tuple(
                Spec.from_dict(item) for item in payload.get(chain_key, ())
            )
            for chain_key, _ in _ML_CHAINS + (("metrics", ""),)
        }
        classifier = payload.get("classifier")
        return cls(
            name=str(payload.get("name", "ml_spec")),
            classifier=cast(Spec, Spec.from_dict(classifier) if classifier is not None else None),
            random_seed=payload.get("random_seed"),
            version=str(payload.get("version", "1.0")),
            **chains,
        )
