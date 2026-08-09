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
"""Turn a :class:`~habit.spec.specs.HabitatSpec` into live components (L3).

A spec is a declaration; this module is the single place that reads it and
constructs the corresponding objects through the domain registries. There is
exactly one construction site on purpose: a spec field with no construction
site is worse than an unsupported one, because the analysis runs, the
provenance records the step, and the numbers come from a pipeline that never
applied it. Both the recipe layer and the scikit-learn adapter build their
components here, so the two entry points cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from habit.domain.classification import ClassifierRegistry
from habit.domain.evaluation import MetricRegistry
from habit.domain.feature_preprocessing import (
    CohortPreprocessingChain,
    SubjectPreprocessingChain,
    build_methods,
)
from habit.domain.feature_selection import FeatureSelectorRegistry
from habit.domain.habitat_features import HabitatFeatureExtractorRegistry
from habit.domain.habitat_model import HabitatModelFitterRegistry
from habit.domain.pipeline import SubjectPipeline, TablePipeline
from habit.domain.postprocess import (
    ConnectedComponentPostprocess,
    build_connected_component_postprocess,
)
from habit.domain.protocols import Seedable
from habit.domain.supervoxel import SupervoxelizerRegistry
from habit.domain.supervoxel_features import SupervoxelFeatureExtractorRegistry
from habit.domain.table_preprocessing import TablePreprocessorRegistry
from habit.domain.table_protocols import Metric
from habit.domain.trees import (
    build_habitat_extractor,
    build_supervoxel_extractor,
    build_voxel_extractor,
)
from habit.domain.voxel_features import VoxelFeatureExtractorRegistry
from habit.exceptions import ComponentNotFoundError, HABITAPIError
from habit.registry.core import ComponentRegistry
from habit.spec.specs import HabitatSpec, MLSpec, Spec
from habit.utils.log_utils import get_module_logger

_logger = get_module_logger(__name__)

__all__ = [
    "HabitatComponents",
    "build_habitat_components",
    "build_subject_chain",
    "build_table_step",
    "build_table_pipeline",
    "build_ml_metrics",
    "validate_habitat_spec_registry",
]


def build_subject_chain(
    steps: Sequence[Any],
) -> Optional[SubjectPreprocessingChain]:
    """
    Build a stateless preprocessing chain, or ``None`` for an empty spec.

    Args:
        steps: Ordered method specs from a ``HabitatSpec`` chain field.

    Returns:
        The chain, or ``None`` when no step was configured. ``None`` and an
        empty chain are deliberately distinct: the chains reject emptiness so
        that "no preprocessing" is stated once, in the spec.
    """
    if not steps:
        return None
    return SubjectPreprocessingChain(build_methods(list(steps)))


@dataclass(frozen=True)
class HabitatComponents:
    """
    Every component a habitat spec declares, built once and reused.

    Grouping them is what makes :meth:`pipeline` possible, and that method is
    the point: fit and predict must assemble the SAME stages in the SAME
    order, differing only in whether an assigner is attached.

    Attribute names match :class:`~habit.spec.specs.HabitatSpec` fields and
    :class:`~habit.domain.pipeline.SubjectPipeline` parameters (assembled
    preprocessor chains use the singular form, as on the pipeline).

    Attributes:
        voxel_feature_extractor: Produces the per-voxel feature field.
        supervoxelizer: Groups voxels into supervoxels; ``None`` means the
            design clusters voxels directly.
        supervoxel_feature_extractor: Describes each supervoxel; ``None``
            when the supervoxelizer's own feature means are used.
        voxel_feature_preprocessor: Per-subject preprocessing of voxel
            features (assembled chain), or ``None``.
        supervoxel_feature_preprocessor: Per-subject preprocessing of
            supervoxel features, or ``None``.
        cohort_feature_preprocessor: Cohort-level preprocessing; the only
            leakage-sensitive step in habitat definition.
        habitat_model_fitter: Learns the habitat definition from clustering
            units.
        habitat_features: Habitat feature families to compute after
            assignment. May be empty: defining habitats and describing them
            are separate acts, and the v0.1 ``habitat`` command performs
            only the first.
        on_geometry_mismatch: Image/mask geometry policy forwarded to
            :class:`~habit.domain.pipeline.SubjectPipeline` (default
            ``"resample_mask"``).
        postprocess_supervoxel: Optional supervoxel label cleanup.
        postprocess_habitat: Optional final habitat label cleanup.
    """

    voxel_feature_extractor: Any
    supervoxelizer: Optional[Any]
    supervoxel_feature_extractor: Optional[Any]
    voxel_feature_preprocessor: Optional[SubjectPreprocessingChain]
    supervoxel_feature_preprocessor: Optional[SubjectPreprocessingChain]
    cohort_feature_preprocessor: Optional[CohortPreprocessingChain]
    habitat_model_fitter: Any
    habitat_features: Tuple[Any, ...]
    on_geometry_mismatch: str = "resample_mask"
    postprocess_supervoxel: Optional[ConnectedComponentPostprocess] = None
    postprocess_habitat: Optional[ConnectedComponentPostprocess] = None

    def pipeline(
        self,
        *,
        assigner: Optional[Any],
        observer: Optional[Any] = None,
    ) -> SubjectPipeline:
        """
        Assemble the subject pipeline.

        Args:
            assigner: Fitted assigner for prediction, or ``None`` to build the
                fit-time pipeline that only produces clustering units.
            observer: Optional step observer for debugging / QA. Never part of
                the analysis fingerprint.

        Returns:
            The pipeline. The cohort chain is attached only when an assigner
            is present: at fit time the chain is not yet fitted, and at
            predict time it must run before assignment.
        """
        return SubjectPipeline(
            voxel_feature_extractor=self.voxel_feature_extractor,
            supervoxelizer=self.supervoxelizer,
            habitat_assigner=assigner,
            supervoxel_feature_extractor=self.supervoxel_feature_extractor,
            voxel_feature_preprocessor=self.voxel_feature_preprocessor,
            supervoxel_feature_preprocessor=self.supervoxel_feature_preprocessor,
            cohort_feature_preprocessor=(
                self.cohort_feature_preprocessor if assigner is not None else None
            ),
            on_geometry_mismatch=self.on_geometry_mismatch,
            postprocess_supervoxel=self.postprocess_supervoxel,
            postprocess_habitat=self.postprocess_habitat,
            observer=observer,
        )


def build_habitat_components(spec: HabitatSpec) -> HabitatComponents:
    """
    Instantiate every pipeline component declared by ``spec``.

    Args:
        spec: The analysis declaration, with any caller overrides already
            folded in.

    Returns:
        The constructed components, seeded from ``spec.random_seed`` when it
        is set.

    Raises:
        ComponentNotFoundError: If a spec names an unregistered component.
        ConfigurationError: If a component's parameters fail validation.
    """
    voxel_feature_extractor = build_voxel_extractor(spec.voxel_feature_extractor)
    supervoxelizer = None
    if spec.supervoxelizer is not None:
        supervoxelizer = SupervoxelizerRegistry.create(
            spec.supervoxelizer.name, **spec.supervoxelizer.params
        )
    supervoxel_feature_extractor = None
    if spec.supervoxel_feature_extractor is not None:
        supervoxel_feature_extractor = build_supervoxel_extractor(
            spec.supervoxel_feature_extractor
        )
    voxel_feature_preprocessor = build_subject_chain(
        spec.voxel_feature_preprocessors
    )
    supervoxel_feature_preprocessor = build_subject_chain(
        spec.supervoxel_feature_preprocessors
    )
    cohort_feature_preprocessor = None
    if spec.cohort_feature_preprocessors:
        cohort_feature_preprocessor = CohortPreprocessingChain(
            build_methods(list(spec.cohort_feature_preprocessors))
        )
    habitat_model_fitter = HabitatModelFitterRegistry.create(
        spec.habitat_model_fitter.name, **spec.habitat_model_fitter.params
    )
    habitat_features = tuple(
        build_habitat_extractor(feature_spec)
        for feature_spec in spec.habitat_features
    )
    postprocess_supervoxel = build_connected_component_postprocess(
        spec.postprocess_supervoxel
    )
    postprocess_habitat = build_connected_component_postprocess(
        spec.postprocess_habitat
    )
    if postprocess_supervoxel is not None and supervoxelizer is None:
        _logger.warning(
            "HabitatSpec.postprocess_supervoxel is set but no supervoxelizer "
            "is configured; supervoxel cleanup is ignored."
        )
        postprocess_supervoxel = None
    if spec.random_seed is not None:
        for component in (
            voxel_feature_extractor,
            supervoxelizer,
            supervoxel_feature_extractor,
            voxel_feature_preprocessor,
            supervoxel_feature_preprocessor,
            cohort_feature_preprocessor,
            habitat_model_fitter,
            *habitat_features,
        ):
            if component is None:
                continue
            setter = getattr(component, "set_random_state", None)
            if isinstance(component, Seedable) or callable(setter):
                component.set_random_state(spec.random_seed)
    return HabitatComponents(
        voxel_feature_extractor=voxel_feature_extractor,
        supervoxelizer=supervoxelizer,
        supervoxel_feature_extractor=supervoxel_feature_extractor,
        voxel_feature_preprocessor=voxel_feature_preprocessor,
        supervoxel_feature_preprocessor=supervoxel_feature_preprocessor,
        cohort_feature_preprocessor=cohort_feature_preprocessor,
        habitat_model_fitter=habitat_model_fitter,
        habitat_features=habitat_features,
        on_geometry_mismatch=spec.on_geometry_mismatch,
        postprocess_supervoxel=postprocess_supervoxel,
        postprocess_habitat=postprocess_habitat,
    )


def _require_registered_name(spec_entry: Spec, registry: ComponentRegistry[Any]) -> None:
    """
    Verify one spec entry names a registered implementation.

    Args:
        spec_entry: Component declaration from a :class:`HabitatSpec`.
        registry: Registry that should contain ``spec_entry.name``.

    Raises:
        ComponentNotFoundError: When the name is absent from ``registry``.
    """
    if registry.get(spec_entry.name) is None:
        raise ComponentNotFoundError(
            f"Unknown {registry.kind} {spec_entry.name!r} in domain "
            f"{registry.domain!r}. Available: {registry.available()}"
        )


def _require_registered_tree(spec_entry: Spec, registry: ComponentRegistry[Any]) -> None:
    """
    Verify one feature-tree node (and its descendants) is registered.

    A node carrying ``children`` is a combiner node: its own name must be
    registered in the combiner domain and every child is validated
    recursively against the granularity's leaf registry (leaves) or the
    combiner registry again (nested combiners).

    Args:
        spec_entry: Feature-tree node from a :class:`HabitatSpec`.
        registry: Leaf-extractor registry of the node's granularity.

    Raises:
        ComponentNotFoundError: When any node name is unknown.
    """
    from habit.domain.combiners import CombinerRegistry

    if "children" in spec_entry.params:
        _require_registered_name(spec_entry, CombinerRegistry)
        if not spec_entry.params["children"]:
            raise ComponentNotFoundError(
                f"Combiner {spec_entry.name!r} in domain {registry.domain!r} "
                "requires a non-empty 'children' list."
            )
        for child in spec_entry.params["children"]:
            _require_registered_tree(Spec.from_dict(child), registry)
        return
    _require_registered_name(spec_entry, registry)


def validate_habitat_spec_registry(spec: HabitatSpec) -> None:
    """
    Verify every component declared by a habitat spec is registered.

    Intended for ``habit check-config`` and other pre-run validation paths so
    unknown component names fail before a long pipeline job starts.

    Args:
        spec: Parsed habitat analysis specification.

    Raises:
        ComponentNotFoundError: When any declared component name is unknown.
    """
    from habit.domain.assignment.registry import HabitatAssignerRegistry
    from habit.domain.feature_preprocessing.registry import (
        FeaturePreprocessingMethodRegistry,
    )

    _require_registered_tree(
        spec.voxel_feature_extractor, VoxelFeatureExtractorRegistry
    )
    if spec.supervoxelizer is not None:
        _require_registered_name(spec.supervoxelizer, SupervoxelizerRegistry)
    if spec.supervoxel_feature_extractor is not None:
        _require_registered_tree(
            spec.supervoxel_feature_extractor,
            SupervoxelFeatureExtractorRegistry,
        )
    for step in spec.voxel_feature_preprocessors:
        _require_registered_name(step, FeaturePreprocessingMethodRegistry)
    for step in spec.supervoxel_feature_preprocessors:
        _require_registered_name(step, FeaturePreprocessingMethodRegistry)
    for step in spec.cohort_feature_preprocessors:
        _require_registered_name(step, FeaturePreprocessingMethodRegistry)
    _require_registered_name(spec.habitat_model_fitter, HabitatModelFitterRegistry)
    _require_registered_name(spec.habitat_assigner, HabitatAssignerRegistry)
    for feature_spec in spec.habitat_features:
        _require_registered_tree(feature_spec, HabitatFeatureExtractorRegistry)
    for field_name in ("postprocess_supervoxel", "postprocess_habitat"):
        entry = getattr(spec, field_name)
        if entry is None:
            continue
        if entry.name != "connected_components":
            raise ComponentNotFoundError(
                f"HabitatSpec.{field_name} must be named 'connected_components'; "
                f"got {entry.name!r}."
            )


#: Registries a name in ``MLSpec.steps`` may resolve against, in lookup
#: order. The two vocabularies are disjoint by construction (``variance``
#: versus ``variance_filter``, ``correlation`` versus
#: ``correlation_filter``), so the order is documentation rather than
#: precedence -- and :func:`build_table_step` refuses to guess if that ever
#: stops being true.
_TABLE_STEP_REGISTRIES: Tuple[Tuple[str, ComponentRegistry], ...] = (
    ("table_preprocessor", TablePreprocessorRegistry),
    ("feature_selector", FeatureSelectorRegistry),
)


def build_table_step(entry: Spec) -> Any:
    """
    Build one step of ``MLSpec.steps`` by resolving its name.

    An ordered step list carries no per-entry "kind" tag, because the tag
    would be a second source of truth for something the registries already
    know. The name alone therefore has to identify the component, which it
    does: the preprocessor and selector vocabularies do not overlap.

    Args:
        entry: One step spec from :attr:`habit.spec.specs.MLSpec.steps`.

    Returns:
        Any: The constructed table preprocessor or feature selector. Both
        satisfy the fit/transform contract ``TablePipeline`` needs, which is
        precisely why one ordered list can hold either.

    Raises:
        ComponentNotFoundError: If no registry knows the name. The message
            names both vocabularies, since a user reading it does not
            necessarily know which one their step belongs to.
        HABITAPIError: If BOTH registries know the name. Silently preferring
            one would run a different algorithm than the other reading of
            the same YAML, so the ambiguity is reported instead.
    """
    matches = [
        (kind, registry)
        for kind, registry in _TABLE_STEP_REGISTRIES
        if entry.name in registry.available()
    ]
    if len(matches) > 1:
        raise HABITAPIError(
            f"Table step {entry.name!r} is registered as "
            f"{[kind for kind, _ in matches]}. An ordered step list "
            "identifies a component by name alone, so an ambiguous name "
            "cannot be resolved; rename one of the registrations."
        )
    if not matches:
        raise ComponentNotFoundError(
            f"No table preprocessor or feature selector named "
            f"{entry.name!r}. Registered table preprocessors: "
            f"{sorted(TablePreprocessorRegistry.available())}; registered "
            f"feature selectors: {sorted(FeatureSelectorRegistry.available())}."
        )
    _, registry = matches[0]
    return registry.create(entry.name, **entry.params)


def build_table_pipeline(spec: MLSpec) -> TablePipeline:
    """
    Instantiate the tabular modelling pipeline declared by ``spec``.

    Reads :attr:`~habit.spec.specs.MLSpec.steps`, the single ordered step
    list. A spec declared through the deprecated three-chain fields has
    already been folded into that list by ``MLSpec`` itself, in the
    documented order (pre-preprocessing selection, preprocessing,
    post-preprocessing selection), so both layouts assemble through this one
    path and cannot drift.

    Args:
        spec: The modelling declaration, with any caller overrides already
            folded in.

    Returns:
        The pipeline: every declared step in ``spec.steps`` order, then the
        terminal classifier -- seeded from ``spec.random_seed`` when it is
        set. ``TablePipeline.fit`` runs steps in this order on the training
        rows only, so every selector's statistics come from the training
        split regardless of where in the chain it sits.

    Raises:
        ComponentNotFoundError: If a spec names an unregistered component.
        HABITAPIError: If a step name is ambiguous across the two registries.
        ConfigurationError: If a component's parameters fail validation.
    """
    steps = [build_table_step(entry) for entry in spec.steps]
    model = ClassifierRegistry.create(spec.classifier.name, **spec.classifier.params)
    pipeline = TablePipeline(steps=steps, model=model)
    if spec.random_seed is not None:
        pipeline.set_random_state(spec.random_seed)
    return pipeline


def build_ml_metrics(
    spec: MLSpec, *, default_names: Sequence[str] = ()
) -> Tuple[Metric, ...]:
    """
    Instantiate the evaluation metric panel declared by ``spec``.

    Args:
        spec: The modelling declaration.
        default_names: Registered metric names to build when the spec
            declares no metric panel. The default is empty -- an empty panel
            is a statement, and only a recipe may substitute its own
            defaults.

    Returns:
        The metric instances, in spec order.

    Raises:
        ComponentNotFoundError: If a spec names an unregistered metric.
        ConfigurationError: If a metric's parameters fail validation.
    """
    metric_specs = spec.metrics or tuple(
        # Defaults are bare names: recipe-level panels never smuggle
        # parameters past the spec.
        Spec(name=name)
        for name in default_names
    )
    return tuple(
        MetricRegistry.create(entry.name, **entry.params) for entry in metric_specs
    )
