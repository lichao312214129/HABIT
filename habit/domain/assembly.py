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
from habit.domain.protocols import Seedable
from habit.domain.supervoxel import SupervoxelizerRegistry
from habit.domain.supervoxel_features import SupervoxelFeatureExtractorRegistry
from habit.domain.table_preprocessing import TablePreprocessorRegistry
from habit.domain.table_protocols import Metric
from habit.domain.voxel_features import VoxelFeatureExtractorRegistry
from habit.spec.specs import HabitatSpec, MLSpec, Spec

__all__ = [
    "HabitatComponents",
    "build_habitat_components",
    "build_subject_chain",
    "build_table_pipeline",
    "build_ml_metrics",
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

    Attributes:
        voxel_extractor: Produces the per-voxel feature field.
        supervoxelizer: Groups voxels into supervoxels; ``None`` means the
            design clusters voxels directly.
        supervoxel_extractor: Describes each supervoxel; ``None`` when the
            supervoxel's own feature means are used.
        voxel_chain: Per-subject preprocessing of voxel features.
        supervoxel_chain: Per-subject preprocessing of supervoxel features.
        cohort_chain: Cohort-level preprocessing; the only leakage-sensitive
            step in habitat definition.
        fitter: Learns the habitat definition from clustering units.
        extractors: Habitat feature families to compute after assignment.
            May be empty: defining habitats and describing them are separate
            acts, and the v0.1 ``habitat`` command performs only the first.
    """

    voxel_extractor: Any
    supervoxelizer: Optional[Any]
    supervoxel_extractor: Optional[Any]
    voxel_chain: Optional[SubjectPreprocessingChain]
    supervoxel_chain: Optional[SubjectPreprocessingChain]
    cohort_chain: Optional[CohortPreprocessingChain]
    fitter: Any
    extractors: Tuple[Any, ...]

    def pipeline(self, *, assigner: Optional[Any]) -> SubjectPipeline:
        """
        Assemble the subject pipeline.

        Args:
            assigner: Fitted assigner for prediction, or ``None`` to build the
                fit-time pipeline that only produces clustering units.

        Returns:
            The pipeline. The cohort chain is attached only when an assigner
            is present: at fit time the chain is not yet fitted, and at
            predict time it must run before assignment.
        """
        return SubjectPipeline(
            voxel_feature_extractor=self.voxel_extractor,
            supervoxelizer=self.supervoxelizer,
            habitat_assigner=assigner,
            supervoxel_feature_extractor=self.supervoxel_extractor,
            voxel_feature_preprocessor=self.voxel_chain,
            supervoxel_feature_preprocessor=self.supervoxel_chain,
            cohort_feature_preprocessor=(
                self.cohort_chain if assigner is not None else None
            ),
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
    voxel_extractor = VoxelFeatureExtractorRegistry.create(
        spec.voxel_feature_extractor.name,
        **spec.voxel_feature_extractor.params,
    )
    supervoxelizer = None
    if spec.supervoxelizer is not None:
        supervoxelizer = SupervoxelizerRegistry.create(
            spec.supervoxelizer.name, **spec.supervoxelizer.params
        )
    supervoxel_extractor = None
    if spec.supervoxel_feature_extractor is not None:
        supervoxel_extractor = SupervoxelFeatureExtractorRegistry.create(
            spec.supervoxel_feature_extractor.name,
            **spec.supervoxel_feature_extractor.params,
        )
    voxel_chain = build_subject_chain(spec.voxel_feature_preprocessors)
    supervoxel_chain = build_subject_chain(spec.supervoxel_feature_preprocessors)
    cohort_chain = None
    if spec.cohort_feature_preprocessors:
        cohort_chain = CohortPreprocessingChain(
            build_methods(list(spec.cohort_feature_preprocessors))
        )
    fitter = HabitatModelFitterRegistry.create(
        spec.habitat_model_fitter.name, **spec.habitat_model_fitter.params
    )
    extractors = tuple(
        HabitatFeatureExtractorRegistry.create(
            feature_spec.name, **feature_spec.params
        )
        for feature_spec in spec.habitat_features
    )
    if spec.random_seed is not None:
        for component in (
            voxel_extractor,
            supervoxelizer,
            supervoxel_extractor,
            voxel_chain,
            supervoxel_chain,
            cohort_chain,
            fitter,
            *extractors,
        ):
            if component is None:
                continue
            setter = getattr(component, "set_random_state", None)
            if isinstance(component, Seedable) or callable(setter):
                component.set_random_state(spec.random_seed)
    return HabitatComponents(
        voxel_extractor=voxel_extractor,
        supervoxelizer=supervoxelizer,
        supervoxel_extractor=supervoxel_extractor,
        voxel_chain=voxel_chain,
        supervoxel_chain=supervoxel_chain,
        cohort_chain=cohort_chain,
        fitter=fitter,
        extractors=extractors,
    )


def build_table_pipeline(spec: MLSpec) -> TablePipeline:
    """
    Instantiate the tabular modelling pipeline declared by ``spec``.

    Args:
        spec: The modelling declaration, with any caller overrides already
            folded in.

    Returns:
        The pipeline: preprocessing steps first, then feature selectors,
        then the terminal classifier, seeded from ``spec.random_seed`` when
        it is set.

    Raises:
        ComponentNotFoundError: If a spec names an unregistered component.
        ConfigurationError: If a component's parameters fail validation.
    """
    steps = [
        TablePreprocessorRegistry.create(entry.name, **entry.params)
        for entry in spec.table_preprocessors
    ]
    steps.extend(
        FeatureSelectorRegistry.create(entry.name, **entry.params)
        for entry in spec.feature_selectors
    )
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
