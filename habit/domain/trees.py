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
"""Feature composition trees: recursive evaluation of nested feature specs.

A feature spec is a TREE: leaf nodes are extractors (their elements are
modality names) and internal nodes are combiners (their elements are child
nodes listed under the ``children`` parameter). This module holds the three
granularity-specific wrappers that evaluate such a tree -- voxel,
supervoxel, habitat -- plus the ``build_*`` constructors the assembly layer
uses instead of calling the leaf registries directly.

The wrappers implement the existing level protocols
(:class:`~habit.domain.protocols.VoxelFeatureExtractor` and siblings), so
pipelines, recipes, seeding, and provenance see no difference between a
single extractor and a composed tree. Combiners themselves only ever see
plain ``DataFrame`` blocks, which is what makes one combiner implementation
reusable at every granularity.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.contracts.habitat import (
    HabitatMap,
    Supervoxelization,
    VoxelFeatureField,
)
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.combiners import CombinerRegistry
from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.domain.protocols import Seedable
from habit.domain.supervoxel_features.registry import (
    SupervoxelFeatureExtractorRegistry,
)
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.exceptions import ComponentNotFoundError, HABITAPIError
from habit.spec.specs import Spec

__all__ = [
    "VoxelFeatureTree",
    "SupervoxelFeatureTree",
    "HabitatFeatureTree",
    "build_voxel_extractor",
    "build_supervoxel_extractor",
    "build_habitat_extractor",
]


def _normalise_spec(entry: Any, *, owner: str) -> Spec:
    """
    Accept a child node as a :class:`Spec` or a ``{'name', 'params'}`` dict.

    Args:
        entry: Child node payload.
        owner: Parent node name used in error messages.

    Returns:
        The child as a Spec.

    Raises:
        HABITAPIError: For any other payload type.
    """
    if isinstance(entry, Spec):
        return entry
    if isinstance(entry, Mapping):
        return Spec.from_dict(entry)
    raise HABITAPIError(
        f"{owner}: a child node must be a Spec or a mapping with 'name' and "
        f"'params'; got {type(entry).__name__}."
    )


def _child_source_label(spec: Spec) -> str:
    """
    Return the source label one child node contributes to its combiner.

    Resolution order: the ``as_`` alias when set, then the ``modality``
    parameter (the leaf's input source), then the node name. Combiner
    parameters keyed by child (e.g. ``weights``) match against this label.
    """
    alias = spec.params.get("as_")
    if alias:
        return str(alias)
    modality = spec.params.get("modality")
    if modality:
        return str(modality)
    return spec.name


def _split_tree_params(
    spec: Spec,
) -> Tuple[Dict[str, Any], List[Spec], Optional[str], Optional[str]]:
    """
    Split a combiner node's params into combiner params and child specs.

    ``children`` holds the tree structure; ``as_`` and ``roi`` are
    tree-level concerns handled by the wrapper (output rename and ROI
    override); everything else is forwarded to the combiner constructor.
    The legacy passthrough keys ``modalities`` / ``expression`` -- recorded
    by config translation for provenance -- are dropped so combiner params
    models (``extra="forbid"``) do not reject them.

    Args:
        spec: The combiner node's spec.

    Returns:
        ``(combiner_params, child_specs, alias, roi)``.

    Raises:
        HABITAPIError: If the node has no children.
    """
    params = dict(spec.params)
    raw_children = params.pop("children", None)
    alias = params.pop("as_", None)
    roi = params.pop("roi", None)
    params.pop("modalities", None)
    params.pop("expression", None)
    if not raw_children:
        raise HABITAPIError(
            f"{spec.name}: a combiner node requires a non-empty 'children' "
            "list of extractor/combiner specs."
        )
    children = [_normalise_spec(entry, owner=spec.name) for entry in raw_children]
    return (
        params,
        children,
        str(alias) if alias is not None else None,
        str(roi) if roi is not None else None,
    )


class _FeatureTreeBase:
    """
    Shared construction and seeding logic of the per-granularity wrappers.

    Args:
        spec: The combiner node's specification (kept verbatim for
            provenance: its fingerprint covers the whole subtree).
        build_child: Constructor turning each child spec into an evaluated
            child component (the per-granularity ``build_*`` function).
    """

    def __init__(self, spec: Spec, build_child: Callable[[Spec], Any]) -> None:
        if CombinerRegistry.get(spec.name) is None:
            raise HABITAPIError(
                f"{spec.name!r} is not a registered combiner; available: "
                f"{list(CombinerRegistry.available())}. A feature tree's "
                "internal nodes must be combiners, its leaves extractors."
            )
        combiner_params, child_specs, self._alias, self._roi = _split_tree_params(spec)
        self._spec = spec
        self._combiner = CombinerRegistry.create(spec.name, **combiner_params)
        self._children: List[Any] = [build_child(child) for child in child_specs]
        self._sources: Tuple[str, ...] = tuple(
            _child_source_label(child) for child in child_specs
        )
        if self._roi is not None:
            self._force_roi(self._roi)

    def _force_roi(self, roi: str) -> None:
        """
        Override the ROI of every leaf descendant, recursively.

        Mirrors the historical ``ConcatVoxelFeatures`` rule: the ROI is a
        tree-level decision, stated once at the root, so sibling leaves can
        never silently describe different regions.
        """
        for child in self._children:
            nested = getattr(child, "_force_roi", None)
            if callable(nested):
                nested(roi)
            elif hasattr(child, "roi"):
                child.roi = roi

    @property
    def spec(self) -> Spec:
        """Return the tree's specification, used for provenance."""
        return self._spec

    def set_random_state(self, seed: int) -> None:
        """
        Propagate the seed to every stochastic descendant.

        Combiners are deterministic by construction; leaf children
        implementing :class:`~habit.domain.protocols.Seedable` receive the
        seed, so a seeded tree behaves exactly like a seeded single
        extractor.
        """
        for child in self._children:
            if isinstance(child, Seedable):
                child.set_random_state(seed)

    def _context(self, subject_id: Optional[str] = None) -> Dict[str, Any]:
        """Assemble the combiner evaluation context for one call."""
        context: Dict[str, Any] = {"sources": self._sources}
        if subject_id is not None:
            context["subject_id"] = subject_id
        return context

    def _apply_alias(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Rename the merged block after the node-level ``as_`` alias.

        A combiner node's alias renames ONE output column (the common case:
        ``ratio(...)`` aliased to a study-specific name). Multi-column
        outputs keep their children's names, because renaming a matrix to
        one name would reintroduce the duplicate-column ambiguity the
        combiners guard against.
        """
        if self._alias is None:
            return frame
        if frame.shape[1] != 1:
            raise HABITAPIError(
                f"{self._spec.name}: 'as_' renames a single output column; "
                f"this combiner produced {frame.shape[1]} columns "
                f"({list(frame.columns)})."
            )
        renamed = frame.copy()
        renamed.columns = [self._alias]
        return renamed


class VoxelFeatureTree(_FeatureTreeBase):
    """
    Evaluate a voxel-level feature tree into one ``VoxelFeatureField``.

    Children are evaluated recursively; their fields must describe the same
    voxel population (identical ``voxel_index``), which is what makes
    row-aligned block combination meaningful. The merged block is wrapped
    back into a field carrying the shared geometry and a provenance edge.
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__(spec, build_voxel_extractor)

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute the composed voxel feature field for one subject.

        Args:
            subject: Subject providing images and the ROI mask.

        Returns:
            One row per ROI voxel; columns as produced by the combiner.

        Raises:
            HABITAPIError: If two children describe different voxel
                populations.
        """
        fields = [child(subject) for child in self._children]
        base = fields[0]
        for field in fields[1:]:
            if not np.array_equal(field.voxel_index, base.voxel_index):
                raise HABITAPIError(
                    f"{self._spec.name}: child fields disagree on the voxel "
                    "population (voxel_index differs). Siblings of a voxel "
                    "tree must share ROI and grid; check each child's roi "
                    "parameter."
                )
        blocks = [field.feature_frame() for field in fields]
        merged = self._combiner(
            blocks, context=self._context(subject.subject_id)
        )
        merged = self._apply_alias(merged)
        return base.with_feature_frame(
            merged,
            produced_by=f"combiner.{self._combiner.spec.name}",
            spec_fingerprint=self._spec.fingerprint(),
        )


class SupervoxelFeatureTree(_FeatureTreeBase):
    """
    Evaluate a supervoxel-level feature tree into one ``Supervoxelization``.

    Children describe the SAME partition (same ``label_array``) with
    different feature columns; the combiner merges those columns. Statistical
    leaves (``mean`` / ``std`` / ``percentile``) additionally need the
    subject's voxel feature field, supplied through :meth:`bind_fields` by
    the pipeline before the call.
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__(spec, build_supervoxel_extractor)
        self._working: Optional[VoxelFeatureField] = None
        self._original: Optional[VoxelFeatureField] = None

    def bind_fields(
        self,
        working: Optional[VoxelFeatureField] = None,
        original: Optional[VoxelFeatureField] = None,
    ) -> None:
        """
        Supply the voxel feature fields statistical leaves aggregate from.

        Propagates recursively so nested trees and direct statistical leaves
        both receive the fields. ``working`` is the field AFTER the
        subject-level preprocessing chain (the signal the supervoxelizer
        partitioned); ``original`` is the field BEFORE it, enabling
        ``source="original"`` aggregation.

        Args:
            working: Post-preprocessing voxel feature field, or None.
            original: Pre-preprocessing voxel feature field, or None.
        """
        self._working = working
        self._original = original
        for child in self._children:
            binder = getattr(child, "bind_fields", None)
            if callable(binder):
                binder(working=working, original=original)

    def __call__(
        self, subject: Subject, partition: Supervoxelization
    ) -> Supervoxelization:
        """
        Compute the composed supervoxel features for one subject.

        Args:
            subject: Owning subject.
            partition: The supervoxel partition to describe.

        Returns:
            The same regions with the merged feature columns.
        """
        parts = [child(subject, partition) for child in self._children]
        blocks = [part.feature_frame() for part in parts]
        merged = self._combiner(
            blocks, context=self._context(subject.subject_id)
        )
        merged = self._apply_alias(merged)
        return partition.with_feature_frame(
            merged,
            produced_by=f"combiner.{self._combiner.spec.name}",
            spec_fingerprint=self._spec.fingerprint(),
        )


class HabitatFeatureTree(_FeatureTreeBase):
    """
    Evaluate a habitat-level feature tree into one ``FeatureTable``.

    Children produce one-row-per-subject tables; the combiner merges their
    feature columns and the result is wrapped back with the first child's
    column roles.
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__(spec, build_habitat_extractor)

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the composed habitat features for one subject.

        Args:
            subject: Subject supplying original images when a child family
                needs intensity information.
            habitat_map: Habitat labels for that subject.

        Returns:
            A single-row-per-subject feature table with explicit column
            roles.
        """
        tables = [child(subject, habitat_map) for child in self._children]
        first = tables[0]
        # ``feature_matrix`` returns the feature columns indexed by the id
        # columns, which is exactly the row-aligned block a combiner needs.
        # Combiners align rows positionally and drop the index, so the id
        # index is restored afterwards for the rebuilt table.
        blocks = [table.feature_matrix() for table in tables]
        merged = self._combiner(
            blocks, context=self._context(subject.subject_id)
        )
        merged = self._apply_alias(merged)
        merged.index = blocks[0].index
        return FeatureTable(
            frame=merged.reset_index(),
            id_columns=first.id_columns,
            feature_columns=tuple(str(column) for column in merged.columns),
            outcome=first.outcome,
            provenance=first.provenance,
        )


def _build_extractor(
    spec: Any,
    *,
    leaf_registry: Any,
    tree_class: type,
    domain: str,
) -> Any:
    """
    Route one feature spec to a leaf extractor or a combiner tree.

    Args:
        spec: The node's spec (or its dict form).
        leaf_registry: Registry of the domain's leaf extractors.
        tree_class: Wrapper class for combiner nodes of this granularity.
        domain: Registry domain name used in error messages.

    Returns:
        The constructed component.

    Raises:
        HABITAPIError: If a combiner is used without children.
        ComponentNotFoundError: If the name is registered nowhere.
    """
    normalised = _normalise_spec(spec, owner=domain)
    if "children" in normalised.params:
        return tree_class(normalised)
    if leaf_registry.get(normalised.name) is not None:
        return leaf_registry.create(normalised.name, **dict(normalised.params))
    if CombinerRegistry.get(normalised.name) is not None:
        raise HABITAPIError(
            f"{normalised.name!r} is a combiner and requires a 'children' "
            "list of extractor specs; a combiner without children has "
            "nothing to merge."
        )
    raise ComponentNotFoundError(
        f"Unknown {domain} {normalised.name!r}. Available extractors: "
        f"{list(leaf_registry.available())}; combiners (need 'children'): "
        f"{list(CombinerRegistry.available())}"
    )


def build_voxel_extractor(spec: Any) -> Any:
    """
    Build one voxel-level node: a leaf extractor or a combiner tree.

    Args:
        spec: ``Spec`` (or dict form) of the voxel feature step.

    Returns:
        A :class:`~habit.domain.protocols.VoxelFeatureExtractor`.
    """
    return _build_extractor(
        spec,
        leaf_registry=VoxelFeatureExtractorRegistry,
        tree_class=VoxelFeatureTree,
        domain="voxel_feature_extractor",
    )


def build_supervoxel_extractor(spec: Any) -> Any:
    """
    Build one supervoxel-level node: a leaf extractor or a combiner tree.

    Args:
        spec: ``Spec`` (or dict form) of the supervoxel feature step.

    Returns:
        A :class:`~habit.domain.protocols.SupervoxelFeatureExtractor`.
    """
    return _build_extractor(
        spec,
        leaf_registry=SupervoxelFeatureExtractorRegistry,
        tree_class=SupervoxelFeatureTree,
        domain="supervoxel_feature_extractor",
    )


def build_habitat_extractor(spec: Any) -> Any:
    """
    Build one habitat-level node: a leaf extractor or a combiner tree.

    Args:
        spec: ``Spec`` (or dict form) of one habitat feature family.

    Returns:
        A :class:`~habit.domain.protocols.HabitatFeatureExtractor`.
    """
    return _build_extractor(
        spec,
        leaf_registry=HabitatFeatureExtractorRegistry,
        tree_class=HabitatFeatureTree,
        domain="habitat_feature_extractor",
    )
