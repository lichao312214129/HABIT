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
"""The domain protocols (L3).

The core habitat pipeline rests on exactly EIGHT domain protocols, and each
one is a term that already exists in habitat imaging research (voxel feature,
feature preprocessing, supervoxel, supervoxel feature, habitat model, habitat
map, habitat feature). A generic ``Operator.transform(x)`` would be more
flexible and strictly less useful: a radiologist reading
``HabitatModelFitter.fit(units)`` understands it immediately, whereas a
generic name teaches them nothing and gives an extension author no hint about
which slot to implement. A ninth protocol, ``ImagePerturbation``, serves the
precision-analysis domain (simulated test-retest) and follows the same
one-call-per-subject convention.

``SupervoxelFeatureExtractor`` was added after the initial five: growing a
partition and describing its regions are two independent scientific choices
(v0.1 configured them in two separate YAML blocks, ``habitat_segmentation.
supervoxel`` and ``feature_construction.supervoxel_level``), and fusing them
into ``Supervoxelizer`` made per-supervoxel radiomics inexpressible -- that
family needs the ORIGINAL IMAGES, which a ``VoxelFeatureField`` does not
carry. A tenth protocol, ``Combiner``, factors multi-modality composition
out of the extractors: extractor leaves each describe ONE modality's signal,
combiner nodes merge the sibling blocks, and the composition itself becomes
an extensible plugin domain rather than a fixed set of extractor flags.

The last two, ``SubjectFeaturePreprocessor`` and
``CohortFeaturePreprocessor``, split feature preprocessing along the axis that
actually matters: whether the fitted statistics cross subject boundaries.
Individual-level preprocessing is stateless BY NATURE -- its statistics come
from the single matrix in front of it, so training and prediction are the same
computation -- while cohort-level preprocessing is stateful by nature and is
the one leakage-sensitive step in habitat definition. Neither protocol
mentions granularity, which is the point: the stateless one applies equally to
voxel features and to supervoxel features. v0.1's
``preprocessing_for_subject_level`` / ``preprocessing_for_group_level``
conflated state ownership with granularity, and consequently offered no
stateless normalisation for per-supervoxel features at all.

Call convention: single ``__call__`` per subject-level protocol, with NO
class-body verb aliases (``extract = __call__`` etc. would bind the function
object at class-definition time, so a subclass overriding ``__call__`` would
leave the alias pointing at the parent's implementation -- a silent
divergence). Readability at the call site comes from the variable name
(``voxel_features(subject)``), not from a second method name.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from habit.contracts.habitat import (
    HabitatMap,
    HabitatModel,
    Supervoxelization,
    VoxelFeatureField,
)
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable
from habit.spec.specs import Spec

__all__ = [
    "VoxelFeatureExtractor",
    "SubjectFeaturePreprocessor",
    "CohortFeaturePreprocessor",
    "Supervoxelizer",
    "SupervoxelFeatureExtractor",
    "HabitatModelFitter",
    "HabitatAssigner",
    "HabitatFeatureExtractor",
    "ImagePerturbation",
    "Preprocessor",
    "Combiner",
    "PoolingMarker",
    "Seedable",
]


@runtime_checkable
class VoxelFeatureExtractor(Protocol):
    """
    Turn a subject's images into per-voxel feature vectors inside the ROI.

    Implementations in HABIT cover raw intensity, kinetic curves, local
    entropy, and voxel-wise radiomics. Because this is a protocol rather than
    a hard-coded branch, an external group can plug in voxel embeddings from
    a self-supervised or foundation model without modifying HABIT.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification, used for provenance and caching."""

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute per-voxel features for one subject.

        Args:
            subject: Subject providing images and the ROI mask.

        Returns:
            Feature vectors for every voxel inside the ROI.

        Raises:
            GeometryError: If the required modalities and the mask do not
                share a compatible voxel grid.
        """


@runtime_checkable
class SubjectFeaturePreprocessor(Protocol):
    """
    Preprocess one subject's feature matrix using only that subject's data.

    Stateless by construction: every statistic comes from the matrix passed
    in, so there is nothing to fit, nothing to persist, and no way to leak
    training information into a validation subject. This is what makes the
    protocol a plain callable rather than a fit/transform pair.

    Deliberately says nothing about granularity. The matrix rows may be
    voxels or supervoxels, and the SAME implementation serves both -- which
    is the whole reason this is separate from
    :class:`CohortFeaturePreprocessor` rather than being one protocol per
    pipeline position. Scientifically the purpose is to remove BETWEEN-subject
    variation (scanner, sequence, intensity scale), and that only works when
    each subject is normalised by its own distribution.

    A plain ``DataFrame`` is the input and output type because the
    computation is genuinely type-agnostic; the typed contracts bridge to it
    through their ``feature_frame()`` / ``with_feature_frame()`` pair.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def __call__(self, block: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess one unit-by-feature matrix.

        Args:
            block: Rows are clustering units (voxels or supervoxels) of ONE
                subject, columns are features.

        Returns:
            The preprocessed matrix with the same rows in the same order.
            Columns may be a subset when the chain filters features.
        """


@runtime_checkable
class CohortFeaturePreprocessor(Protocol):
    """
    Preprocess the pooled cohort feature matrix with fitted statistics.

    Stateful by construction, and the mirror image of
    :class:`SubjectFeaturePreprocessor`: its purpose is to place units from
    DIFFERENT subjects in one comparable feature space, so the clusters that
    define habitats mean the same thing for everyone. That requires shared
    statistics, which requires fitting, which makes this the single
    leakage-sensitive step of habitat definition -- ``fit`` must see training
    data only.

    The fitted state therefore belongs in the published
    :class:`~habit.contracts.habitat.HabitatModel`: a habitat definition
    applied to a new cohort without its cohort-level preprocessing would put
    that cohort in a different feature space while appearing to work.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def fit(self, block: pd.DataFrame) -> "CohortFeaturePreprocessor":
        """
        Learn the transformation from the pooled TRAINING matrix.

        Args:
            block: Rows are clustering units from every training subject.

        Returns:
            ``self``, fitted.
        """

    def transform(self, block: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted transformation to a pooled matrix.

        Args:
            block: Matrix carrying the feature columns seen at fit time.

        Returns:
            The preprocessed matrix, rows preserved.
        """


@runtime_checkable
class Supervoxelizer(Protocol):
    """
    Partition one subject's ROI into supervoxels.

    Scientific motivation: voxel-level features are noisy, and clustering
    voxels directly across subjects tends to recover scanner scale
    differences rather than intratumoral heterogeneity. Aggregating within
    spatially coherent supervoxels, after per-subject normalisation, is what
    makes the subsequent population-level clustering biologically meaningful.

    Implementations describe each region with the mean of the voxel features
    they partitioned, which is free once the partition exists and is what
    v0.1 always computed. That default is a summary, not a commitment: pass a
    :class:`SupervoxelFeatureExtractor` to describe the same regions
    differently (per-supervoxel radiomics, for instance).
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def __call__(self, field: VoxelFeatureField) -> Supervoxelization:
        """
        Group voxels into supervoxels and summarise them by feature mean.

        Args:
            field: Per-voxel features for one subject.

        Returns:
            The supervoxel partition together with per-supervoxel means.
        """


@runtime_checkable
class SupervoxelFeatureExtractor(Protocol):
    """
    Describe each supervoxel of one subject, replacing the default summary.

    Structurally the twin of :class:`HabitatFeatureExtractor`: both answer
    "given a partition of this subject plus the original images, what
    describes each region?". They differ only in granularity and in who
    consumes the answer -- supervoxel features feed the cohort-level
    clustering that DEFINES habitats, habitat features feed outcome
    modelling.

    Separating this from :class:`Supervoxelizer` is what keeps two
    independent scientific choices independent. It is also what makes
    per-supervoxel radiomics expressible at all: those families need the
    subject's original intensities, which a
    :class:`~habit.contracts.habitat.VoxelFeatureField` deliberately does
    not carry.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def __call__(
        self,
        subject: Subject,
        partition: Supervoxelization,
    ) -> Supervoxelization:
        """
        Recompute one subject's per-supervoxel features.

        Args:
            subject: Subject supplying original images when the family needs
                intensity information.
            partition: The subject's supervoxel partition, whose
                ``label_array`` defines the regions to describe.

        Returns:
            The SAME partition carrying the newly computed features. Returning
            a ``Supervoxelization`` rather than a bare table keeps the
            downstream contract single-typed: the fitter and the assigner
            consume partitions, and never need to know which extractor
            described them.
        """


@runtime_checkable
class HabitatModelFitter(Protocol):
    """
    Fit the population-level habitat definition. THIS IS THE COHORT-LEVEL STEP.

    This is the only place in HABIT where information crosses subject
    boundaries, and it is what makes habitat labels comparable between
    patients and between cohorts. Its output, :class:`HabitatModel`, is the
    artefact a study should publish so that other groups can reproduce the
    same habitat definition on their own data.

    Named ``*Fitter`` (lifelines / statsmodels convention), NOT
    ``*Estimator``: sklearn reserves "estimator" for objects whose ``fit``
    returns ``self`` and that are ``clone()``-able, whereas this ``fit``
    returns a NEW :class:`HabitatModel` artefact.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def fit(
        self,
        units: Sequence[Supervoxelization],
        *,
        cohort: Optional[Cohort] = None,
    ) -> HabitatModel:
        """
        Learn the shared habitat definition from all subjects.

        Args:
            units: Supervoxelizations in a defined, reproducible order.
                Order is part of the contract because clustering can be
                order-sensitive.
            cohort: Cohort the units came from, used only to record a
                non-identifiable fingerprint inside the model.

        Returns:
            A self-contained habitat model applicable to unseen subjects.
        """


@runtime_checkable
class HabitatAssigner(Protocol):
    """
    Assign habitat labels to one subject using a fitted model.

    Keeping this separate from the fitter is what enforces train/predict
    consistency structurally rather than by convention: prediction has no
    way to re-learn anything, because everything it needs is inside the
    model.

    The model is supplied to the CONSTRUCTOR, not to the call. Two
    consequences follow: the assigner becomes an ordinary one-argument
    callable (a subject-level operator like every other step), and an
    assigner cannot be constructed without a fitted model, so "predicting
    before fitting" becomes unrepresentable.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    @property
    def model(self) -> HabitatModel:
        """Return the fitted habitat definition this assigner projects."""

    def __call__(self, supervoxel_map: Supervoxelization) -> HabitatMap:
        """
        Project the fitted habitat definition onto one subject.

        Args:
            supervoxel_map: Supervoxelization of the subject to label.

        Returns:
            The subject's habitat label image, tagged with the model's id.

        Raises:
            CompatibilityError: If the supervoxel features do not provide
                the feature names the model requires.
        """


@runtime_checkable
class HabitatFeatureExtractor(Protocol):
    """
    Compute habitat-level features for one subject.

    HABIT's differentiating feature families live here: MSI (multiregional
    spatial interaction), ITH score (topological fragmentation), volume and
    connectivity descriptors, and per-habitat or whole-habitat radiomics.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute one family of habitat-level features for one subject.

        This is the only binary subject-level operator, because habitat
        features genuinely need both the labels and the original intensities.
        Both arguments are subject-scoped, so it is still subject-level.

        Args:
            subject: Subject supplying original images when the family needs
                intensity information.
            habitat_map: Habitat labels for that subject.

        Returns:
            A single-row-per-subject feature table with explicit column
            roles.
        """


@runtime_checkable
class Combiner(Protocol):
    """
    Merge the feature blocks produced by sibling nodes of a feature tree.

    A combiner is the internal node of a feature composition tree: it never
    touches images, subjects, or the filesystem -- only the column blocks its
    child nodes already produced. This is what keeps the multi-modality
    matrix open-ended: extractors answer "how do I describe ONE modality's
    signal", combiners answer "how do I merge SIBLING descriptions", and new
    combination strategies (weighting, ratios, kinetic slopes, formulas) plug
    in through the ``habit.combiner`` entry point without changing any
    extractor.

    The same protocol serves every granularity: at voxel level the blocks are
    the children ``VoxelFeatureField.feature_frame()`` matrices (rows are ROI
    voxels in C order); at supervoxel level they are the children
    ``Supervoxelization.feature_frame()`` matrices (rows are supervoxels); at
    habitat level they are one-row-per-subject frames. Rows are always
    aligned positionally across siblings, because the tree wrapper guarantees
    every child describes the same units.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification, used for provenance."""

    def __call__(
        self,
        blocks: Sequence[pd.DataFrame],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Combine sibling feature blocks into one block.

        Args:
            blocks: Child blocks in child order, one per sibling node. All
                blocks share the same row count (enforced by the tree
                wrapper).
            context: Optional evaluation context supplied by the tree
                wrapper. Recognised keys:

                - ``"sources"``: source label of each child, in child order
                  (a leaf's ``as_`` alias when set, else its ``modality``,
                  else the node name). Combiners whose parameters are keyed
                  by child (e.g. ``weights``) resolve them against these
                  labels.
                - ``"subject_id"``: id of the subject being processed, for
                  combiners whose parameters are subject-keyed (``kinetic``
                  acquisition times).

        Returns:
            The merged block, with the same row count as the inputs.
        """


@runtime_checkable
class ImagePerturbation(Protocol):
    """
    Turn one subject into a perturbed copy of itself: a simulated re-acquisition.

    Scientific role: voxel-wise features are only worth clustering if they
    survive the small acquisition variations a scanner inevitably introduces
    (noise, sub-voxel patient shifts, slight angulation). An image
    perturbation replays those variations in silico so feature repeatability
    can be measured BEFORE any habitat is computed (Prior et al., Radiol
    Artif Intell 2024;6(2):e230118).

    The contract is deliberately the narrowest possible: one subject in, one
    perturbed subject out, same grid, same keys. Chaining several
    perturbations composes a full simulated retest, and implementing this
    protocol is all a new perturbation family (bias fields, motion ghosts,
    resampling artefacts) has to do.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a perturbed copy of ``subject``.

        Args:
            subject: Subject providing images (and masks, for geometric
                perturbations).
            rng: Random generator for the stochastic steps; supplied by the
                caller so one seed drives an entire perturbation chain.
                Deterministic perturbations accept and ignore it.

        Returns:
            A new subject on the SAME voxel grid with perturbed images (and
            perturbed masks for geometric perturbations); the original is
            left untouched.
        """


@runtime_checkable
class Preprocessor(Protocol):
    """
    Image-volume preprocessor: one subject in, one processed subject out.

    Scientific role: the steps that prepare raw MR/CT volumes for habitat
    analysis (resample, reorient, N4, z-score, histogram, CLAHE,
    registration). Each implementation is one named step in the
    ``preprocessor`` plugin domain. The recipe / atomic API chains them;
    a third party can call ``op(subject)`` on a single case.

    ``images`` and ``mask_roi`` are call-site assembly, not scientific
    parameters: omit them to process every modality on the subject.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""

    def __call__(
        self,
        subject: Subject,
        *,
        images: Optional[Sequence[str]] = None,
        mask_roi: Optional[str] = None,
    ) -> Subject:
        """
        Return a processed copy of ``subject``.

        Args:
            subject: One imaging subject.
            images: Optional modality keys to process. ``None`` means every
                image on the subject.
            mask_roi: Optional ROI key for mask-aware steps (z-score in
                mask, N4 mask, histogram landmarks). Unused steps ignore it.

        Returns:
            A new subject; the input is not mutated.
        """


@runtime_checkable
class PoolingMarker(Protocol):
    """
    Marker for the subject→cohort fan-in watershed in a stage list.

    The built-in ``pool`` component performs no numeric work; the stage
    executor recognises the marker and calls
    :func:`~habit.domain.pooling.fan_in`. Third-party packages may register
    alternate markers in the ``pooling`` domain (entry point group
    ``habit.pooling``) when they need a discoverable watershed slot.
    """

    @property
    def spec(self) -> Spec:
        """Return the marker specification (provenance / fingerprint)."""

    def __call__(self) -> Any:
        """Return a small descriptor confirming the watershed (optional)."""


@runtime_checkable
class Seedable(Protocol):
    """
    Explicit control of the random state of a stochastic component.

    Habitat analysis is unusually seed-sensitive: k-means initialisation,
    GMM initialisation and SLIC seeding all shift the resulting habitat
    definition, and cohort-level clustering is additionally order-sensitive.
    MONAI reached the same conclusion with ``Randomizable``; HABIT uses the
    adjective name ``Seedable`` with a deliberately simpler signature.

    Components that are deterministic simply do not implement this protocol,
    which is itself useful information for the provenance record.
    """

    def set_random_state(self, seed: int) -> None:
        """
        Set the component's random state.

        Args:
            seed: Seed applied to every stochastic decision the component
                makes.
        """
