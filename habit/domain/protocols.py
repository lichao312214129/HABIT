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
"""The five domain protocols (L3).

There are exactly FIVE domain protocols, and each one is a term that already
exists in habitat imaging research (voxel feature, supervoxel, habitat model,
habitat map, habitat feature). A generic ``Operator.transform(x)`` would be
more flexible and strictly less useful: a radiologist reading
``HabitatModelFitter.fit(units)`` understands it immediately, whereas a
generic name teaches them nothing and gives an extension author no hint about
which slot to implement.

Call convention: single ``__call__`` per subject-level protocol, with NO
class-body verb aliases (``extract = __call__`` etc. would bind the function
object at class-definition time, so a subclass overriding ``__call__`` would
leave the alias pointing at the parent's implementation -- a silent
divergence). Readability at the call site comes from the variable name
(``voxel_features(subject)``), not from a second method name.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

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
    "Supervoxelizer",
    "HabitatModelFitter",
    "HabitatAssigner",
    "HabitatFeatureExtractor",
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
class Supervoxelizer(Protocol):
    """
    Partition one subject's ROI into supervoxels and summarise each of them.

    Scientific motivation: voxel-level features are noisy, and clustering
    voxels directly across subjects tends to recover scanner scale
    differences rather than intratumoral heterogeneity. Aggregating within
    spatially coherent supervoxels, after per-subject normalisation, is what
    makes the subsequent population-level clustering biologically meaningful.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def __call__(self, field: VoxelFeatureField) -> Supervoxelization:
        """
        Group voxels into supervoxels and aggregate their features.

        Args:
            field: Per-voxel features for one subject.

        Returns:
            The supervoxel partition together with per-supervoxel features.
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
