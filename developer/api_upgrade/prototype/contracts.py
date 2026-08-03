"""
HABIT v1.0 design prototype -- L2 data contracts.

STATUS: design-stage prototype. Signatures and invariants are the deliverable;
bodies raise ``NotImplementedError`` on purpose. Nothing here is imported by the
shipped ``habit`` package, so this file cannot affect v0.1.x behaviour.

This module defines the vocabulary that every other HABIT layer speaks. The
guiding rule is that each type must be explainable in the language of habitat
imaging research -- an abstraction that can only be justified in software terms
does not belong here.

Layering rule enforced for this file:
    - It may import numpy/pandas/pydantic-level primitives only.
    - It must NOT know about YAML, ``out_dir``, ``run_mode``, CLI, or logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import pandas as pd

__all__ = [
    "Geometry",
    "ImageRef",
    "ImageVolume",
    "MaskVolume",
    "Subject",
    "Cohort",
    "VoxelFeatureField",
    "Supervoxelization",
    "HabitatMap",
    "HabitatModel",
    "CohortFingerprint",
    "FeatureTable",
    "Provenance",
    "RunManifest",
    "StudyResult",
]


# ---------------------------------------------------------------------------
# Provenance -- part of the data structure, not a separate reporting feature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """
    Immutable record answering "how was this object produced?".

    Provenance travels with every derived object instead of being assembled at
    the end of a workflow. This is what allows a third party who used only one
    HABIT component inside their own pipeline to still emit a complete methods
    description for a manuscript.

    Attributes:
        produced_by: Registered component name that created this object, e.g.
            ``"supervoxelizer.kmeans"``.
        spec_fingerprint: Stable hash of the algorithm specification used, so
            two runs can be compared for scientific equivalence.
        inputs: Provenance of every object consumed to produce this one. This
            forms a directed acyclic graph back to the raw images.
        software: Version fingerprint of HABIT and the scientifically relevant
            dependencies (e.g. PyRadiomics, SimpleITK, scikit-learn).
        random_seed: Seed in effect when the object was produced, or ``None``
            when the producing step is deterministic.
        created_at: ISO-8601 UTC timestamp.
        notes: Free-form annotations that must never be required for
            reproduction; they exist for human readers only.
    """

    produced_by: str
    spec_fingerprint: str
    inputs: Tuple["Provenance", ...] = ()
    software: Mapping[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None
    created_at: Optional[str] = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def derive(
        self,
        *,
        produced_by: str,
        spec_fingerprint: str,
        random_seed: Optional[int] = None,
    ) -> "Provenance":
        """
        Create the provenance of an object derived from this one.

        Operator authors never write provenance by hand; base classes call this
        so that the propagation rule stays uniform across the codebase.

        Args:
            produced_by: Registered name of the component doing the derivation.
            spec_fingerprint: Fingerprint of that component's specification.
            random_seed: Seed used by the derivation, when applicable.

        Returns:
            A new ``Provenance`` whose ``inputs`` contains ``self``.
        """
        raise NotImplementedError("design prototype")


# ---------------------------------------------------------------------------
# Geometry and image primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """
    Spatial definition shared by every volumetric object of one subject.

    Two volumetric objects may only be combined when their geometries are
    compatible. Making geometry an explicit, comparable value is what lets HABIT
    accept images produced by other tools (nnU-Net, MONAI, 3D Slicer) without a
    directory convention acting as the implicit contract.

    Attributes:
        shape: Voxel grid size as ``(z, y, x)``.
        spacing: Physical voxel size in mm as ``(z, y, x)``.
        origin: Physical coordinate of voxel ``(0, 0, 0)``.
        direction: Row-major 3x3 direction cosine matrix, flattened.
        frame_of_reference: Optional identifier tying several series to the same
            physical space, used to detect silently mismatched registrations.
    """

    shape: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]
    frame_of_reference: Optional[str] = None

    def is_compatible_with(self, other: "Geometry", *, tolerance: float = 1e-5) -> bool:
        """
        Report whether two geometries describe the same voxel grid.

        Args:
            other: Geometry to compare against.
            tolerance: Absolute tolerance for floating-point comparison of
                spacing, origin, and direction.

        Returns:
            ``True`` when the grids coincide within ``tolerance``.
        """
        raise NotImplementedError("design prototype")


@runtime_checkable
class ImageRef(Protocol):
    """
    Lazy handle to volumetric data.

    This protocol is the single most important reason HABIT can serve both the
    notebook user with 30 subjects and the batch user with 3000. Operators
    always receive an ``ImageRef`` and decide when to materialise it, so:

    - small cohorts can stay fully in memory and compose freely;
    - large cohorts pass lightweight handles across process boundaries;
    - third parties can back a subject with PACS, zarr, a torch tensor, or an
      in-memory array by implementing this protocol alone.
    """

    @property
    def geometry(self) -> Geometry:
        """Return grid definition without materialising voxel data."""

    def load(self) -> np.ndarray:
        """Materialise and return the voxel array."""

    # Note: ``ImageVolume`` / ``MaskVolume`` below are the already-materialised
    # counterparts and should satisfy this protocol structurally (``load()``
    # returning their own array), so there is ONE family of image types, not a
    # parallel eager/lazy pair.


@dataclass(frozen=True)
class ImageVolume:
    """
    Materialised intensity volume bound to a geometry.

    Attributes:
        array: Voxel intensities.
        geometry: Spatial definition of ``array``.
        modality: Modality or sequence label, e.g. ``"T1"``, ``"delay2"``.
    """

    array: np.ndarray
    geometry: Geometry
    modality: Optional[str] = None


@dataclass(frozen=True)
class MaskVolume:
    """
    Materialised label volume bound to a geometry.

    Attributes:
        array: Integer labels; ``0`` denotes background.
        geometry: Spatial definition of ``array``.
        roi_name: Name of the delineated region, e.g. ``"tumor"``.
    """

    array: np.ndarray
    geometry: Geometry
    roi_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Subject and cohort -- replacing the directory convention as the data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Subject:
    """
    One imaging subject.

    This replaces the v0.1 contract in which "a subject" only existed as a
    folder name discovered by scanning ``data_dir``. Making the subject an
    object is what decouples HABIT from its own directory layout and therefore
    what makes it embeddable.

    Attributes:
        subject_id: Identifier unique within a cohort.
        images: Modality name to lazy image handle.
        masks: ROI name to lazy mask handle.
        metadata: Clinical or acquisition attributes. Never required for
            computation; consumed by downstream modelling and reporting.
    """

    subject_id: str
    images: Mapping[str, ImageRef]
    masks: Mapping[str, ImageRef]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def image(self, modality: str) -> ImageVolume:
        """
        Materialise one modality.

        Args:
            modality: Key into ``images``.

        Returns:
            The loaded intensity volume.

        Raises:
            KeyError: If the modality is absent for this subject.
        """
        raise NotImplementedError("design prototype")

    def mask(self, roi_name: Optional[str] = None) -> MaskVolume:
        """
        Materialise one ROI mask.

        Args:
            roi_name: Key into ``masks``. When ``None`` and exactly one mask
                exists, that mask is returned.

        Returns:
            The loaded label volume.

        Raises:
            KeyError: If the ROI is absent.
            ValueError: If ``roi_name`` is ``None`` and the subject has more
                than one mask, since silently picking one would be unsafe.
        """
        raise NotImplementedError("design prototype")


class Cohort(Sequence[Subject]):
    """
    Ordered collection of subjects.

    Order is part of the contract, not an implementation detail: population-level
    clustering can be sensitive to subject order, so a reproducible cohort must
    have a defined, recorded ordering.

    Args:
        subjects: Subjects in canonical order.
        name: Human-readable cohort name used in reports, e.g. ``"training"``.
        metadata: Cohort-level attributes such as centre, scanner, or study.
    """

    def __init__(
        self,
        subjects: Sequence[Subject],
        *,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        raise NotImplementedError("design prototype")

    def __len__(self) -> int:
        raise NotImplementedError("design prototype")

    def __getitem__(self, index: int) -> Subject:  # type: ignore[override]
        raise NotImplementedError("design prototype")

    def __iter__(self) -> Iterator[Subject]:
        raise NotImplementedError("design prototype")

    @property
    def subject_ids(self) -> Tuple[str, ...]:
        """Return the subject identifiers in canonical cohort order."""
        raise NotImplementedError("design prototype")

    @classmethod
    def from_directory(
        cls,
        root: Path,
        *,
        modalities: Sequence[str],
        roi: str,
        name: Optional[str] = None,
    ) -> "Cohort":
        """
        Build a cohort from HABIT's conventional directory layout.

        A thin shortcut over ``DirectoryDataSource(...).load()``, provided
        because reading a folder is the overwhelmingly common first line of a
        notebook session and should not require learning the adapter layer.

        Args:
            root: Directory root holding one subdirectory per subject.
            modalities: Modality keys to load, in the order the analysis needs.
            roi: Mask key identifying the region of interest.
            name: Human-readable cohort name used in reports.

        Returns:
            A cohort with a defined, reproducible subject order.
        """
        raise NotImplementedError("design prototype")

    def filter(self, predicate: Any) -> "Cohort":
        """
        Return a new cohort containing subjects satisfying ``predicate``.

        Args:
            predicate: Callable receiving a :class:`Subject` and returning bool.

        Returns:
            A new cohort preserving the relative order of retained subjects.
        """
        raise NotImplementedError("design prototype")

    def map(
        self,
        op: Any,
        *,
        backend: Optional[Any] = None,
        checkpoint: Optional[Any] = None,
    ) -> Sequence[Any]:
        """
        Apply a subject-level operator to every subject, in cohort order.

        This is the middle rung of a deliberate three-step ladder:
        ``op(subject)`` for one subject, ``cohort.map(op)`` for all of them, and
        ``cohort.map(op, backend=...)`` only when parallelism, per-subject
        timeouts or resume are actually wanted. Because ``backend`` defaults to
        a serial one, a researcher can complete an entire study without ever
        learning that execution backends exist.

        Args:
            op: Any subject-level operator, i.e. any of the four subject-level
                domain protocols or a :class:`SubjectPipeline`.
            backend: Execution strategy. Serial when omitted.
            checkpoint: Store enabling resume. Disabled when omitted.

        Returns:
            Results in cohort order, not completion order, so that downstream
            cohort-level steps stay reproducible.

        Raises:
            ProcessingError: If any subject failed and the backend's failure
                policy is to propagate.
        """
        raise NotImplementedError("design prototype")

    def summarize(self) -> "CohortFingerprint":
        """
        Summarise the cohort for provenance and model cards.

        Named ``summarize`` rather than ``fingerprint`` because it returns a
        rich summary object, whereas :meth:`Spec.fingerprint` returns a hash
        string; two same-named methods should not return such different things.
        The returned type keeps the name ``CohortFingerprint`` because that is
        the term nnU-Net already uses for a dataset summary.

        Returns:
            A fingerprint safe to embed in a shared :class:`HabitatModel`, i.e.
            containing no identifiable patient information.
        """
        raise NotImplementedError("design prototype")


@dataclass(frozen=True)
class CohortFingerprint:
    """
    Non-identifiable description of the cohort behind a fitted model.

    Sharing a habitat model without describing the cohort that defined it would
    be scientifically meaningless, but sharing subject identifiers would be
    unsafe. This type is the deliberate middle ground.

    Attributes:
        n_subjects: Number of subjects used for fitting.
        modalities: Modality names consumed, in canonical order.
        subject_id_digest: Salted digest of the ordered subject id list, which
            proves two runs used the same cohort without revealing identifiers.
        name: Optional cohort label, e.g. ``"HCC-DCE-training"``.
        description: Free-text description intended for a model card.
    """

    n_subjects: int
    modalities: Tuple[str, ...]
    subject_id_digest: str
    name: Optional[str] = None
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# The habitat pipeline vocabulary: voxel -> supervoxel -> habitat -> feature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoxelFeatureField:
    """
    Per-voxel feature vectors inside one subject's ROI.

    This is where every habitat analysis begins. In v0.1 it existed only as an
    anonymous ``DataFrame`` passed between pipeline steps, which made it
    impossible for an external tool to supply its own voxel features (for
    example embeddings from a self-supervised model).

    Attributes:
        subject_id: Owning subject.
        feature_names: Column names in ``values`` order.
        values: Array of shape ``(n_voxels, n_features)``.
        voxel_index: Array of shape ``(n_voxels, 3)`` giving the ``(z, y, x)``
            grid position of each row, so the field can be rendered back into
            image space.
        geometry: Grid the indices refer to.
        provenance: How this field was produced.
    """

    subject_id: str
    feature_names: Tuple[str, ...]
    values: np.ndarray
    voxel_index: np.ndarray
    geometry: Geometry
    provenance: Provenance

    def to_frame(self) -> pd.DataFrame:
        """Return the field as a DataFrame for inspection and interoperability."""
        raise NotImplementedError("design prototype")


@dataclass(frozen=True)
class Supervoxelization:
    """
    Within-subject partition of the ROI into supervoxels, plus their features.

    Scientific role: supervoxels denoise voxel-level features and reduce the
    clustering unit from a single voxel to a coherent local region, which is the
    first step of the ``two_step`` strategy.

    Attributes:
        subject_id: Owning subject.
        label_array: Supervoxel id per voxel, shape equal to the ROI grid;
            ``0`` denotes voxels outside the ROI.
        features: Index is supervoxel id, columns are aggregated features. This
            is the payload that a federated deployment would transmit instead of
            the images themselves.
        geometry: Grid ``label_array`` refers to.
        provenance: How this partition was produced.
    """

    subject_id: str
    label_array: np.ndarray
    features: pd.DataFrame
    geometry: Geometry
    provenance: Provenance


@dataclass(frozen=True)
class HabitatMap:
    """
    Habitat label image for one subject.

    Attributes:
        subject_id: Owning subject.
        label_array: Habitat id per voxel; ``0`` denotes background.
        geometry: Grid ``label_array`` refers to.
        model_id: Identifier of the :class:`HabitatModel` that assigned these
            labels. Without it, habitat ids from different runs are not
            comparable, which is the most common silent error in habitat studies.
        habitat_ids: Habitat ids the model can assign, in canonical order. Note
            that a given subject need not contain all of them.
        provenance: How this map was produced.
    """

    subject_id: str
    label_array: np.ndarray
    geometry: Geometry
    model_id: str
    habitat_ids: Tuple[int, ...]
    provenance: Provenance


@dataclass(frozen=True)
class HabitatModel:
    """
    Population-level habitat definition -- HABIT's primary scientific artefact.

    In v0.1 this was serialised as an opaque ``habitat_pipeline.pkl`` byproduct.
    Promoting it to a first-class, self-describing object is what enables the
    strategic goal: a habitat definition published alongside a paper can be
    loaded by other groups and applied to their own cohorts, exactly the way a
    pretrained segmentation model circulates today.

    Attributes:
        model_id: Stable identifier derived from the specification fingerprint.
        n_habitats: Number of habitats this model can assign.
        feature_names: Features consumed for assignment, in required order.
        centroids: Population cluster centres, shape
            ``(n_habitats, n_features)``.
        preprocessing_state: State learned at fit time and required at apply
            time, e.g. binning edges and normalisation statistics. Keeping this
            inside the model is what guarantees train/predict consistency.
        spec_payload: Serialisable form of the full algorithm specification, so
            the model can describe itself and be exported back to YAML.
        cohort_fingerprint: Non-identifiable description of the defining cohort.
        provenance: Software, dependency, and seed fingerprint.
    """

    model_id: str
    n_habitats: int
    feature_names: Tuple[str, ...]
    centroids: np.ndarray
    preprocessing_state: Mapping[str, Any]
    spec_payload: Mapping[str, Any]
    cohort_fingerprint: CohortFingerprint
    provenance: Provenance

    def summary(self) -> str:
        """
        Return a human-readable model card.

        Named ``summary`` (statsmodels convention) rather than ``describe``,
        because in scientific Python ``DataFrame.describe()`` already returns a
        statistics table, and this returns prose. Intended for both notebook
        inspection and inclusion in a manuscript's supplementary material.
        """
        raise NotImplementedError("design prototype")

    def assigner(self, name: str = "nearest_centroid", **params: Any) -> Any:
        """
        Build an assigner that projects this model onto individual subjects.

        Assigners take their model at construction time, so this factory is the
        ordinary way to obtain one and keeps the common case to a single call:
        ``labels = model.assigner()(supervoxel_map)``.

        Args:
            name: Registered ``habitat_assigner`` implementation name.
            **params: Parameters for that implementation.

        Returns:
            A one-argument callable from a supervoxel map to a habitat map.
        """
        raise NotImplementedError("design prototype")

    def save(self, path: Path) -> Path:
        """
        Persist the model in a versioned, self-describing format.

        Deliberately not a bare pickle: a shared scientific artefact must remain
        readable across HABIT versions, or fail with an explicit incompatibility
        message rather than a deserialisation error.

        Args:
            path: Destination file path.

        Returns:
            The written path.
        """
        raise NotImplementedError("design prototype")

    @classmethod
    def load(cls, path: Path) -> "HabitatModel":
        """
        Load a model previously written by :meth:`save`.

        Args:
            path: Source file path.

        Returns:
            The reconstructed model.

        Raises:
            CompatibilityError: If the file was produced by an incompatible
                HABIT version, with guidance on which version can read it.
        """
        raise NotImplementedError("design prototype")


@dataclass(frozen=True)
class FeatureTable:
    """
    Feature table with explicit column semantics.

    v0.1 passed bare DataFrames whose column roles were conventions spread
    across the codebase. Making the roles explicit removes a whole class of
    leakage bugs, e.g. an identifier accidentally entering the model matrix.

    Attributes:
        frame: The underlying table.
        id_columns: Columns identifying the unit of analysis, e.g. ``subject``.
        feature_columns: Columns usable as model inputs.
        outcome_column: Clinical-outcome column when present. Named ``outcome``
            because that is the medical-research term for the predicted
            endpoint; the previous name ``label_column`` contradicted its own
            docstring.
        provenance: How this table was produced.
    """

    frame: pd.DataFrame
    id_columns: Tuple[str, ...]
    feature_columns: Tuple[str, ...]
    outcome_column: Optional[str] = None
    provenance: Optional[Provenance] = None

    def feature_matrix(self) -> pd.DataFrame:
        """
        Return only the model-input columns, indexed by the id columns.

        Named ``feature_matrix`` rather than ``features`` so it cannot be
        confused with running feature extraction, and because it returns a
        matrix-like frame rather than a list of features.
        """
        raise NotImplementedError("design prototype")

    def join(self, other: "FeatureTable") -> "FeatureTable":
        """
        Join another table on the shared id columns.

        Args:
            other: Table to merge; must share ``id_columns``.

        Returns:
            A new table whose provenance records both inputs.

        Raises:
            ValueError: If the id columns do not match.
        """
        raise NotImplementedError("design prototype")


# ---------------------------------------------------------------------------
# What a completed study hands back
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    """
    Everything needed to describe and audit one completed analysis.

    Assembled from the :class:`Provenance` records that travelled with the data,
    so it reports what actually ran rather than what was requested. That
    distinction is the whole point: a methods paragraph derived from a
    configuration file would describe intent, while this one describes fact,
    including subjects that failed and were excluded.

    Attributes:
        spec_payload: Serialised specification of the analysis that ran.
        provenance: Root provenance of the primary result.
        subject_outcomes: Per-subject success or failure, keyed by subject id.
        started_at: ISO-8601 start timestamp.
        finished_at: ISO-8601 completion timestamp.
    """

    spec_payload: Mapping[str, Any]
    provenance: Provenance
    subject_outcomes: Mapping[str, str] = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def software_versions(self) -> Mapping[str, str]:
        """Return HABIT and dependency versions captured at execution time."""
        raise NotImplementedError("design prototype")

    def random_seeds(self) -> Mapping[str, int]:
        """Return the seed used by each stochastic component."""
        raise NotImplementedError("design prototype")

    def describe_methods(self, style: str = "radiology") -> str:
        """
        Render the executed analysis as a manuscript methods paragraph.

        Deliberately the same verb and signature as
        :meth:`HabitatSpec.describe_methods`; the difference is completeness,
        not vocabulary. The spec describes what was intended and can be read
        before running, this describes what happened and includes versions,
        seeds and excluded subjects.

        Args:
            style: Target venue convention, e.g. ``"radiology"`` or ``"nature"``.

        Returns:
            English prose that states only steps that actually executed.
        """
        raise NotImplementedError("design prototype")

    def checklist(self, standard: str) -> pd.DataFrame:
        """
        Return an item-by-item compliance table for a reporting standard.

        Args:
            standard: One of ``"IBSI"``, ``"CLEAR"``, ``"METRICS"``,
                ``"TRIPOD+AI"``.

        Returns:
            One row per checklist item with the value HABIT can evidence and,
            where it cannot, an explicit statement that the item needs a human
            answer. Silently marking unverifiable items as satisfied would make
            the whole feature untrustworthy.
        """
        raise NotImplementedError("design prototype")

    def to_json(self, path: Optional[Path] = None) -> str:
        """
        Serialise the manifest, optionally writing it to disk.

        Args:
            path: Destination file. When ``None`` the JSON text is only
                returned.

        Returns:
            The JSON text.
        """
        raise NotImplementedError("design prototype")


@dataclass(frozen=True)
class StudyResult:
    """
    What a fitted study hands back, entirely in memory.

    Nothing here has touched the filesystem. Writing is a separate, explicit
    act via :meth:`save`, which is what allows the identical code to run inside
    someone else's service where there is no output directory at all.

    Attributes:
        habitat_model: The population-level habitat definition. Named in full
            rather than ``model`` because ``model`` already means a trained
            classifier elsewhere in HABIT.
        pipeline: The subject-level procedure that applies that definition, so
            that model and procedure can be shipped together for external
            validation.
        features: Habitat-level features for the fitted cohort.
        habitat_maps: Per-subject habitat label images, in cohort order.
        manifest: Provenance and reporting for this run.
    """

    habitat_model: "HabitatModel"
    pipeline: Any
    features: FeatureTable
    habitat_maps: Tuple["HabitatMap", ...]
    manifest: RunManifest

    def save(self, out_dir: Path) -> Path:
        """
        Write every artefact of this study to a directory.

        Args:
            out_dir: Destination directory, created when missing.

        Returns:
            The directory written to.
        """
        raise NotImplementedError("design prototype")
