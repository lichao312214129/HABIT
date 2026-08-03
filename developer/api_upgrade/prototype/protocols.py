"""
HABIT v1.0 design prototype -- L3 domain protocols and L2 execution contracts.

STATUS: design-stage prototype. Signatures and contracts are the deliverable.

Three ideas drive this module.

1. There are exactly FIVE domain protocols, and each one is a term that already
   exists in habitat imaging research (voxel feature, supervoxel, habitat model,
   habitat map, habitat feature). A generic ``Operator.transform(x)`` would be
   more flexible and strictly less useful: a radiologist reading
   ``HabitatModelFitter.fit(units)`` understands it immediately, whereas a
   generic name teaches them nothing and gives an extension author no hint about
   which slot to implement.

2. Every computation is either SUBJECT-LEVEL or COHORT-LEVEL. That single
   distinction simultaneously defines the parallelism boundary, the checkpoint
   boundary, the train/predict boundary, and -- in future -- the federation
   boundary, where subject-level work runs inside the hospital and only
   supervoxel features leave it.

3. ONE SUBJECT IS THE ATOMIC CALL. Every subject-level operator is a plain
   callable that takes one subject's payload and returns one subject's result,
   so ``field = voxel_features(subject)`` works with no cohort, no backend, and
   no configuration. Cohorts, execution backends and checkpoints are optional
   machinery layered on top, never a precondition for doing one piece of work.

   This is the convention MONAI, TorchIO and PyRadiomics converged on, and it is
   a hard requirement for the ecosystem goal rather than a matter of taste: only
   a single-sample callable can be dropped into ``monai.transforms.Compose``,
   driven by a torch ``DataLoader``, or debugged on the one patient whose result
   looks wrong. Each protocol additionally exposes a domain-named alias
   (``extract``, ``build``, ``map``) so that reading code still reads like
   habitat imaging rather than like generic dataflow; the alias and ``__call__``
   are the same function, never two code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from .contracts import (
    Cohort,
    FeatureTable,
    HabitatMap,
    HabitatModel,
    Subject,
    Supervoxelization,
    VoxelFeatureField,
)
from .spec import Spec

__all__ = [
    "VoxelFeatureExtractor",
    "Supervoxelizer",
    "HabitatModelFitter",
    "HabitatAssigner",
    "HabitatFeatureExtractor",
    "SubjectPipeline",
    "Seedable",
    "SubjectOperator",
    "CohortOperator",
    "SubjectResult",
    "ExecutionBackend",
    "CheckpointStore",
    "DataSource",
    "ResultWriter",
    "ComponentRegistry",
]

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


# ---------------------------------------------------------------------------
# The five domain protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class VoxelFeatureExtractor(Protocol):
    """
    Turn a subject's images into per-voxel feature vectors inside the ROI.

    Implementations in HABIT cover raw intensity, kinetic curves, local entropy,
    and voxel-wise radiomics. Because this is a protocol rather than a hard-coded
    branch, an external group can plug in voxel embeddings from a self-supervised
    or foundation model without modifying HABIT -- the single most likely
    extension direction for habitat analysis over the next few years.
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
            GeometryError: If the required modalities and the mask do not share
                a compatible voxel grid.
        """

    # NOTE: there is deliberately NO ``extract = __call__`` alias. A class-body
    # alias binds the function object defined at that moment, so a subclass that
    # overrides ``__call__`` would leave ``extract`` pointing at the parent's
    # implementation -- a silent divergence. It would also force every
    # ``runtime_checkable`` implementer to expose two names. MONAI, TorchIO,
    # PyRadiomics and sklearn all use a single public call name; readability at
    # the call site comes from the variable name (``voxel_features(subject)``),
    # not from a second method.


@runtime_checkable
class Supervoxelizer(Protocol):
    """
    Partition one subject's ROI into supervoxels and summarise each of them.

    Scientific motivation: voxel-level features are noisy, and clustering voxels
    directly across subjects tends to recover scanner scale differences rather
    than intratumoral heterogeneity. Aggregating within spatially coherent
    supervoxels, after per-subject normalisation, is what makes the subsequent
    population-level clustering biologically meaningful.
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

    # No ``build = __call__`` alias, for the reasons given on
    # ``VoxelFeatureExtractor.__call__``.


@runtime_checkable
class HabitatModelFitter(Protocol):
    """
    Fit the population-level habitat definition. THIS IS THE COHORT-LEVEL STEP.

    This is the only place in HABIT where information crosses subject
    boundaries, and it is what makes habitat labels comparable between patients
    and between cohorts. Its output, :class:`HabitatModel`, is the artefact a
    study should publish so that other groups can reproduce the same habitat
    definition on their own data.

    Named ``*Fitter`` (lifelines / statsmodels convention), NOT ``*Estimator``:
    sklearn reserves "estimator" for objects whose ``fit`` returns ``self`` and
    that are ``clone()``-able, whereas this ``fit`` returns a NEW
    :class:`HabitatModel` artefact. Because HABIT components also expose
    ``get_params``/``set_params``, calling this an estimator would make
    ``sklearn.base.clone`` and ``Pipeline`` treat it as one and then break.
    The ``*Estimator`` name is reserved for the genuine sklearn adapters in
    ``habit.compat.sklearn``.
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
            units: Supervoxelizations in a defined, reproducible order. Order is
                part of the contract because clustering can be order-sensitive.
            cohort: Cohort the units came from, used only to record a
                non-identifiable fingerprint inside the model.

        Returns:
            A self-contained habitat model applicable to unseen subjects.
        """


@runtime_checkable
class HabitatAssigner(Protocol):
    """
    Assign habitat labels to one subject using a fitted model.

    Named ``HabitatAssigner`` rather than ``HabitatMapper`` because ``map`` is
    already taken in this codebase by the functional ``Cohort.map(op)`` and
    ``ExecutionBackend.map(op, items)`` (the ``Pool.map`` sense); a third,
    different meaning would collide. ``assign`` also avoids the ORM-mapper
    reading.

    Keeping this separate from the fitter is what enforces train/predict
    consistency structurally rather than by convention: prediction has no way
    to re-learn anything, because everything it needs is inside the model.

    The model is supplied to the CONSTRUCTOR, not to the call. Two consequences
    follow, and both are the reason for the choice:

    - the assigner becomes an ordinary one-argument callable, so it is a
      ``SubjectOperator`` like every other subject-level step and needs no
      special-case binding when composed or executed;
    - an assigner cannot be constructed without a fitted model, so "predicting
      before fitting" stops being a runtime error and becomes unrepresentable.

    ``HabitatModel.assigner()`` is the convenience factory for the common case.
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
            CompatibilityError: If the supervoxel features do not provide the
                feature names the model requires.
        """

    # No ``assign = __call__`` alias; see ``VoxelFeatureExtractor.__call__``.


@runtime_checkable
class HabitatFeatureExtractor(Protocol):
    """
    Compute habitat-level features for one subject.

    HABIT's differentiating feature families live here: MSI (multiregional
    spatial interaction), ITH score (topological fragmentation), non-radiomics
    volume and connectivity descriptors, and per-habitat or whole-habitat
    radiomics. Their mathematical definitions are unchanged by this refactor;
    only the way they are invoked changes.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute one family of habitat-level features for one subject.

        This is the only binary subject-level operator, because habitat features
        genuinely need both the labels and the original intensities. Both
        arguments are subject-scoped, so it is still subject-level: viewed as a
        :class:`SubjectOperator` its item is the ``(subject, habitat_map)``
        pair, which :class:`SubjectPipeline` forms by zipping on ``subject_id``.

        Args:
            subject: Subject supplying original images when the family needs
                intensity information.
            habitat_map: Habitat labels for that subject.

        Returns:
            A single-row-per-subject feature table with explicit column roles.
        """

    # No ``name`` property: the registered name already lives at ``spec.name``,
    # and a second attribute would only be another place to disagree.
    # No ``extract = __call__`` alias; see ``VoxelFeatureExtractor.__call__``.


# ---------------------------------------------------------------------------
# Composition and reproducibility support
# ---------------------------------------------------------------------------


class SubjectPipeline:
    """
    The subject-level chain composed into a single callable.

    This is the one CONCRETE class in this module; everything else here is a
    Protocol. It is concrete because there is exactly one sensible way to run
    the chain, and because callers must be able to construct it directly.

    This is HABIT's answer to ``monai.transforms.Compose``. A generic ``Compose``
    cannot be reused directly because HABIT's steps are heterogeneously typed --
    ``Subject -> VoxelFeatureField -> Supervoxelization -> HabitatMap`` -- and
    erasing those types to a single dict would discard exactly the contracts
    that make the design checkable. A named pipeline keeps the types and still
    yields one callable.

    Its practical value is that a fitted :class:`HabitatModel` plus a
    ``SubjectPipeline`` is precisely the pair a study needs to publish for
    external validation: the definition, and the procedure that applies it.

    Args:
        voxel_feature_extractor: Step producing per-voxel features.
        supervoxelizer: Step producing supervoxels. ``None`` clusters voxels
            directly, which is what the one-step and direct-pooling designs do.
        habitat_assigner: Step assigning habitat labels, already bound to a
            model.
    """

    def __init__(
        self,
        voxel_feature_extractor: VoxelFeatureExtractor,
        supervoxelizer: Optional[Supervoxelizer],
        habitat_assigner: HabitatAssigner,
    ) -> None:
        raise NotImplementedError("design prototype")

    @property
    def spec(self) -> Spec:
        """Return the composed specification of every stage."""
        raise NotImplementedError("design prototype")

    def __call__(self, subject: Subject) -> HabitatMap:
        """
        Run voxel features, supervoxelisation and mapping for one subject.

        Args:
            subject: The subject to label.

        Returns:
            The subject's habitat label image.
        """
        raise NotImplementedError("design prototype")

    def extract_features(
        self,
        subject: Subject,
        extractors: Sequence[HabitatFeatureExtractor],
    ) -> FeatureTable:
        """
        Run the pipeline and then the requested habitat feature families.

        Named ``extract_features`` rather than ``features`` so that it reads as
        an action; a bare noun on a callable object suggests an attribute.

        Args:
            subject: The subject to process.
            extractors: Habitat feature families to compute.

        Returns:
            One feature table for that subject, joined across families.
        """
        raise NotImplementedError("design prototype")


@runtime_checkable
class Seedable(Protocol):
    """
    Explicit control of the random state of a stochastic component.

    Habitat analysis is unusually seed-sensitive: k-means initialisation, GMM
    initialisation and SLIC seeding all shift the resulting habitat definition,
    and cohort-level clustering is additionally order-sensitive. Leaving each
    component to invent its own ``random_state`` parameter -- the v0.1 situation
    -- makes a run impossible to reseed as a whole and impossible to report
    honestly. MONAI reached the same conclusion with ``Randomizable``.

    Named ``Seedable`` (adjective, like ``Iterable`` / ``Hashable`` /
    MONAI's ``Randomizable``) rather than the noun ``SeedControl``. It is not
    called ``Randomizable`` because MONAI's signature is
    ``set_random_state(seed=None, state=None) -> self`` plus a ``randomize()``
    method, and reusing the name with a different contract would repeat the
    estimator mistake.

    Components that are deterministic simply do not implement this protocol,
    which is itself useful information for the provenance record.
    """

    def set_random_state(self, seed: int) -> None:
        """
        Set the component's random state.

        Args:
            seed: Seed applied to every stochastic decision the component makes.
        """


# ---------------------------------------------------------------------------
# The two-level execution contract
# ---------------------------------------------------------------------------


@runtime_checkable
class SubjectOperator(Protocol, Generic[TIn, TOut]):
    """
    A computation that touches exactly one subject.

    Declaring this is a contract, not a hint: it tells the execution backend
    that the work may be parallelised, checkpointed, retried, isolated on
    failure, and -- in a federated deployment -- executed inside the hospital
    that owns the images.

    Named ``SubjectOperator`` rather than the abbreviated ``SubjectLevelOp``:
    the design prose already says "算子/operator", and ``Op`` is both an
    abbreviation and, in a medical library, confusable with "operation".

    Note what this protocol does NOT introduce: a second method name. It is
    ``__call__`` plus two pieces of metadata, so every one of the four
    subject-level domain protocols satisfies it automatically and no plugin
    author ever writes an adapter. An earlier draft gave this protocol its own
    ``apply()`` verb, which would have forced every implementation to expose the
    same computation twice under two names.

    Implementations must be free of shared mutable state so they can be sent to
    a worker process.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification, used as part of the cache key."""

    def __call__(self, item: TIn) -> TOut:
        """
        Process one subject's payload.

        Args:
            item: The subject-scoped input.

        Returns:
            The subject-scoped output.
        """

    def cache_key(self, item: TIn) -> str:
        """
        Return a stable key identifying this computation for checkpointing.

        Args:
            item: The subject-scoped input.

        Returns:
            A key combining the subject identity and the spec fingerprint, so
            that changing an algorithm parameter correctly invalidates a resumed
            run instead of silently reusing stale results.
        """


@runtime_checkable
class CohortOperator(Protocol, Generic[TIn, TOut]):
    """
    A computation that must observe the whole cohort at once.

    Cohort-level operations cannot be parallelised across subjects and cannot be
    resumed per subject. Habitat model fitting and population-level feature
    preprocessing are the two instances in HABIT.
    """

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""

    def fit(self, items: Sequence[TIn], **context: Any) -> TOut:
        """
        Aggregate across subjects to produce a shared artefact.

        Args:
            items: Subject-level payloads in a defined order.
            **context: Optional keyword context an implementation may accept,
                e.g. :class:`HabitatModelFitter` takes ``cohort=`` to record
                a non-identifiable fingerprint. Declared here so that the more
                specific domain signatures remain compatible with this one.

        Returns:
            The cohort-level artefact, e.g. a :class:`HabitatModel`.
        """


@dataclass(frozen=True)
class SubjectResult(Generic[TOut]):
    """
    Result slot for one subject, distinguishing success from isolated failure.

    Named ``SubjectResult`` rather than ``Outcome``: in medical research
    "outcome" already means the predicted clinical endpoint (survival,
    response), and ``FeatureTable.outcome_column`` uses it that way, so reusing
    it for "did this subject's computation succeed" would be a genuine
    domain-name collision.

    Batch habitat analysis must be able to continue when a single subject
    fails, while still reporting that failure honestly. Returning an explicit
    result rather than raising keeps that policy in the backend instead of
    scattering try/except through the algorithms.

    Attributes:
        subject_id: Subject this result belongs to.
        value: Computed result when successful, otherwise ``None``.
        error: Captured exception when failed, otherwise ``None``.
        from_cache: Whether the value was restored from a checkpoint instead of
            being recomputed.
    """

    subject_id: str
    value: Optional[TOut]
    error: Optional[BaseException]
    from_cache: bool = False

    def result(self) -> TOut:
        """
        Return the value or re-raise the captured failure.

        Named ``result`` after ``concurrent.futures.Future.result()``, the
        standard-library anchor for "give me the value or re-raise the error".

        Returns:
            The successful value.

        Raises:
            BaseException: The originally captured error, when this result
                represents a failure.
        """
        raise NotImplementedError("design prototype")


@runtime_checkable
class ExecutionBackend(Protocol):
    """
    Strategy for executing subject-level work.

    Every scheduling concern that v0.1 kept in the configuration schema --
    ``processes``, per-subject timeouts, graceful shutdown, spawn timeouts,
    failure policy, OOM backoff, resume -- belongs here instead. Algorithms then
    contain no scheduling code at all, and adding a Dask or cluster backend
    requires no change to any algorithm.

    A backend is an OPTIONAL ACCELERATOR, never a precondition. ``op(subject)``
    is always available directly, ``Cohort.map(op)`` runs the whole cohort with
    an implicit serial backend, and an explicit backend is constructed only when
    the user wants parallelism, timeouts or resume. A reader of the examples
    must never come away thinking that infrastructure has to be assembled before
    any work can be done -- that impression alone would push notebook users back
    to the CLI.
    """

    def map(
        self,
        op: SubjectOperator[TIn, TOut],
        items: Iterable[TIn],
        *,
        checkpoint: Optional["CheckpointStore"] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[SubjectResult[TOut]]:
        """
        Apply a subject-level operation across many subjects.

        Args:
            op: The subject-level operation to run.
            items: Subject-scoped inputs.
            checkpoint: Optional store used to skip already-computed subjects
                and to persist new results as they complete.
            progress: Optional callback receiving ``(completed, total)``.

        Returns:
            An iterator of outcomes in completion order; each outcome names its
            subject so callers can restore the canonical order.
        """


@runtime_checkable
class CheckpointStore(Protocol):
    """
    Persistence for resumable subject-level results.

    Kept orthogonal to both algorithms and backends so that resume behaviour is
    testable on its own and can be disabled entirely in notebook usage.
    """

    def get(self, key: str) -> Optional[Any]:
        """Return a previously stored result, or ``None`` when absent."""

    def put(self, key: str, value: Any) -> None:
        """Store a result under ``key``."""


# ---------------------------------------------------------------------------
# L1 boundaries: where data comes from and where artefacts go
# ---------------------------------------------------------------------------


@runtime_checkable
class DataSource(Protocol):
    """
    Anything that can produce a cohort.

    This protocol is the concrete mechanism behind the goal of embedding HABIT
    into the wider ecosystem. The v0.1 directory convention becomes one
    implementation among several rather than the only way in, so data prepared
    by nnU-Net, MONAI, a DICOM export, or an in-memory notebook session are all
    equally valid entry points.
    """

    def load(self) -> Cohort:
        """
        Build the cohort described by this source.

        Named ``load`` (a verb, like nibabel/MONAI readers) rather than the
        noun ``cohort()``, which read like an attribute.

        Returns:
            A cohort with a defined, reproducible subject order.
        """


@runtime_checkable
class ResultWriter(Protocol):
    """
    Anything that can persist HABIT outputs.

    Named ``ResultWriter`` rather than ``ArtifactSink``: "sink" is data-flow
    jargon (Beam/Flink/GStreamer), and "artifact" in radiology already means an
    imaging artefact (motion artefact, susceptibility artefact) -- a genuine
    domain collision. Every method is ``write_*``, so "Writer" (compare
    ``monai.data.ImageWriter``) is the accurate name.

    Separating the writer from the algorithms is what allows a caller to run a
    full habitat analysis entirely in memory, which is impossible in v0.1 where
    every workflow writes to ``out_dir`` by construction.
    """

    def write_habitat_map(self, habitat_map: HabitatMap) -> Optional[str]:
        """Persist one habitat map and return its location, when applicable."""

    def write_feature_table(self, table: FeatureTable, name: str) -> Optional[str]:
        """Persist one feature table and return its location, when applicable."""

    def write_habitat_model(self, model: HabitatModel) -> Optional[str]:
        """Persist a fitted habitat model and return its location."""


# ---------------------------------------------------------------------------
# Registry with introspection (serves both plugin authors and LLM agents)
# ---------------------------------------------------------------------------


@runtime_checkable
class ComponentRegistry(Protocol):
    """
    Name-to-implementation registry for ONE component family.

    This mirrors the registry classes that already exist in v0.1.x
    (``PreprocessorFactory``, ``ModelFactory``, ``FeatureExtractorRegistry``,
    ...): the class itself identifies the family, so callers pass only the
    implementation name and never a second magic string.

    The introspection members are what let an automated research agent discover
    what HABIT can do and construct a valid specification without a human
    transcribing the documentation, and equally what lets a GUI render a
    parameter form without hard-coding it.

    Cross-family lookup goes through the free functions that v0.1.x already
    exports -- ``list_plugins(domain)``, ``get_plugin_info(name, domain)``,
    ``get_param_schema(name, domain)`` -- which delegate to these registries.
    """

    #: Plugin domain name. The convention is ``domain == snake_case(ProtocolName)``,
    #: singular, so anyone who implements a protocol already knows its domain
    #: without looking it up. Example: the ``Supervoxelizer`` protocol has
    #: domain ``"supervoxelizer"`` and entry point group ``habit.supervoxelizer``.
    domain: ClassVar[str]

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        """
        Return a decorator registering a class under ``name`` in this family.

        Args:
            name: Unique implementation name within this family.

        Returns:
            A class decorator that registers and returns the class unchanged.
        """

    @classmethod
    def create(cls, name: str, **params: Any) -> Any:
        """
        Instantiate a registered component after validating ``params``.

        Args:
            name: Registered implementation name.
            **params: Parameters validated against the component's schema.

        Returns:
            The constructed component.

        Raises:
            ComponentNotFoundError: If the name is not registered.
            ConfigurationError: If the parameters fail schema validation.
        """

    @classmethod
    def available(cls) -> Tuple[str, ...]:
        """Return the registered implementation names in this family."""

    @classmethod
    def params_model(cls, name: str) -> type:
        """
        Return the Pydantic model describing one implementation's parameters.

        JSON Schema for a GUI or an agent is then ``.model_json_schema()``;
        keeping a single source of truth avoids a second, drifting schema.
        """
