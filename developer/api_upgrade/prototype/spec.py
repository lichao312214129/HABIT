"""
HABIT v1.0 design prototype -- Spec, RunPolicy, and YAML/Python isomorphism.

STATUS: design-stage prototype. Signatures and contracts are the deliverable.

This module replaces the v0.1 "god configuration object". ``HabitatAnalysisConfig``
currently mixes four unrelated concerns in one class: where the data lives, what
the algorithm does, how execution is scheduled, and where results are written.
The split rule is deliberately simple:

    - Changing it changes the scientific result      -> Spec
    - It only says where the data comes from         -> DataSource (see protocols)
    - It changes nothing scientific                  -> RunPolicy

The second idea in this module is ISOMORPHISM. Any analysis constructed in
Python must be exportable to YAML, and any YAML must construct an equivalent
Python object. This is what allows HABIT to serve a clinician who only edits
YAML and a methodologist who only writes Python without maintaining two
architectures -- and it is why a paper's supplementary material can simply carry
the exported YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

__all__ = [
    "Spec",
    "HabitatSpec",
    "RunPolicy",
    "load_spec",
]


@dataclass(frozen=True)
class Spec:
    """
    Declarative description of one algorithm and its parameters.

    A ``Spec`` contains no filesystem paths, no worker counts, and no output
    settings. That restriction is what makes it meaningful to hash it, diff two
    of them, embed it in a shared :class:`HabitatModel`, and quote it in a
    manuscript.

    Attributes:
        domain: Plugin domain the component belongs to, plural, matching the
            entry point group ``habit.<domain>``, e.g. ``"feature_extractors"``.
        name: Registered implementation name, e.g. ``"kinetic"``.
        params: Parameters validated against the component's JSON Schema.
    """

    domain: str
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """
        Return a stable hash of the canonicalised specification.

        Canonicalisation sorts keys, normalises numeric types, and drops
        parameters left at their documented defaults, so that two specifications
        which are scientifically identical always agree. The fingerprint is used
        as the habitat model identity, the checkpoint key, and the cache key.

        Returns:
            A short hexadecimal digest.
        """
        raise NotImplementedError("design prototype")

    def to_dict(self) -> Mapping[str, Any]:
        """Return the plain-mapping form used for YAML and JSON serialisation."""
        raise NotImplementedError("design prototype")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Spec":
        """
        Build a specification from a mapping, validating it against the schema.

        Args:
            payload: Mapping with ``domain``, ``name``, and ``params``.

        Returns:
            A validated specification.

        Raises:
            ConfigurationError: If required keys are missing or parameters fail
                schema validation.
            ComponentNotFoundError: If no component is registered under the
                requested ``domain``/``name``.
        """
        raise NotImplementedError("design prototype")

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """
        Serialise to YAML, optionally writing it to disk.

        Args:
            path: Destination file. When ``None`` the YAML text is only
                returned.

        Returns:
            The YAML text.
        """
        raise NotImplementedError("design prototype")

    def describe(self) -> str:
        """
        Return an English sentence describing this step for a methods section.

        The sentence must only state what the component will actually do with
        the given parameters. Generating plausible but unexecuted methods text
        would make the whole reporting feature untrustworthy, so this is a hard
        constraint verified by tests.
        """
        raise NotImplementedError("design prototype")


@dataclass(frozen=True)
class HabitatSpec:
    """
    Composite specification of a complete habitat analysis.

    The three clustering strategies of v0.1 (``two_step``, ``one_step``,
    ``direct_pooling``) are no longer a hard-coded dispatch table. They are just
    different assemblies of the same component slots, so a user can define a
    fourth strategy without modifying HABIT.

    Attributes:
        voxel_features: Specification producing per-voxel features.
        supervoxel: Specification producing supervoxels. ``None`` means voxels
            are clustered directly, which is what ``one_step`` and
            ``direct_pooling`` do.
        habitat_model: Specification of the population-level estimator.
        habitat_features: Specifications of the habitat-level feature families
            to compute, e.g. MSI, ITH score, non-radiomics.
        random_seed: Seed applied to every stochastic component, recorded in
            provenance.
    """

    voxel_features: Spec
    supervoxel: Optional[Spec]
    habitat_model: Spec
    habitat_features: Tuple[Spec, ...] = ()
    random_seed: Optional[int] = None

    def fingerprint(self) -> str:
        """Return a stable hash covering every component specification."""
        raise NotImplementedError("design prototype")

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """
        Export the analysis as YAML.

        This is the Python-to-YAML half of the isomorphism: a methodologist can
        design an analysis in a notebook and hand the exported file to a
        clinical colleague who runs it through the CLI unchanged.
        """
        raise NotImplementedError("design prototype")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "HabitatSpec":
        """
        Build an analysis specification from YAML.

        This is the YAML-to-Python half of the isomorphism and the only path the
        CLI uses, which is why the CLI can stay a thin shell.

        Args:
            path: Path to a v1 specification file.

        Returns:
            The validated composite specification.
        """
        raise NotImplementedError("design prototype")

    def describe_methods(self, style: str = "radiology") -> str:
        """
        Render the specification as a manuscript methods paragraph.

        Args:
            style: Target venue convention, e.g. ``"radiology"`` or
                ``"nature"``. Only affects wording and ordering, never content.

        Returns:
            English prose describing every configured step and its parameters.
        """
        raise NotImplementedError("design prototype")

    @classmethod
    def json_schema(cls) -> Mapping[str, Any]:
        """
        Return the JSON Schema of a full habitat specification.

        Exposed so that a GUI can render a form and an automated agent can
        construct a valid analysis without a human transcribing documentation.
        """
        raise NotImplementedError("design prototype")


@dataclass(frozen=True)
class RunPolicy:
    """
    Everything that affects how a run executes but not what it concludes.

    Keeping these settings out of :class:`Spec` is what makes a specification
    comparable across machines: the same science can run serially in a notebook
    and across 32 processes on a workstation, and the fingerprint stays equal.

    Attributes:
        backend: Execution backend name, e.g. ``"serial"`` or ``"process_pool"``.
        workers: Worker count for parallel backends.
        subject_timeout_sec: Wall-clock budget per subject; ``None`` disables it.
        on_subject_failure: ``"continue"`` to isolate failures, ``"fail_fast"``
            to abort the run.
        oom_backoff: Whether to reduce workers after a fatal memory error.
        oom_reduce_workers_by: How many workers to drop on each backoff step.
        cap_workers_to_gpu_pool: Whether to clamp the worker count to the number
            of usable GPUs, for steps whose components require one.
        resume: Whether to reuse checkpointed subject-level results.
        checkpoint_dir: Where checkpoints live; ``None`` disables persistence.
        cache_dir: Optional cache root for intermediate artefacts.
    """

    backend: str = "serial"
    workers: int = 1
    subject_timeout_sec: Optional[float] = 900.0
    on_subject_failure: str = "continue"
    oom_backoff: bool = True
    oom_reduce_workers_by: int = 1
    cap_workers_to_gpu_pool: bool = False
    resume: bool = True
    checkpoint_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None


def load_spec(source: Union[str, Path, Mapping[str, Any]]) -> HabitatSpec:
    """
    Load a habitat specification from a file path or an in-memory mapping.

    Both inputs take the same validation path, which is the v0.1 property worth
    preserving: a dictionary written in a notebook is validated exactly as
    strictly as a YAML file edited by a clinician.

    Args:
        source: Path to a v1 YAML file, or an equivalent mapping.

    Returns:
        The validated composite specification.

    Raises:
        ConfigurationError: If validation fails, with a message naming the
            offending field path.
    """
    raise NotImplementedError("design prototype")
