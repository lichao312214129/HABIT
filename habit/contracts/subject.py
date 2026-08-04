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
"""Subject and cohort contracts -- replacing the directory convention.

In v0.1 "a subject" only existed as a folder name discovered by scanning a
data directory. Making the subject an object is what decouples HABIT from its
own directory layout and therefore what makes it embeddable.

Note on immutability vs pickling: ``Subject`` stores plain (defensively
copied) dicts rather than ``MappingProxyType`` views, because mapping proxies
cannot be pickled and lazy subjects are designed to cross process boundaries
under parallel backends. Treat the mappings as read-only.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

from habit.exceptions import HABITAPIError, ProcessingError
from habit.contracts.geometry import Geometry
from habit.contracts.image import ImageRef, ImageVolume, MaskVolume

if TYPE_CHECKING:
    # Lazy, typing-only: the directory adapter lives in L1 and the execution
    # backend protocol is only needed for annotations.
    from habit.contracts.ops import ExecutionBackend
    from habit.execution.checkpoint import CheckpointStore

__all__ = [
    "Subject",
    "Cohort",
    "CohortFingerprint",
    "cohort_from_directory",
]

#: Salt for the cohort digest so it cannot be confused with a raw id hash.
_COHORT_DIGEST_SALT = "habit-cohort-v1"


def _materialize_image(ref: ImageRef, modality: str) -> ImageVolume:
    """
    Materialise one image reference into an :class:`ImageVolume`.

    Prefers the richest materialisation the reference offers: an existing
    volume is returned unchanged, a reference with ``load_volume()`` supplies
    full physical metadata, and a bare reference falls back to ``load()``
    plus its ``geometry`` (or an identity grid when none is exposed). When a
    reference cannot label its own volume, the modality key carried by the
    ``Subject`` mapping is applied via an immutable ``dataclasses.replace``.

    Args:
        ref: The lazy reference held by the subject.
        modality: Modality key, used for error messages and labelling.

    Returns:
        The materialised volume.

    Raises:
        KeyError: Propagated when the modality key is absent (handled by the
            caller).
    """
    if isinstance(ref, ImageVolume):
        return ref
    load_volume = getattr(ref, "load_volume", None)
    if callable(load_volume):
        volume = load_volume()
        if isinstance(volume, ImageVolume):
            if volume.modality is None:
                volume = dataclasses.replace(volume, modality=modality)
            return volume
    array = ref.load()
    geometry = getattr(ref, "geometry", None)
    if geometry is None:
        geometry = Geometry.from_array(tuple(int(v) for v in array.shape))
    return ImageVolume(
        data=array,
        spacing=tuple(geometry.spacing),
        origin=tuple(geometry.origin),
        direction=tuple(geometry.direction),
        modality=modality,
    )


def _materialize_mask(ref: ImageRef, roi_name: str) -> MaskVolume:
    """
    Materialise one mask reference into a :class:`MaskVolume`.

    Same materialisation ladder as :func:`_materialize_image`, specialised to
    label volumes.

    Args:
        ref: The lazy reference held by the subject.
        roi_name: ROI key, used for error messages and labelling.

    Returns:
        The materialised mask.
    """
    if isinstance(ref, MaskVolume):
        return ref
    load_volume = getattr(ref, "load_volume", None)
    if callable(load_volume):
        volume = load_volume()
        if isinstance(volume, MaskVolume):
            return volume
    array = ref.load()
    geometry = getattr(ref, "geometry", None)
    if geometry is None:
        geometry = Geometry.from_array(tuple(int(v) for v in array.shape))
    return MaskVolume(
        data=array,
        spacing=tuple(geometry.spacing),
        origin=tuple(geometry.origin),
        direction=tuple(geometry.direction),
        modality=roi_name,
    )


@dataclass(frozen=True)
class Subject:
    """
    One imaging subject.

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

    def __post_init__(self) -> None:
        """Validate identity and defensively copy the mapping fields."""
        if not isinstance(self.subject_id, str) or not self.subject_id.strip():
            raise HABITAPIError("subject_id must be a non-empty string.")
        # Plain dict copies: MappingProxyType would break pickling, and lazy
        # subjects are designed to cross process boundaries (see module doc).
        object.__setattr__(self, "images", dict(self.images))
        object.__setattr__(self, "masks", dict(self.masks))
        object.__setattr__(self, "metadata", dict(self.metadata))

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
        if modality not in self.images:
            raise KeyError(
                f"Subject {self.subject_id!r} has no modality {modality!r}. "
                f"Available: {sorted(self.images)}."
            )
        return _materialize_image(self.images[modality], modality)

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
        if roi_name is None:
            if len(self.masks) != 1:
                raise ValueError(
                    f"Subject {self.subject_id!r} has {len(self.masks)} masks; "
                    "pass roi_name explicitly."
                )
            roi_name = next(iter(self.masks))
        if roi_name not in self.masks:
            raise KeyError(
                f"Subject {self.subject_id!r} has no ROI {roi_name!r}. "
                f"Available: {sorted(self.masks)}."
            )
        return _materialize_mask(self.masks[roi_name], roi_name)


@dataclass(frozen=True)
class CohortFingerprint:
    """
    Non-identifiable description of the cohort behind a fitted model.

    Sharing a habitat model without describing the cohort that defined it
    would be scientifically meaningless, but sharing subject identifiers
    would be unsafe. This type is the deliberate middle ground.

    Attributes:
        n_subjects: Number of subjects used for fitting.
        modalities: Modality names consumed, in canonical order.
        subject_id_digest: Salted digest of the ordered subject id list,
            which proves two runs used the same cohort without revealing
            identifiers.
        name: Optional cohort label, e.g. ``"HCC-DCE-training"``.
        description: Free-text description intended for a model card.
    """

    n_subjects: int
    modalities: Tuple[str, ...]
    subject_id_digest: str
    name: Optional[str] = None
    description: Optional[str] = None


class Cohort(Sequence[Subject]):
    """
    Ordered collection of subjects.

    Order is part of the contract, not an implementation detail:
    population-level clustering can be sensitive to subject order, so a
    reproducible cohort must have a defined, recorded ordering.

    Args:
        subjects: Subjects in canonical order.
        name: Human-readable cohort name used in reports, e.g. ``"training"``.
        metadata: Cohort-level attributes such as centre, scanner, or study.

    Raises:
        HABITAPIError: If ``subject_id`` values are missing, blank, or
            duplicated.
    """

    def __init__(
        self,
        subjects: Sequence[Subject],
        *,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._subjects: Tuple[Subject, ...] = tuple(subjects)
        ids = [subject.subject_id for subject in self._subjects]
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        if duplicates:
            raise HABITAPIError(
                f"Cohort subject_id values must be unique; duplicated: {duplicates}."
            )
        self.name = name
        self.metadata: Dict[str, Any] = dict(metadata or {})

    def __len__(self) -> int:
        return len(self._subjects)

    @overload
    def __getitem__(self, index: int) -> Subject: ...

    @overload
    def __getitem__(self, index: slice) -> "Cohort": ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Subject, "Cohort"]:
        """Index one subject, or slice to a new cohort preserving metadata."""
        if isinstance(index, slice):
            return Cohort(
                self._subjects[index],
                name=self.name,
                metadata=self.metadata,
            )
        return self._subjects[index]

    def __iter__(self) -> Iterator[Subject]:
        return iter(self._subjects)

    @property
    def subject_ids(self) -> Tuple[str, ...]:
        """Return the subject identifiers in canonical cohort order."""
        return tuple(subject.subject_id for subject in self._subjects)

    @classmethod
    def from_directory(
        cls,
        root: Union[str, Path],
        *,
        modalities: Sequence[str],
        roi: str,
        name: Optional[str] = None,
        images_folder: str = "images",
        masks_folder: str = "masks",
    ) -> "Cohort":
        """
        Build a cohort from HABIT's conventional directory layout.

        A thin shortcut over ``DirectoryDataSource(...).load()``, provided
        because reading a folder is the overwhelmingly common first line of a
        notebook session and should not require learning the adapter layer.
        The adapter import is lazy so that the contracts layer never depends
        on the filesystem-touching adapter layer at import time.

        Args:
            root: Directory root holding ``images_folder`` and
                ``masks_folder`` with one subdirectory per subject.
            modalities: Modality keys to load, in the order the analysis
                needs.
            roi: Mask key identifying the region of interest.
            name: Human-readable cohort name used in reports.
            images_folder: Name of the images subdirectory under ``root``.
            masks_folder: Name of the masks subdirectory under ``root``.

        Returns:
            A cohort with a defined, reproducible subject order (sorted
            subject ids).
        """
        from habit.adapters.directory import DirectoryDataSource

        return DirectoryDataSource(
            root,
            modalities=modalities,
            roi=roi,
            images_folder=images_folder,
            masks_folder=masks_folder,
            name=name,
        ).load()

    def filter(self, predicate: Callable[[Subject], bool]) -> "Cohort":
        """
        Return a new cohort containing subjects satisfying ``predicate``.

        Args:
            predicate: Callable receiving a :class:`Subject` and returning
                bool.

        Returns:
            A new cohort preserving the relative order of retained subjects.
        """
        return Cohort(
            [subject for subject in self._subjects if predicate(subject)],
            name=self.name,
            metadata=self.metadata,
        )

    def map(
        self,
        op: Callable[[Subject], Any],
        *,
        backend: Optional["ExecutionBackend"] = None,
        checkpoint: Optional["CheckpointStore"] = None,
    ) -> Sequence[Any]:
        """
        Apply a subject-level operator to every subject, in cohort order.

        This is the middle rung of a deliberate three-step ladder:
        ``op(subject)`` for one subject, ``cohort.map(op)`` for all of them,
        and ``cohort.map(op, backend=...)`` only when parallelism,
        per-subject timeouts or resume are actually wanted. Because
        ``backend`` defaults to a serial one, a researcher can complete an
        entire study without ever learning that execution backends exist.

        Args:
            op: Any subject-level operator, i.e. any of the subject-level
                domain protocols or a ``SubjectPipeline``.
            backend: Execution strategy. Serial when omitted.
            checkpoint: Store enabling resume. Disabled when omitted.

        Returns:
            Results in cohort order, not completion order, so that
            downstream cohort-level steps stay reproducible.

        Raises:
            ProcessingError: If any subject failed; the message lists every
                failed subject id and its error.
        """
        from habit.execution.backends import SerialBackend
        from habit.utils.progress_utils import CustomTqdm

        runner = backend if backend is not None else SerialBackend()
        total = len(self._subjects)
        op_name = type(op).__name__
        bar = CustomTqdm(total=total, desc=f"Cohort.map[{op_name}]")

        def _progress(completed: int, expected: int) -> None:
            bar.total = expected
            bar.n = completed
            bar.refresh()

        try:
            # ``Cohort.map`` deliberately accepts plain callables (the middle
            # rung of the operator ladder); the backends duck-type them through
            # ``_cache_key_of``, which is wider than the declared
            # ``SubjectOperator`` parameter, so cast across that intended gap.
            results: List[Any] = list(
                runner.map(
                    cast(Any, op),
                    self._subjects,
                    checkpoint=checkpoint,
                    progress=_progress,
                )
            )
        finally:
            bar.close()

        by_subject: Dict[str, Any] = {result.subject_id: result for result in results}
        failures = {
            sid: result.error for sid, result in by_subject.items() if result.error
        }
        if failures:
            detail = "; ".join(
                f"{sid}: {type(err).__name__}: {err}" for sid, err in failures.items()
            )
            raise ProcessingError(
                f"{len(failures)}/{total} subject(s) failed in Cohort.map: {detail}"
            )
        ordered: List[Any] = []
        for subject in self._subjects:
            result = by_subject.get(subject.subject_id)
            if result is None:
                raise ProcessingError(
                    f"Backend returned no result for subject "
                    f"{subject.subject_id!r}."
                )
            ordered.append(result.result())
        return ordered

    def summarize(self, description: Optional[str] = None) -> CohortFingerprint:
        """
        Summarise the cohort for provenance and model cards.

        Named ``summarize`` rather than ``fingerprint`` because it returns a
        rich summary object, whereas ``Spec.fingerprint()`` returns a hash
        string; the returned type keeps the name ``CohortFingerprint``
        (nnU-Net's term for a dataset summary).

        Args:
            description: Optional free-text description for a model card.

        Returns:
            A fingerprint safe to embed in a shared ``HabitatModel``, i.e.
            containing no identifiable patient information.
        """
        digest = hashlib.sha256(
            (_COHORT_DIGEST_SALT + "\n" + "\n".join(self.subject_ids)).encode("utf-8")
        ).hexdigest()
        modalities: List[str] = []
        for subject in self._subjects:
            for modality in subject.images:
                if modality not in modalities:
                    modalities.append(modality)
        return CohortFingerprint(
            n_subjects=len(self),
            modalities=tuple(modalities),
            subject_id_digest=digest,
            name=self.name,
            description=description,
        )


def cohort_from_directory(
    root: Union[str, Path],
    *,
    modalities: Sequence[str],
    roi: str,
    name: Optional[str] = None,
    images_folder: str = "images",
    masks_folder: str = "masks",
) -> Cohort:
    """
    Build a cohort from HABIT's conventional directory layout.

    Top-level convenience wrapper kept alongside the class method per the
    v1.0 naming decisions: ``habit.cohort_from_directory(...)`` for notebook
    ergonomics, ``Cohort.from_directory(...)`` for object-style code. This
    function simply delegates to the class method.

    Args:
        root: Directory root holding the images and masks subdirectories.
        modalities: Modality keys to load, in analysis order.
        roi: Mask key identifying the region of interest.
        name: Human-readable cohort name used in reports.
        images_folder: Name of the images subdirectory under ``root``.
        masks_folder: Name of the masks subdirectory under ``root``.

    Returns:
        A cohort with a defined, reproducible subject order.
    """
    return Cohort.from_directory(
        root,
        modalities=modalities,
        roi=roi,
        name=name,
        images_folder=images_folder,
        masks_folder=masks_folder,
    )
