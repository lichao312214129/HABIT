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
"""Object-style habitat study entry points (L4).

A :class:`Study` bundles *what* to compute (the :class:`~habit.spec.specs.HabitatSpec`
and an optional design declaration) separately from *on which cohort* to run it,
and follows the sklearn estimator lifecycle:

.. code-block:: python

    study = two_step_habitat(modalities=["T1", "T2"], n_habitats="auto")
    study.fit(train_cohort)                  # -> self; study.model_ is fitted
    result = study.predict(new_cohort)       # apply the fitted definition
    train_result = study.fit_predict(train_cohort)  # fit + full result

This is the single public entry point for habitat analysis: the function-style
engines live privately in :mod:`habit.recipes.habitat` so users learn one
object with one ``fit`` verb instead of two vocabularies for the same work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from habit.contracts.habitat import HabitatMap, HabitatModel
from habit.contracts.inspection import StepObserver
from habit.contracts.ops import ExecutionBackend, ResultWriter
from habit.contracts.subject import Cohort, Subject
from habit.exceptions import HABITAPIError, NotFittedError
from habit.recipes.habitat import (
    _apply_habitat_model,
    _direct_pooling,
    _fit_habitat,
    _one_step,
    _two_step,
    _with_model_habitat_postprocessing,
)
from habit.recipes.result import StudyResult
from habit.spec.specs import HabitatSpec, Spec

if TYPE_CHECKING:
    from habit.execution.checkpoint import CheckpointStore

__all__ = [
    "Study",
    "two_step_habitat",
    "one_step_habitat",
    "direct_pooling_habitat",
]

#: Design name -> the private validator implementing that design's guards.
_RECIPE_BY_DESIGN: Mapping[str, Callable[..., StudyResult]] = {
    "two_step": _two_step,
    "one_step": _one_step,
    "direct_pooling": _direct_pooling,
}


def _infer_design(spec: HabitatSpec) -> str:
    """
    Derive the study design from the spec's declared dataflow.

    Args:
        spec: The analysis declaration to inspect.

    Returns:
        ``"one_step"`` for ``pooling="none"``; otherwise ``"two_step"`` when
        a supervoxelizer is declared and ``"direct_pooling"`` when not.
    """
    if spec.pooling == "none":
        return "one_step"
    return "two_step" if spec.supervoxelizer is not None else "direct_pooling"


def _coerce_habitat_features(
    habitat_features: Optional[Sequence[Union[str, Spec, Mapping[str, object]]]],
) -> Tuple[Spec, ...]:
    """
    Normalise habitat feature family names into ``Spec`` tuples.

    Args:
        habitat_features: Feature family names, specs, or spec-shaped dicts.

    Returns:
        Immutable tuple of feature-family specs.
    """
    if not habitat_features:
        return ()
    specs: list[Spec] = []
    for entry in habitat_features:
        if isinstance(entry, Spec):
            specs.append(entry)
        elif isinstance(entry, Mapping) and "name" in entry:
            specs.append(Spec.from_dict(entry))
        elif isinstance(entry, str):
            specs.append(Spec(name=entry, params={}))
        else:
            raise HABITAPIError(
                "habitat_features entries must be str, Spec, or mapping with 'name'; "
                f"got {type(entry).__name__}."
            )
    return tuple(specs)


def _habitat_fitter_params(
    n_habitats: Union[int, str],
    *,
    min_habitats: int = 2,
    max_habitats: int = 10,
    validation: str = "elbow",
) -> Mapping[str, object]:
    """
    Build fitter params from the convenience ``n_habitats`` knob.

    Args:
        n_habitats: Fixed cluster count or ``"auto"`` to search within bounds.
        min_habitats: Lower bound when ``n_habitats`` is ``"auto"``.
        max_habitats: Upper bound when ``n_habitats`` is ``"auto"``.
        validation: Selection criterion when searching automatically.

    Returns:
        Parameter mapping for ``habitat_model_fitter``.
    """
    if isinstance(n_habitats, str):
        if n_habitats.strip().lower() not in ("auto", "automatic"):
            raise HABITAPIError(
                f"n_habitats must be an int or 'auto'; got {n_habitats!r}."
            )
        return {
            "n_habitats": None,
            "min_habitats": min_habitats,
            "max_habitats": max_habitats,
            "validation": validation,
        }
    return {
        "n_habitats": int(n_habitats),
        "min_habitats": min_habitats,
        "max_habitats": max_habitats,
        "validation": validation,
    }


def _build_habitat_spec(
    design: str,
    *,
    modalities: Sequence[str],
    n_supervoxels: int = 50,
    n_habitats: Union[int, str] = "auto",
    habitat_features: Optional[Sequence[Union[str, Spec, Mapping[str, object]]]] = None,
    random_seed: Optional[int] = None,
    supervoxel_algorithm: str = "kmeans",
    habitat_fitter_algorithm: str = "kmeans",
    roi: str = "tumor",
) -> HabitatSpec:
    """
    Assemble a :class:`HabitatSpec` for one of the three habitat designs.

    Args:
        design: ``two_step``, ``one_step``, or ``direct_pooling``.
        modalities: Modality names passed to the raw voxel extractor.
        n_supervoxels: Supervoxel count for the two-step design.
        n_habitats: Fixed habitat count or ``"auto"``.
        habitat_features: Optional habitat feature families to compute.
        random_seed: Seed applied to every seedable component.
        supervoxel_algorithm: Registered supervoxelizer / one-step fitter name.
        habitat_fitter_algorithm: Registered cohort fitter name.
        roi: ROI keyword for the raw voxel extractor.

    Returns:
        A fully wired habitat specification.
    """
    fitter_params = dict(
        _habitat_fitter_params(n_habitats),
        n_init=10,
    )
    if design == "two_step":
        return HabitatSpec(
            name="two_step_habitat",
            voxel_feature_extractor=Spec(
                name="raw",
                params={"modalities": list(modalities), "roi": roi},
            ),
            supervoxelizer=Spec(
                name=supervoxel_algorithm,
                params={"n_supervoxels": n_supervoxels},
            ),
            habitat_model_fitter=Spec(
                name=habitat_fitter_algorithm,
                params=fitter_params,
            ),
            habitat_assigner=Spec(name="nearest_centroid", params={}),
            habitat_features=_coerce_habitat_features(habitat_features),
            random_seed=random_seed,
            pooling="cohort",
        )
    if design == "one_step":
        return HabitatSpec(
            name="one_step_habitat",
            voxel_feature_extractor=Spec(
                name="raw",
                params={"modalities": list(modalities), "roi": roi},
            ),
            supervoxelizer=None,
            habitat_model_fitter=Spec(
                name=supervoxel_algorithm,
                params=fitter_params,
            ),
            habitat_assigner=Spec(name="nearest_centroid", params={}),
            habitat_features=_coerce_habitat_features(habitat_features),
            random_seed=random_seed,
            pooling="none",
        )
    if design == "direct_pooling":
        return HabitatSpec(
            name="direct_pooling_habitat",
            voxel_feature_extractor=Spec(
                name="raw",
                params={"modalities": list(modalities), "roi": roi},
            ),
            supervoxelizer=None,
            habitat_model_fitter=Spec(
                name=habitat_fitter_algorithm,
                params=fitter_params,
            ),
            habitat_assigner=Spec(name="nearest_centroid", params={}),
            habitat_features=_coerce_habitat_features(habitat_features),
            random_seed=random_seed,
            pooling="cohort",
        )
    raise HABITAPIError(
        f"Unknown habitat design {design!r}; expected one of "
        f"{sorted(_RECIPE_BY_DESIGN)}."
    )


@dataclass
class Study:
    """
    A habitat analysis declared independently of any cohort.

    The lifecycle mirrors a sklearn estimator: :meth:`fit` learns the
    cohort-level habitat definition and returns ``self``; :meth:`predict`
    projects that definition onto a new cohort; :meth:`fit_predict` fits and
    hands back the full :class:`~habit.recipes.result.StudyResult` in one
    call. Fitted state is exposed through the trailing-underscore attributes
    ``model_`` and ``fit_result_``.

    Attributes:
        spec: The analysis to run.
        design: Optional declared intent (``"two_step"``, ``"one_step"`` or
            ``"direct_pooling"``). When set, ``fit`` validates the spec
            against the design's guards before running, so a mismatched spec
            fails loudly instead of silently running a different dataflow.
            When ``None``, the dataflow declared by the spec itself
            (``pooling`` / stage list) decides what runs.
        model_: The fitted :class:`~habit.contracts.habitat.HabitatModel`;
            ``None`` until fitted, and ``None`` after fitting a ``one_step``
            study (that design defines habitats per subject, so there is no
            cohort-level model to publish).
        fit_result_: The :class:`~habit.recipes.result.StudyResult` produced
            by the latest :meth:`fit`; ``None`` until fitted.

    See Also
    --------
    habit.spec.HabitatSpec : Frozen analysis declaration used by this study.
    habit.contracts.HabitatModel : Fitted cohort habitat definition.
    habit.recipes.StudyResult : In-memory artefacts from ``fit_predict``.
    habit.recipes.two_step_habitat : Factory for the classical two-step design.
    """

    spec: HabitatSpec
    design: Optional[str] = None
    model_: Optional[HabitatModel] = field(
        default=None, init=False, repr=False, compare=False
    )
    fit_result_: Optional[StudyResult] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the declared design before any compute happens."""
        if self.design is not None and self.design not in _RECIPE_BY_DESIGN:
            raise HABITAPIError(
                f"Study design {self.design!r} has no registered recipe; "
                f"expected one of {sorted(_RECIPE_BY_DESIGN)}."
            )

    def fit(
        self,
        cohort: Cohort,
        *,
        backend: Optional[ExecutionBackend] = None,
        checkpoint: Optional[CheckpointStore] = None,
        seed: Optional[int] = None,
        inspect: Optional[StepObserver] = None,
        report: Optional[Any] = None,
        writer: Optional[ResultWriter] = None,
        retain: str = "all",
        on_subject_complete: Optional[
            Callable[[Subject, HabitatMap, HabitatModel], None]
        ] = None,
        persist_subject_models: bool = True,
    ) -> "Study":
        """
        Learn the habitat definition on a cohort; return ``self``.

        Args:
            cohort: Subjects to analyse.
            backend: Optional execution backend (parallelism, resume policy).
            checkpoint: Optional checkpoint store forwarded to per-subject stages.
            seed: Optional override of ``spec.random_seed``.
            inspect: Optional step observer for in-memory debugging / QA.
                Unsupported with the process backend.
            report: Optional :class:`~habit.report.Report` declaring what to
                persist and draw as each subject completes. This is the
                primary streaming API. ``writer`` / ``retain`` /
                ``on_subject_complete`` remain as shorthands that fill an
                implicit report.
            writer: Optional streaming writer (``one_step`` design only):
                each subject's habitat map is persisted the moment the
                backend yields it, so a crashed run keeps completed subjects.
            retain: ``"all"`` (default) keeps every artefact in memory;
                ``"maps"`` drops voxel-level clustering units (the
                memory-dominant payload of voxel-level designs); ``"tables"``
                additionally drops habitat maps and requires ``writer``.
            on_subject_complete: Optional parent-process callback
                ``(subject, habitat_map, model)`` fired once per completed
                subject -- including checkpoint-resumed ones -- before
                retention stripping. Prefer a figure atom on ``report``.
            persist_subject_models: With a streaming ``writer`` and no
                explicit ``report.persist``, also write
                ``<subject_id>.habitatmodel`` for each subject.

        Returns:
            ``self``, fitted: ``model_`` holds the cohort-level definition
            (except for the ``one_step`` design) and ``fit_result_`` the full
            study result.
        """
        if self.design is None:
            recipe: Callable[..., StudyResult] = _fit_habitat
        else:
            recipe = _RECIPE_BY_DESIGN[self.design]
        result = recipe(
            cohort,
            self.spec,
            backend=backend,
            seed=seed,
            checkpoint=checkpoint,
            inspect=inspect,
            writer=writer,
            retain=retain,
            on_subject_complete=on_subject_complete,
            persist_subject_models=persist_subject_models,
            report=report,
        )
        self.model_ = result.habitat_model
        self.fit_result_ = result
        return self

    def predict(
        self,
        cohort: Cohort,
        *,
        backend: Optional[ExecutionBackend] = None,
        checkpoint: Optional[CheckpointStore] = None,
        seed: Optional[int] = None,
        inspect: Optional[StepObserver] = None,
    ) -> StudyResult:
        """
        Apply the fitted habitat definition to a (new) cohort.

        The model's own cohort-level preprocessing state is restored and
        re-applied, because centroids only mean something in the feature
        space they were computed in.

        Args:
            cohort: Subjects to label.
            backend: Optional execution backend. Serial when omitted.
            checkpoint: Optional store enabling per-subject resume; keys
                scope on ``model_.model_id``.
            seed: Optional override of ``spec.random_seed``.
            inspect: Optional step observer for in-memory debugging / QA.
                Unsupported with the process backend.

        Returns:
            The study result for the projected cohort: habitat maps, the
            habitat feature table and the run manifest, all in memory.

        Raises:
            NotFittedError: If the study has no fitted model yet.
            HABITAPIError: If the study ran a ``one_step`` fit, which defines
                habitats per subject and therefore has no cohort-level model
                to apply.
        """
        if self.model_ is None:
            if self.fit_result_ is None:
                raise NotFittedError(
                    "Study is not fitted yet; call fit(cohort) first, or load "
                    "a published definition with Study.from_model(...)."
                )
            raise HABITAPIError(
                f"The {self.design or 'one_step'!r} design defines habitats "
                "inside each subject independently, so there is no "
                "cohort-level model to apply to new data."
            )
        return _apply_habitat_model(
            cohort,
            self.spec,
            self.model_,
            backend=backend,
            seed=seed,
            checkpoint=checkpoint,
            inspect=inspect,
        )

    def fit_predict(
        self,
        cohort: Cohort,
        *,
        backend: Optional[ExecutionBackend] = None,
        checkpoint: Optional[CheckpointStore] = None,
        seed: Optional[int] = None,
        inspect: Optional[StepObserver] = None,
        report: Optional[Any] = None,
        writer: Optional[ResultWriter] = None,
        retain: str = "all",
        on_subject_complete: Optional[
            Callable[[Subject, HabitatMap, HabitatModel], None]
        ] = None,
        persist_subject_models: bool = True,
    ) -> StudyResult:
        """
        Fit on a cohort and return the full study result.

        Equivalent to ``fit(cohort).fit_result_``, provided for the common
        case where the training-cohort artefacts (maps, features, manifest)
        are wanted immediately.

        Args:
            cohort: Subjects to analyse.
            backend: Optional execution backend (parallelism, resume policy).
            checkpoint: Optional checkpoint store forwarded to per-subject stages.
            seed: Optional override of ``spec.random_seed``.
            inspect: Optional step observer for in-memory debugging / QA.
            report: Optional :class:`~habit.report.Report`; see :meth:`fit`.
            writer: Optional streaming writer (``one_step`` design only);
                see :meth:`fit`.
            retain: In-memory retention mode; see :meth:`fit`.
            on_subject_complete: Optional per-subject completion callback;
                see :meth:`fit`.
            persist_subject_models: Write per-subject models when streaming;
                see :meth:`fit`.

        Returns:
            The completed study result.
        """
        self.fit(
            cohort,
            backend=backend,
            checkpoint=checkpoint,
            seed=seed,
            inspect=inspect,
            report=report,
            writer=writer,
            retain=retain,
            on_subject_complete=on_subject_complete,
            persist_subject_models=persist_subject_models,
        )
        assert self.fit_result_ is not None  # guaranteed by fit()
        return self.fit_result_

    @classmethod
    def from_model(
        cls,
        model: Union[HabitatModel, str, Path],
        spec: Optional[HabitatSpec] = None,
    ) -> "Study":
        """
        Build a fitted study from a published habitat model.

        This is the external-validation entry point: load a
        ``.habitatmodel`` artefact (or pass an in-memory
        :class:`~habit.contracts.habitat.HabitatModel`) and call
        :meth:`predict` on the new cohort.

        Args:
            model: A fitted model, or a path to a ``.habitatmodel`` archive.
            spec: The analysis declaration whose upstream stages must match
                the model's training spec. When ``None``, the spec embedded
                in the model archive is used.

        Returns:
            A study whose ``model_`` is already fitted, ready for
            :meth:`predict`.
        """
        if not isinstance(model, HabitatModel):
            model = HabitatModel.load(model)
        if spec is None:
            spec = HabitatSpec.from_dict(model.spec_payload)
        spec = _with_model_habitat_postprocessing(spec, model)
        study = cls(spec=spec, design=_infer_design(spec))
        study.model_ = model
        return study


def two_step_habitat(
    *,
    modalities: Sequence[str],
    n_supervoxels: int = 50,
    n_habitats: Union[int, str] = "auto",
    habitat_features: Optional[Sequence[Union[str, Spec, Mapping[str, object]]]] = None,
    random_seed: Optional[int] = None,
    supervoxel_algorithm: str = "kmeans",
    habitat_fitter_algorithm: str = "kmeans",
    roi: str = "tumor",
) -> Study:
    """
    Declare a classical two-step habitat study.

    Args:
        modalities: Modality names for the raw voxel extractor.
        n_supervoxels: Number of supervoxels per subject.
        n_habitats: Fixed habitat count or ``"auto"`` with elbow search.
        habitat_features: Optional habitat feature families (``"msi"``, etc.).
        random_seed: Seed for every seedable component.
        supervoxel_algorithm: Registered supervoxelizer name.
        habitat_fitter_algorithm: Registered cohort fitter name.
        roi: ROI keyword for voxel extraction.

    Returns:
        A :class:`Study` ready for :meth:`Study.fit`.

    See Also
    --------
    habit.recipes.Study : Sklearn-style fit / fit_predict / predict entry.
    habit.spec.HabitatSpec : Frozen analysis declaration the factory builds.
    habit.recipes.one_step_habitat : Per-subject habitat definition.
    habit.recipes.direct_pooling_habitat : Cohort clustering on voxel features.
    """
    return Study(
        spec=_build_habitat_spec(
            "two_step",
            modalities=modalities,
            n_supervoxels=n_supervoxels,
            n_habitats=n_habitats,
            habitat_features=habitat_features,
            random_seed=random_seed,
            supervoxel_algorithm=supervoxel_algorithm,
            habitat_fitter_algorithm=habitat_fitter_algorithm,
            roi=roi,
        ),
        design="two_step",
    )


def one_step_habitat(
    *,
    modalities: Sequence[str],
    n_habitats: Union[int, str] = "auto",
    habitat_features: Optional[Sequence[Union[str, Spec, Mapping[str, object]]]] = None,
    random_seed: Optional[int] = None,
    clustering_algorithm: str = "kmeans",
    roi: str = "tumor",
) -> Study:
    """
    Declare a one-step habitat study (habitats defined inside each subject).

    Args:
        modalities: Modality names for the raw voxel extractor.
        n_habitats: Fixed habitat count or ``"auto"``.
        habitat_features: Optional habitat feature families.
        random_seed: Seed for every seedable component.
        clustering_algorithm: Registered per-subject fitter name.
        roi: ROI keyword for voxel extraction.

    Returns:
        A :class:`Study` ready for :meth:`Study.fit`.
    """
    return Study(
        spec=_build_habitat_spec(
            "one_step",
            modalities=modalities,
            n_habitats=n_habitats,
            habitat_features=habitat_features,
            random_seed=random_seed,
            supervoxel_algorithm=clustering_algorithm,
            roi=roi,
        ),
        design="one_step",
    )


def direct_pooling_habitat(
    *,
    modalities: Sequence[str],
    n_habitats: Union[int, str] = "auto",
    habitat_features: Optional[Sequence[Union[str, Spec, Mapping[str, object]]]] = None,
    random_seed: Optional[int] = None,
    habitat_fitter_algorithm: str = "kmeans",
    roi: str = "tumor",
) -> Study:
    """
    Declare a direct-pooling habitat study (voxels pooled across the cohort).

    Args:
        modalities: Modality names for the raw voxel extractor.
        n_habitats: Fixed habitat count or ``"auto"``.
        habitat_features: Optional habitat feature families.
        random_seed: Seed for every seedable component.
        habitat_fitter_algorithm: Registered cohort fitter name.
        roi: ROI keyword for voxel extraction.

    Returns:
        A :class:`Study` ready for :meth:`Study.fit`.
    """
    return Study(
        spec=_build_habitat_spec(
            "direct_pooling",
            modalities=modalities,
            n_habitats=n_habitats,
            habitat_features=habitat_features,
            random_seed=random_seed,
            habitat_fitter_algorithm=habitat_fitter_algorithm,
            roi=roi,
        ),
        design="direct_pooling",
    )
