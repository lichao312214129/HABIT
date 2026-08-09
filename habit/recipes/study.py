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
and the recipe design) separately from *on which cohort* to run it. That split
is the v1.0 API surface described in ``developer/api_upgrade/07`` §9.2: recipes
build studies, ``study.fit(cohort)`` returns a :class:`~habit.recipes.result.StudyResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from habit.contracts.ops import ExecutionBackend
from habit.contracts.subject import Cohort
from habit.exceptions import HABITAPIError
from habit.recipes.habitat import direct_pooling, one_step, two_step
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

#: Recipe name -> the L4 function implementing that design.
_RECIPE_BY_DESIGN: Mapping[str, Callable[..., StudyResult]] = {
    "two_step": two_step,
    "one_step": one_step,
    "direct_pooling": direct_pooling,
}


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


@dataclass(frozen=True)
class Study:
    """
    A habitat analysis declared independently of any cohort.

    Attributes:
        spec: The analysis to run.
        design: Recipe identifier (``two_step``, ``one_step``, ``direct_pooling``).
    """

    spec: HabitatSpec
    design: str

    def fit(
        self,
        cohort: Cohort,
        *,
        backend: Optional[ExecutionBackend] = None,
        checkpoint: Optional[CheckpointStore] = None,
        seed: Optional[int] = None,
    ) -> StudyResult:
        """
        Run this study on a cohort and return an in-memory result.

        Args:
            cohort: Subjects to analyse.
            backend: Optional execution backend (parallelism, resume policy).
            checkpoint: Optional checkpoint store forwarded to per-subject stages.
            seed: Optional override of ``spec.random_seed``.

        Returns:
            The completed study result.
        """
        recipe = _RECIPE_BY_DESIGN.get(self.design)
        if recipe is None:
            raise HABITAPIError(
                f"Study design {self.design!r} has no registered recipe."
            )
        return recipe(
            cohort,
            self.spec,
            backend=backend,
            checkpoint=checkpoint,
            seed=seed,
        )


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
