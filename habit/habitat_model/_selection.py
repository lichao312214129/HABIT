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
"""Shared habitat-count selection for the built-in fitters.

The arithmetic lives in :mod:`habit.kernels.cluster_selection`; this module
only adapts it to the fitters -- validating the requested criteria, evaluating
each candidate count once no matter how many criteria were asked for, and
recording an auditable report of what was scored and what won.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from habit.exceptions import HABITAPIError
from habit.kernels.cluster_selection import score_direction, vote_best_index

__all__ = [
    "normalize_validation",
    "select_cluster_count",
    "build_selection_report",
]

#: Type of the caller-supplied scorer: given a candidate count and the
#: criteria to satisfy, fit once and return one score per criterion.
CandidateScorer = Callable[[int, Sequence[str]], Mapping[str, float]]


def normalize_validation(
    validation: Union[str, Iterable[str]],
    supported: Sequence[str],
) -> Tuple[str, ...]:
    """
    Normalise the ``validation`` argument to a tuple of criteria.

    A single name selects on its own; several names each cast one vote and the
    majority wins, mirroring the multi-method voting the v0.1 configuration
    schema allowed.

    Args:
        validation: One criterion name, or an iterable of names.
        supported: Criteria this fitter can compute.

    Returns:
        The requested criteria, de-duplicated, in the given order.

    Raises:
        HABITAPIError: If empty, or if any name is unsupported.
    """
    if isinstance(validation, str):
        requested = [validation]
    else:
        requested = [str(name) for name in validation]

    methods: List[str] = []
    for name in requested:
        if name not in methods:
            methods.append(name)
    if not methods:
        raise HABITAPIError("validation requires at least one criterion.")

    unknown = [name for name in methods if name not in tuple(supported)]
    if unknown:
        raise HABITAPIError(
            f"validation must be drawn from {tuple(supported)}; "
            f"got unsupported {unknown}."
        )
    return tuple(methods)


def select_cluster_count(
    candidates: Sequence[int],
    methods: Sequence[str],
    score_candidate: CandidateScorer,
) -> Tuple[int, Dict[str, List[float]]]:
    """
    Score every candidate count and return the selected one.

    Args:
        candidates: Candidate cluster counts, ascending.
        methods: Validation criteria, already normalised.
        score_candidate: Called once per candidate with all criteria, so a
            candidate is fitted a single time regardless of how many criteria
            are being voted on.

    Returns:
        ``(chosen_count, scores_by_method)`` where each score list follows
        ``candidates`` order.

    Raises:
        HABITAPIError: If ``candidates`` is empty or a criterion was not
            scored for every candidate.
    """
    counts = [int(k) for k in candidates]
    if not counts:
        raise HABITAPIError("Habitat-count selection requires candidates.")

    scores_by_method: Dict[str, List[float]] = {name: [] for name in methods}
    for count in counts:
        scored = score_candidate(count, tuple(methods))
        for name in methods:
            if name not in scored:
                raise HABITAPIError(
                    f"Criterion {name!r} was not scored for {count} habitats."
                )
            scores_by_method[name].append(float(scored[name]))

    chosen_index = vote_best_index(scores_by_method, methods)
    return counts[chosen_index], scores_by_method


def build_selection_report(
    candidates: Sequence[int],
    methods: Sequence[str],
    scores_by_method: Mapping[str, Sequence[float]],
    chosen: int,
) -> Dict[str, Any]:
    """
    Describe an automatic habitat-count selection for the model artefact.

    Carrying the curves on the model is what lets a downstream writer redraw
    the validation plot, or a reviewer re-derive the count, without refitting.

    Args:
        candidates: Candidate counts that were evaluated, ascending.
        methods: Criteria that voted.
        scores_by_method: Criterion -> score per candidate.
        chosen: The selected habitat count.

    Returns:
        A JSON-serialisable report.
    """
    return {
        "candidates": [int(k) for k in candidates],
        "methods": [str(name) for name in methods],
        "directions": {str(name): score_direction(name) for name in methods},
        "scores": {
            str(name): [float(value) for value in scores_by_method[name]]
            for name in methods
        },
        "selected": int(chosen),
    }
