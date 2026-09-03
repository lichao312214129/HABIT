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
"""The two feature-preprocessing chains: per subject and per cohort.

The distinction between them is NOT which granularity they process -- both
accept any unit-by-feature matrix -- but whether their fitted state crosses
subject boundaries:

* :class:`SubjectPreprocessingChain` recomputes everything from the matrix
  in front of it and keeps nothing. Its purpose is to remove BETWEEN-subject
  variation (scanner, sequence, intensity scale), which only works if each
  subject is normalised by its own statistics. Being stateless, it is
  identical at training and prediction time and cannot leak.

* :class:`CohortPreprocessingChain` learns from the pooled training cohort
  and replays that state on every later matrix. Its purpose is the opposite:
  to place different subjects' units in ONE comparable feature space, so the
  clusters that define habitats mean the same thing across subjects. Being
  stateful, it is the only leakage-sensitive step in habitat definition, and
  its state belongs in the published
  :class:`~habit.contracts.habitat.HabitatModel`.

Because state ownership is the only difference, the methods themselves are
shared: the identical ``winsorize`` implementation serves a per-subject voxel
chain and a cohort-level supervoxel chain.

v0.1 named these ``preprocessing_for_subject_level`` and
``preprocessing_for_group_level``, which conflated the axis above with data
granularity: the subject-level block always ran on voxel features, and the
group-level block ran on supervoxel features under ``two_step`` but on voxel
features under ``direct_pooling``. Separating the axes leaves the per-subject
chain reusable at BOTH granularities, which v0.1 could not express -- per
supervoxel radiomics had no stateless normalisation step available to it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.feature_preprocessing.methods import Impute
from habit.feature_preprocessing.registry import (
    FeaturePreprocessingMethodRegistry,
)
from habit.spec.specs import Spec

__all__ = [
    "CohortPreprocessingChain",
    "SubjectPreprocessingChain",
    "build_methods",
]

#: Method that must precede the rest, and is inserted when not requested.
_IMPUTE_NAME = "impute"


def build_methods(
    steps: Sequence[Union[Spec, Mapping[str, Any]]]
) -> Tuple[Any, ...]:
    """
    Instantiate preprocessing methods from their specifications.

    Args:
        steps: Ordered method specifications, each a
            :class:`~habit.spec.specs.Spec` or a
            ``{"name": ..., "params": {...}}`` mapping.

    Returns:
        The constructed methods, in order.

    Raises:
        ComponentNotFoundError: If a step names an unregistered method.
        ConfigurationError: If a step's parameters fail schema validation.
    """
    methods: List[Any] = []
    for step in steps:
        if isinstance(step, Spec):
            name, params = step.name, dict(step.params)
        else:
            name = str(step["name"])
            params = dict(step.get("params") or {})
        methods.append(FeaturePreprocessingMethodRegistry.create(name, **params))
    return tuple(methods)


def _validate_methods(methods: Sequence[Any], owner: str) -> Tuple[Any, ...]:
    """
    Check that every chain member implements the method interface.

    Args:
        methods: Candidate methods.
        owner: Chain name used in the error message.

    Returns:
        The methods as a tuple.

    Raises:
        HABITAPIError: If ``methods`` is empty or a member lacks
            ``fit``/``transform``/``spec``.
    """
    if not methods:
        raise HABITAPIError(
            f"{owner} requires at least one method; an empty chain should be "
            "expressed as no chain at all (None)."
        )
    for method in methods:
        missing = [
            attribute
            for attribute in ("fit", "transform", "spec")
            if not hasattr(method, attribute)
        ]
        if missing:
            raise HABITAPIError(
                f"{owner} received {type(method).__name__}, which lacks "
                f"{missing}. Feature preprocessing methods must provide "
                "fit(block), transform(block, state) and a spec property."
            )
    return tuple(methods)


def _ensure_impute_first(methods: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """
    Guarantee the chain starts by making its input finite.

    Every other method assumes finite values, so a chain without imputation
    would either crash inside scikit-learn or compute a quantile over an
    infinity. Rather than treat that as the user's problem, a default
    :class:`~habit.feature_preprocessing.methods.Impute` is prepended
    -- and because it enters ``methods``, it also enters the chain's spec and
    provenance. Nothing is applied that the record does not show.

    Args:
        methods: Methods as configured.

    Returns:
        The methods, with imputation first. A configuration that names
        ``impute`` explicitly is left untouched, including its position, so a
        study can deliberately impute later in the chain.
    """
    if any(method.spec.name == _IMPUTE_NAME for method in methods):
        return methods
    return (Impute(),) + methods


def _reject_non_finite(block: pd.DataFrame, owner: str) -> pd.DataFrame:
    """
    Turn non-finite values produced by a chain into an explicit error.

    The chain's input is imputed, so a non-finite OUTPUT means a method
    produced it -- a log of a non-positive argument, a division by a
    degenerate range. v0.1 coerced these to NaN and let the clustering step
    trip over them deep inside scikit-learn; naming the offending columns
    here is numerically equivalent whenever v0.1 would not have crashed.

    Args:
        block: Chain output to check.
        owner: Chain name used in the error message.

    Returns:
        ``block`` unchanged when every value is finite.

    Raises:
        HABITAPIError: If any value is not finite.
    """
    if block.empty:
        return block
    values = block.to_numpy(dtype=np.float64, copy=False)
    if np.isfinite(values).all():
        return block
    culprits = [
        str(column)
        for column in block.columns
        if not np.isfinite(block[column].to_numpy(dtype=np.float64)).all()
    ]
    raise HABITAPIError(
        f"{owner} produced non-finite values in feature column(s) "
        f"{culprits[:10]}"
        f"{' (and more)' if len(culprits) > 10 else ''}. A method generated "
        "them from finite inputs -- for example 'log' whose learned offset "
        "leaves a non-positive argument. Reorder or reconfigure the chain."
    )


class SubjectPreprocessingChain:
    """
    Stateless preprocessing of one subject's feature matrix.

    Applies its methods using statistics computed from the matrix it is
    given, discarding them afterwards. That is exactly what individual-level
    preprocessing means, and it is why this chain needs no train/predict
    distinction: an external validation subject is normalised by its own
    distribution, never by the training cohort's.

    The same instance can preprocess voxel features and supervoxel features,
    because neither the methods nor this chain inspect what a row represents.

    Args:
        methods: Ordered methods to apply. Must be non-empty. Imputation is
            prepended when not named explicitly.
    """

    def __init__(self, methods: Sequence[Any]) -> None:
        self._methods = _ensure_impute_first(
            _validate_methods(methods, "SubjectPreprocessingChain")
        )

    @property
    def methods(self) -> Tuple[Any, ...]:
        """Return the ordered methods, including any inserted imputation."""
        return self._methods

    @property
    def spec(self) -> Spec:
        """Return the composed specification of every method."""
        return Spec(
            name="subject_feature_preprocessor",
            params={"steps": [method.spec.to_dict() for method in self._methods]},
        )

    def set_random_state(self, seed: int) -> None:
        """
        Seed every stochastic method in the chain.

        Args:
            seed: Seed forwarded to methods exposing ``set_random_state``.
        """
        for method in self._methods:
            setter = getattr(method, "set_random_state", None)
            if callable(setter):
                setter(seed)

    def __call__(self, block: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess one matrix, fitting and discarding state per call.

        Args:
            block: Unit-by-feature matrix (rows = voxels or supervoxels).

        Returns:
            The preprocessed matrix, row-aligned and row-count preserving.
            Column count may shrink when a filtering method is present.

        Raises:
            HABITAPIError: If a method produces non-finite values.
        """
        if block.empty:
            return block.copy()
        current = block
        for method in self._methods:
            current = method.transform(current, method.fit(current))
        return _reject_non_finite(current, "SubjectPreprocessingChain")


class CohortPreprocessingChain:
    """
    Stateful preprocessing of the pooled cohort feature matrix.

    Learns its statistics ONCE from the training cohort and applies that
    frozen state to every later matrix, which is what makes units from
    different subjects comparable and therefore what makes a habitat
    definition transferable. It is also the single place where habitat
    definition can leak test information, so ``fit`` must see training data
    only.

    The fitted state is exposed via :attr:`state` and restorable via
    :meth:`from_state`, because it has to travel inside the published
    :class:`~habit.contracts.habitat.HabitatModel`: applying a habitat
    definition to a new cohort without its cohort-level preprocessing would
    silently place that cohort in a different feature space.

    Args:
        methods: Ordered methods to apply. Must be non-empty. Imputation is
            prepended when not named explicitly.
    """

    def __init__(self, methods: Sequence[Any]) -> None:
        self._methods = _ensure_impute_first(
            _validate_methods(methods, "CohortPreprocessingChain")
        )
        self._states: Optional[Tuple[Mapping[str, Any], ...]] = None
        self._fit_columns: Tuple[str, ...] = ()
        self._output_columns: Tuple[str, ...] = ()

    @property
    def methods(self) -> Tuple[Any, ...]:
        """Return the ordered methods, including any inserted imputation."""
        return self._methods

    @property
    def is_fitted(self) -> bool:
        """Return whether the chain has learned its state."""
        return self._states is not None

    @property
    def fit_columns(self) -> Tuple[str, ...]:
        """Return the feature columns the chain was fitted on."""
        return self._fit_columns

    @property
    def output_columns(self) -> Tuple[str, ...]:
        """Return the feature columns surviving the fitted chain."""
        return self._output_columns

    @property
    def spec(self) -> Spec:
        """Return the composed specification of every method."""
        return Spec(
            name="cohort_feature_preprocessor",
            params={"steps": [method.spec.to_dict() for method in self._methods]},
        )

    def set_random_state(self, seed: int) -> None:
        """
        Seed every stochastic method in the chain.

        Args:
            seed: Seed forwarded to methods exposing ``set_random_state``.
        """
        for method in self._methods:
            setter = getattr(method, "set_random_state", None)
            if callable(setter):
                setter(seed)

    def fit(self, block: pd.DataFrame) -> "CohortPreprocessingChain":
        """
        Learn every method's state from the TRAINING matrix.

        Args:
            block: Pooled training matrix (rows = units from every training
                subject).

        Returns:
            ``self``, fitted.

        Raises:
            HABITAPIError: If ``block`` has no rows or a method produces
                non-finite values.
        """
        if block.empty:
            raise HABITAPIError(
                "CohortPreprocessingChain.fit requires a non-empty pooled "
                "matrix; cohort-level statistics cannot be learned from no "
                "units."
            )
        current = block
        states: List[Mapping[str, Any]] = []
        for method in self._methods:
            state = method.fit(current)
            states.append(state)
            current = method.transform(current, state)
        _reject_non_finite(current, "CohortPreprocessingChain")
        self._states = tuple(states)
        self._fit_columns = tuple(str(column) for column in block.columns)
        self._output_columns = tuple(str(column) for column in current.columns)
        return self

    def transform(self, block: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted state to a matrix.

        Args:
            block: Matrix carrying the feature columns seen at fit time.

        Returns:
            The preprocessed matrix.

        Raises:
            HABITAPIError: If the chain is unfitted, the matrix lacks fitted
                columns, or a method produces non-finite values.
        """
        if self._states is None:
            raise HABITAPIError(
                "CohortPreprocessingChain must be fitted before transform. "
                "Fit it on the training cohort, or restore a fitted chain "
                "from a HabitatModel."
            )
        missing = [
            column for column in self._fit_columns if column not in block.columns
        ]
        if missing:
            raise HABITAPIError(
                "CohortPreprocessingChain was fitted on feature columns "
                f"{list(self._fit_columns)} but the matrix to transform is "
                f"missing {missing}. Apply the same upstream steps as at fit "
                "time, or re-fit."
            )
        current = block[list(self._fit_columns)]
        for method, state in zip(self._methods, self._states):
            current = method.transform(current, state)
        return _reject_non_finite(current, "CohortPreprocessingChain")

    def fit_transform(self, block: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on a matrix and return its transformation.

        Args:
            block: Pooled training matrix.

        Returns:
            The transformed training matrix.
        """
        return self.fit(block).transform(block)

    @property
    def state(self) -> Dict[str, Any]:
        """
        Return the fitted state for storage inside a habitat model.

        Returns:
            A mapping holding the chain specification, each method's state and
            the fitted/output column schemas. Method states may contain fitted
            scikit-learn objects, so the payload is pickle-serialisable rather
            than JSON-serialisable -- the same contract
            :meth:`~habit.contracts.habitat.HabitatModel.save` already uses
            for model payloads.

        Raises:
            HABITAPIError: If the chain is not fitted.
        """
        if self._states is None:
            raise HABITAPIError(
                "CohortPreprocessingChain has no state until it is fitted."
            )
        return {
            "spec": self.spec.to_dict(),
            "states": list(self._states),
            "fit_columns": list(self._fit_columns),
            "output_columns": list(self._output_columns),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CohortPreprocessingChain":
        """
        Restore a fitted chain from :attr:`state`.

        Args:
            state: Payload previously produced by :attr:`state`.

        Returns:
            The restored, fitted chain.

        Raises:
            HABITAPIError: If the payload is not a cohort chain state or its
                method count disagrees with its specification.
        """
        spec_payload = state.get("spec") or {}
        steps = (spec_payload.get("params") or {}).get("steps")
        if steps is None:
            raise HABITAPIError(
                "CohortPreprocessingChain.from_state requires a payload "
                "produced by CohortPreprocessingChain.state."
            )
        chain = cls(build_methods([Spec.from_dict(step) for step in steps]))
        states = list(state.get("states") or ())
        if len(states) != len(chain.methods):
            raise HABITAPIError(
                f"CohortPreprocessingChain state holds {len(states)} method "
                f"state(s) but its specification names {len(chain.methods)}."
            )
        chain._states = tuple(states)
        chain._fit_columns = tuple(
            str(column) for column in state.get("fit_columns") or ()
        )
        chain._output_columns = tuple(
            str(column) for column in state.get("output_columns") or ()
        )
        return chain
