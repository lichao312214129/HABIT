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
"""
Feature Selector Registry

Provides a centralized registry for feature selection methods with metadata support.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from habit.registry.base import CallableRegistry
from habit.utils.log_utils import get_module_logger

logger = get_module_logger('ml.selector_registry')


class SelectorRegistry(CallableRegistry[Callable]):
    """
    Registry of feature selection functions.

    Uses the shared :class:`~habit.core.common.registry.CallableRegistry`
    contract (``register`` / ``get`` / ``available`` / ``get_entry`` /
    ``entries`` / ``register_params_model`` / ``get_params_model``). Register a
    selector with::

        @SelectorRegistry.register("variance", display_name="Variance Threshold",
                                   default_before_z_score=True)
        def variance_selector(context: SelectorContext) -> List[str]:
            ...

    Each entry stores ``func`` + ``display_name`` + ``default_before_z_score``.
    """

    kind = "feature selector"

    #: Selectors run after z-score scaling unless the entry opts in.
    default_metadata: Dict[str, Any] = {"default_before_z_score": False}


@dataclass(frozen=True)
class SelectorContext:
    """
    Explicit runtime context passed into selector functions.

    Standardizing this context makes selector contracts explicit and reduces
    runtime surprises from implicit parameter-name matching.

    Alignment guarantee
    -------------------
    ``run_selector`` builds this context so that ``X`` is already restricted to
    the candidate features and ``selected_features == list(X.columns)``, with
    identical ordering. Selector implementations may therefore derive feature
    names from either source and get the same answer. Any statistic computed
    column-wise on ``X`` can be labelled positionally with
    ``selected_features`` without risk of misalignment.
    """

    X: pd.DataFrame
    y: pd.Series
    selected_features: List[str]
    outdir: Optional[str] = None
    logger: Optional[logging.Logger] = None


@dataclass(frozen=True)
class SelectorResult:
    """Normalized selector output payload."""

    selected_features: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

def _resolve_candidate_features(
    X: pd.DataFrame,
    selected_features: Optional[List[str]],
) -> List[str]:
    """
    Validate and normalize the candidate feature list for one selector run.

    This is the single place where the selector input contract is enforced, so
    that individual selectors never have to defend themselves against
    misaligned inputs. Three failure modes are ruled out here:

    1. Duplicated column labels in ``X``. ``X[cols]`` would then expand a single
       requested label into several columns, silently breaking the
       "one column per feature name" assumption every selector relies on.
    2. Feature names absent from ``X``. Failing here produces an actionable
       error naming the offending features, instead of a bare pandas ``KeyError``
       raised deep inside a selector.
    3. Duplicated entries in ``selected_features``, which would produce repeated
       columns and inflate per-feature statistics.

    Args:
        X: Full feature matrix handed to :func:`run_selector`.
        selected_features: Candidate feature names, or ``None`` to use every
            column of ``X``.

    Returns:
        List[str]: De-duplicated candidate names, order preserved, every one of
        which is guaranteed to exist as exactly one column of ``X``.

    Raises:
        ValueError: If ``X`` carries duplicated column labels, or if any
            requested feature is missing from ``X``.
    """
    duplicated_columns = X.columns[X.columns.duplicated()].unique().tolist()
    if duplicated_columns:
        raise ValueError(
            "Feature matrix contains duplicated column labels, which makes "
            "feature-name based selection ambiguous. Duplicated: "
            f"{duplicated_columns}"
        )

    if selected_features is None:
        return X.columns.tolist()

    # dict.fromkeys de-duplicates while preserving first-seen order.
    unique_features = list(dict.fromkeys(selected_features))
    if len(unique_features) != len(selected_features):
        logger.warning(
            "selected_features contained %d duplicate entrie(s); duplicates were dropped.",
            len(selected_features) - len(unique_features),
        )

    known_columns = set(X.columns)
    missing = [f for f in unique_features if f not in known_columns]
    if missing:
        raise ValueError(
            f"selected_features references {len(missing)} feature(s) absent from "
            f"the input data: {missing}"
        )

    return unique_features


def run_selector(name: str, 
                X: pd.DataFrame, 
                y: pd.Series, 
                selected_features: Optional[List[str]] = None,
                **kwargs) -> List[str]:
    """
    Orchestrate the execution of a feature selector with smart parameter injection.

    The candidate features are validated and normalized up front, then ``X`` is
    sliced to exactly those columns. Both the sliced frame and the name list
    handed to the selector are derived from that same slice, so a selector can
    never see a feature-name list that disagrees with its data columns.
    """
    candidate_features = _resolve_candidate_features(
        X=X,
        selected_features=selected_features,
    )
    X_candidates = X[candidate_features]

    info = SelectorRegistry.get_entry(name)
    if info is None:
        raise ValueError(f"Feature selector '{name}' not found in registry.")
    selector_func = info['func']
    sig = inspect.signature(selector_func)
    bound_args = {}

    context = SelectorContext(
        X=X_candidates,
        y=y,
        # Names are read back off the sliced frame rather than reusing the input
        # list, making the frame the single source of truth for both order and
        # membership.
        selected_features=list(X_candidates.columns),
        outdir=kwargs.get("outdir"),
        logger=kwargs.get("logger", logger),
    )

    if "context" in sig.parameters:
        # New explicit contract.
        bound_args["context"] = context
        for param_name, param in sig.parameters.items():
            if param_name == "context":
                continue
            if param_name in kwargs:
                bound_args[param_name] = kwargs[param_name]
            elif param.default is not inspect.Parameter.empty:
                continue
            elif param.kind not in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                logger.warning(
                    "Required parameter '%s' for selector '%s' not found in config kwargs.",
                    param_name,
                    name,
                )
    else:
        # Backward-compatible contract with alias-based injection.
        roles = {
            "X": context.X,
            "y": context.y,
            "selected_features": context.selected_features,
            "outdir": context.outdir,
            "logger": context.logger,
        }
        alias_map = {
            "X": ["X", "x", "data", "features"],
            "y": ["y", "Y", "target", "labels"],
            "selected_features": ["selected_features", "feature_names"],
            "outdir": ["outdir", "output_dir", "save_path"],
            "logger": ["logger", "log"],
        }

        for param_name, param in sig.parameters.items():
            matched_role = None
            for role, aliases in alias_map.items():
                if param_name in aliases:
                    matched_role = role
                    break

            if matched_role:
                bound_args[param_name] = roles[matched_role]
            elif param_name in kwargs:
                bound_args[param_name] = kwargs[param_name]
            elif param.default is not inspect.Parameter.empty:
                continue
            elif param.kind not in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                logger.warning(
                    "Required parameter '%s' for selector '%s' not found in inputs or config.",
                    param_name,
                    name,
                )

    # Execute and handle return value
    try:
        logger.info(f"Running feature selector: {info['display_name']} on {len(candidate_features)} features...")
        result = selector_func(**bound_args)
        selected_list = _normalize_selected_features(result=result, selector_name=name)
        selected_list = _restrict_to_candidates(
            selected=selected_list,
            candidate_features=candidate_features,
            selector_name=name,
        )
        logger.info(f"Selector {name} completed: {len(selected_list)} features retained.")
        return selected_list
    except Exception as e:
        logger.error(f"Error executing selector '{name}': {e}")
        raise


def _restrict_to_candidates(
    selected: List[str],
    candidate_features: List[str],
    selector_name: str,
) -> List[str]:
    """
    Reduce a selector's output to a duplicate-free subset of its own input.

    Downstream code (pipelines, report writers) indexes DataFrames with the
    returned names, so a name that was never a candidate would surface much
    later as an opaque ``KeyError``. Filtering here keeps that guarantee local
    to the registry instead of relying on every caller to intersect.

    Names outside the candidate set are dropped at ``debug`` level rather than
    ``warning`` because some selectors legitimately consult an external
    reference table: ``icc``, for instance, returns every feature that passed
    its ICC threshold in a precomputed JSON file, which routinely covers
    features absent from the current matrix.

    Args:
        selected: Raw feature names returned by the selector.
        candidate_features: Names the selector was allowed to choose from.
        selector_name: Registry key, used for log messages only.

    Returns:
        List[str]: Names from ``selected`` that are candidates, first occurrence
        kept, original relative order preserved (some selectors, e.g. ``mrmr``,
        return names ranked by importance).
    """
    candidate_set = set(candidate_features)
    kept: List[str] = []
    seen: set = set()
    dropped: List[str] = []

    for feature in selected:
        if feature not in candidate_set:
            dropped.append(feature)
            continue
        if feature in seen:
            continue
        seen.add(feature)
        kept.append(feature)

    if dropped:
        logger.debug(
            "Selector '%s' returned %d name(s) outside its candidate set; dropped: %s",
            selector_name,
            len(dropped),
            dropped,
        )

    return kept


def _normalize_selected_features(result: Any, selector_name: str) -> List[str]:
    """
    Normalize diverse selector outputs into a list of feature names.

    Supported return formats:
    - List[str]
    - Tuple[List[str], ...]
    - Dict with key 'selected_features'
    - SelectorResult
    """
    if isinstance(result, SelectorResult):
        return list(result.selected_features)

    if isinstance(result, dict):
        if "selected_features" not in result:
            raise ValueError(
                f"Selector '{selector_name}' returned dict without 'selected_features' key."
            )
        return list(result["selected_features"])

    if isinstance(result, tuple):
        if not result:
            return []
        return list(result[0])

    return list(result)