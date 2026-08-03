# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Helpers for forwarding user-configured parameters to third-party estimators.

HABIT lets users set model hyperparameters in YAML, and those parameters are
forwarded to scikit-learn / XGBoost estimators. Two facts make a blind forward
unsafe:

1. The accepted keyword set differs per estimator class. ``random_state`` is
   injected globally by the pipeline builder, but ``KNeighborsClassifier`` and
   the naive Bayes estimators do not accept it, so forwarding it raises
   ``TypeError`` at construction time.
2. The accepted keyword set changes across library versions. A parameter that
   was valid when a config was written can disappear after an upgrade.

This module filters a parameter mapping down to what the target estimator
actually accepts and reports exactly what was dropped, so that a wrong or stale
key surfaces as an actionable message instead of an opaque crash.
"""

from __future__ import annotations

import contextlib
import difflib
import inspect
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Tuple

from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

# Parameters injected by HABIT itself rather than written by the user. Several
# estimators legitimately have no such parameter (KNN, naive Bayes), so dropping
# these is routine and is logged at debug level instead of warning level.
AUTO_INJECTED_PARAMS: FrozenSet[str] = frozenset({"random_state"})


@dataclass
class ParamReport:
    """
    Outcome of filtering one estimator's parameters.

    Attributes:
        model_name: HABIT model registry name.
        estimator: Class name of the underlying estimator.
        accepted: Parameter names forwarded to the estimator.
        ignored: User-supplied names the estimator does not accept.
        auto_dropped: HABIT-injected names the estimator does not accept.
    """

    model_name: str
    estimator: str
    accepted: List[str] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)
    auto_dropped: List[str] = field(default_factory=list)


# Collects reports for ``habit check-config`` without threading a reporter
# object through PipelineBuilder and every model wrapper.
_ACTIVE_REPORTS: ContextVar[Optional[List[ParamReport]]] = ContextVar(
    "habit_param_reports", default=None
)

# When true, an unsupported user parameter raises instead of warning.
_STRICT_MODE: ContextVar[bool] = ContextVar("habit_strict_model_params", default=False)


@contextlib.contextmanager
def collect_param_reports() -> Iterator[List[ParamReport]]:
    """
    Collect a :class:`ParamReport` for every estimator built inside the block.

    Used by ``habit check-config`` to tell the user which configured parameters
    will actually be applied, without running a training job.

    Yields:
        List[ParamReport]: Reports appended as estimators are constructed.
    """
    reports: List[ParamReport] = []
    token = _ACTIVE_REPORTS.set(reports)
    try:
        yield reports
    finally:
        _ACTIVE_REPORTS.reset(token)


@contextlib.contextmanager
def strict_model_params(enabled: bool = True) -> Iterator[None]:
    """
    Turn unsupported model parameters into errors inside the block.

    Args:
        enabled: When True, an unsupported user parameter raises ``ValueError``
            instead of emitting a warning.
    """
    token = _STRICT_MODE.set(bool(enabled))
    try:
        yield
    finally:
        _STRICT_MODE.reset(token)


def get_accepted_params(estimator_cls: type) -> Tuple[FrozenSet[str], bool]:
    """
    Inspect an estimator constructor and return the keywords it accepts.

    Args:
        estimator_cls: Estimator class to inspect, e.g.
            ``sklearn.neighbors.KNeighborsClassifier``.

    Returns:
        Tuple[FrozenSet[str], bool]: The explicitly named keyword parameters,
        and whether the constructor also declares ``**kwargs``. When it does,
        any keyword is forwardable and the name set is only informational
        (``XGBClassifier`` is the practical example).
    """
    try:
        signature = inspect.signature(estimator_cls.__init__)
    except (TypeError, ValueError):
        # C extensions without an introspectable signature: forward everything
        # rather than silently dropping parameters the estimator may support.
        return frozenset(), True

    accepted: set[str] = set()
    accepts_var_keyword = False
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_keyword = True
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        accepted.add(name)
    return frozenset(accepted), accepts_var_keyword


def _describe_rejected(rejected: List[str], accepted: FrozenSet[str]) -> str:
    """
    Build a human-readable report for rejected parameter names.

    Each rejected name is annotated with the closest accepted name when one is
    similar enough, which turns a silent typo into a fixable hint.

    Args:
        rejected: Parameter names the estimator does not accept.
        accepted: Parameter names the estimator does accept.

    Returns:
        str: Comma-separated description, e.g. ``n_estimator (did you mean
        'n_estimators'?), foo``.
    """
    parts: List[str] = []
    for name in sorted(rejected):
        suggestions = difflib.get_close_matches(name, sorted(accepted), n=1, cutoff=0.7)
        if suggestions:
            parts.append(f"{name} (did you mean '{suggestions[0]}'?)")
        else:
            parts.append(name)
    return ", ".join(parts)


def filter_estimator_params(
    estimator_cls: type,
    params: Mapping[str, Any],
    *,
    model_name: str = "",
) -> Dict[str, Any]:
    """
    Keep only the parameters that ``estimator_cls`` can accept.

    Args:
        estimator_cls: Target estimator class.
        params: Candidate parameters, typically merged from wrapper defaults and
            the user's YAML ``params`` block.
        model_name: HABIT model registry name used in log messages. Falls back
            to the estimator class name when empty.

    Returns:
        Dict[str, Any]: Subset of ``params`` that can be forwarded safely. When
        the estimator declares ``**kwargs``, ``params`` is returned unchanged.

    Raises:
        ValueError: When strict mode is active (see :func:`strict_model_params`)
            and a user-supplied parameter is not accepted by the estimator.
    """
    label = model_name or estimator_cls.__name__
    accepted, accepts_var_keyword = get_accepted_params(estimator_cls)
    if accepts_var_keyword:
        # The estimator forwards unknown keywords itself (XGBoost), so filtering
        # here would reject parameters it actually supports.
        _record_report(
            ParamReport(
                model_name=label,
                estimator=estimator_cls.__name__,
                accepted=sorted(params),
            )
        )
        return dict(params)

    kept: Dict[str, Any] = {}
    rejected_user: List[str] = []
    rejected_auto: List[str] = []
    for key, value in params.items():
        if key in accepted:
            kept[key] = value
        elif key in AUTO_INJECTED_PARAMS:
            rejected_auto.append(key)
        else:
            rejected_user.append(key)

    _record_report(
        ParamReport(
            model_name=label,
            estimator=estimator_cls.__name__,
            accepted=sorted(kept),
            ignored=sorted(rejected_user),
            auto_dropped=sorted(rejected_auto),
        )
    )

    if rejected_auto:
        logger.debug(
            "[%s] %s does not accept HABIT-injected parameter(s): %s. Dropped.",
            label,
            estimator_cls.__name__,
            ", ".join(sorted(rejected_auto)),
        )
    if rejected_user:
        message = (
            f"[{label}] {len(rejected_user)} configured parameter(s) not supported "
            f"by {estimator_cls.__name__}: "
            f"{_describe_rejected(rejected_user, accepted)}. "
            "Check the spelling and the installed library version."
        )
        if _STRICT_MODE.get():
            raise ValueError(message)
        logger.warning("%s Ignored.", message)
    return kept


def _record_report(report: ParamReport) -> None:
    """
    Append a report when a collection context is active.

    Args:
        report: Filtering outcome for one estimator.
    """
    reports = _ACTIVE_REPORTS.get()
    if reports is not None:
        reports.append(report)


def build_estimator_params(
    estimator_cls: type,
    defaults: Mapping[str, Any],
    user_params: Mapping[str, Any],
    *,
    model_name: str = "",
) -> Dict[str, Any]:
    """
    Merge wrapper defaults with user parameters, then filter for the estimator.

    User values override the wrapper defaults, and any additional user key is
    forwarded as-is so that YAML can reach parameters the wrapper does not name
    explicitly. Keys whose value is ``None`` are dropped when the wrapper did
    not declare them as a default, so that an unset optional parameter falls
    back to the estimator's own default instead of an explicit ``None``.

    Args:
        estimator_cls: Target estimator class.
        defaults: Wrapper-level defaults applied when the user omits a key.
        user_params: Parameters coming from the YAML ``params`` block.
        model_name: HABIT model registry name used in log messages.

    Returns:
        Dict[str, Any]: Parameters ready to splat into ``estimator_cls(...)``.
    """
    merged: Dict[str, Any] = dict(defaults)
    for key, value in user_params.items():
        if key not in merged and value is None:
            continue
        merged[key] = value
    return filter_estimator_params(estimator_cls, merged, model_name=model_name)
