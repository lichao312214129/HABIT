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
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Tuple

from habit.exceptions import HABITAPIError
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

# Parameters injected by HABIT itself rather than written by the user. Several
# estimators legitimately have no such parameter (KNN, naive Bayes), so dropping
# these is routine and is logged at debug level instead of warning level.
AUTO_INJECTED_PARAMS: FrozenSet[str] = frozenset({"random_state"})

#: Reserved key under which a thin-wrapper component accepts vendor-specific
#: keyword arguments (policy: developer/api_upgrade/08_naming_decisions.md).
#: The mapping is part of the component's ``spec.params`` whenever non-empty,
#: so passthrough parameters are covered by the spec fingerprint.
ESTIMATOR_PARAMS_KEY: str = "estimator_params"


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


def get_accepted_params(estimator_cls: Callable[..., Any]) -> Tuple[FrozenSet[str], bool]:
    """
    Inspect a vendor callable and return the keywords it accepts.

    Args:
        estimator_cls: Estimator class to inspect, e.g.
            ``sklearn.neighbors.KNeighborsClassifier``, or a plain vendor
            function such as ``skimage.segmentation.slic``. For classes the
            ``__init__`` signature is inspected; for functions the call
            signature itself.

    Returns:
        Tuple[FrozenSet[str], bool]: The explicitly named keyword parameters,
        and whether the constructor also declares ``**kwargs``. When it does,
        any keyword is forwardable and the name set is only informational
        (``XGBClassifier`` is the practical example).
    """
    try:
        source = estimator_cls.__init__ if inspect.isclass(estimator_cls) else estimator_cls
        signature = inspect.signature(source)
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


def validate_estimator_params(
    estimator_params: Optional[Mapping[str, Any]],
    *,
    declared: Iterable[str] = (),
    fixed: Iterable[str] = (),
    owner: str,
) -> Dict[str, Any]:
    """
    Validate vendor keyword arguments passed through a thin wrapper.

    Three kinds of keys are rejected at construction time, because each of
    them has exactly one authoritative source elsewhere and allowing a second
    one would create silent ambiguity:

    - keys colliding with a ``declared`` constructor parameter of the wrapper
      (they must be set directly so they stay validated, documented and
      visible to GUI forms);
    - keys the wrapper passes to the vendor call itself (``fixed``), whose
      override would silently break the component contract (e.g. the
      ``start_label`` of a supervoxel partition);
    - HABIT-injected names (:data:`AUTO_INJECTED_PARAMS`): seeds belong to
      ``set_random_state`` / the spec-level ``random_seed``, never to a
      constructor parameter (v1.0 naming decisions).

    Args:
        estimator_params: User-supplied vendor kwargs, or ``None``.
        declared: Names of the wrapper's declared constructor parameters.
        fixed: Names the wrapper passes to the vendor call itself.
        owner: Dotted component name used in error messages
            (``"<domain>.<name>"``).

    Returns:
        Dict[str, Any]: Plain dict copy of ``estimator_params`` (empty when
        ``None``).

    Raises:
        HABITAPIError: On a non-mapping value, a non-string key, or any
        colliding key.
    """
    if estimator_params is None:
        return {}
    if not isinstance(estimator_params, Mapping):
        raise HABITAPIError(
            f"[{owner}] {ESTIMATOR_PARAMS_KEY} must be a mapping of vendor "
            f"keyword arguments; got {type(estimator_params).__name__}."
        )
    declared_set = set(declared)
    fixed_set = set(fixed)
    problems: List[str] = []
    for key in estimator_params:
        if not isinstance(key, str):
            problems.append(f"non-string key {key!r}")
        elif key in declared_set:
            problems.append(
                f"{key!r} is a declared parameter of {owner}; pass it "
                f"directly, not via {ESTIMATOR_PARAMS_KEY}"
            )
        elif key in fixed_set:
            problems.append(
                f"{key!r} is fixed by {owner} and cannot be overridden"
            )
        elif key in AUTO_INJECTED_PARAMS:
            problems.append(
                f"{key!r} is injected by HABIT through set_random_state / "
                "the spec-level random_seed and must not be set here"
            )
    if problems:
        raise HABITAPIError(
            f"[{owner}] invalid {ESTIMATOR_PARAMS_KEY}: " + "; ".join(problems)
        )
    return dict(estimator_params)


def check_passthrough_accepted(
    vendor_callable: Callable[..., Any],
    estimator_params: Mapping[str, Any],
    *,
    owner: str,
) -> None:
    """
    Verify passthrough kwargs against the vendor callable's signature.

    Passthrough keys are part of the component's spec fingerprint, so they
    must actually reach the vendor callable: an unacceptable key is an error
    here, never a warning-and-drop -- otherwise the fingerprint would claim a
    configuration the run did not use.

    Args:
        vendor_callable: Estimator class or function about to be called.
        estimator_params: Vendor kwargs to splat into the call.
        owner: Dotted component name used in error messages.

    Raises:
        HABITAPIError: When the callable does not declare ``**kwargs`` and at
            least one key is absent from its signature.
    """
    if not estimator_params:
        return
    accepted, accepts_var_keyword = get_accepted_params(vendor_callable)
    if accepts_var_keyword:
        return
    rejected = [key for key in estimator_params if key not in accepted]
    if rejected:
        vendor_name = getattr(vendor_callable, "__name__", type(vendor_callable).__name__)
        raise HABITAPIError(
            f"[{owner}] {ESTIMATOR_PARAMS_KEY} key(s) not accepted by "
            f"{vendor_name}: {_describe_rejected(rejected, accepted)}."
        )


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
