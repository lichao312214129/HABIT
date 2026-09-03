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
Deprecation helpers for HABIT's public API.

HABIT follows semantic versioning: within a major release line, public
symbols are never removed without a deprecation period. These helpers give
every deprecation the same shape -- a dedicated warning category (so
downstream projects can filter or escalate HABIT deprecations specifically)
and a message that always names the version the symbol was deprecated in and
the version it will be removed in.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, TypeVar

__all__ = [
    "HabitDeprecationWarning",
    "HabitPendingDeprecationWarning",
    "deprecated",
]


class HabitDeprecationWarning(DeprecationWarning):
    """A public HABIT API is deprecated and scheduled for removal."""


class HabitPendingDeprecationWarning(PendingDeprecationWarning):
    """A public HABIT API will be deprecated in a future release."""


def build_deprecation_message(
    subject: str,
    since: str,
    *,
    alternative: str = "",
    removed_in: str = "",
) -> str:
    """
    Compose the canonical HABIT deprecation message.

    Args:
        subject: Human-readable name of the deprecated symbol, e.g.
            ``"habit.recipes.Study.fit_predict"``.
        since: Version in which the deprecation started, e.g. ``"1.0.0"``.
        alternative: Replacement symbol to migrate to, if one exists.
        removed_in: Version in which the symbol will be removed; an empty
            string means the removal release is not scheduled yet.

    Returns:
        A message naming both the deprecation and the removal version, plus
        the migration target when known.
    """
    message = f"{subject} is deprecated since version {since}"
    if removed_in:
        message += f" and will be removed in version {removed_in}"
    else:
        message += " and will be removed in a future release"
    message += "."
    if alternative:
        message += f" Use {alternative} instead."
    return message


_F = TypeVar("_F", bound=Callable[..., Any])


def deprecated(
    since: str,
    *,
    alternative: str = "",
    removed_in: str = "",
) -> Callable[[_F], _F]:
    """
    Decorate a function or class so every use emits ``HabitDeprecationWarning``.

    Classes keep their identity (the decorator wraps ``__init__`` in place
    rather than wrapping the class object) so ``isinstance`` checks,
    subclassing, and pickling are unaffected.

    Args:
        since: Version in which the deprecation started, e.g. ``"1.0.0"``.
        alternative: Replacement symbol to migrate to, if one exists.
        removed_in: Version in which the symbol will be removed.

    Returns:
        A decorator applying the warning to the decorated callable.
    """

    def _decorate(obj: _F) -> _F:
        message = build_deprecation_message(
            getattr(obj, "__qualname__", repr(obj)),
            since,
            alternative=alternative,
            removed_in=removed_in,
        )

        if inspect.isclass(obj):
            # Capture the original initializer BEFORE rebinding it on the
            # class; referencing ``obj.__init__`` inside the wrapper would
            # resolve to the wrapper itself and recurse without bound.
            original_init = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)  # type: ignore[arg-type]
            def _warned_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(message, HabitDeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            obj.__init__ = _warned_init  # type: ignore[assignment, method-assign]
            return obj

        @functools.wraps(obj)
        def _warned_call(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, HabitDeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        return _warned_call  # type: ignore[return-value]

    return _decorate
