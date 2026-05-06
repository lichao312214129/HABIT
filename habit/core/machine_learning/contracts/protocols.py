"""
Structural contracts shared across the machine-learning subpackage.

The :class:`WorkflowResult` protocol defines the *minimum* shape that any
runner output must satisfy in order to be consumed by reporting components
(``ModelStore``, ``ReportWriter``, ``PlotComposer``).  Encoding the shape as
a :class:`typing.Protocol` (rather than a base dataclass) lets concrete
result objects stay frozen dataclasses with their own fields while still
plugging into the same reporting seam.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from .plan import WorkflowPlan


@runtime_checkable
class WorkflowResult(Protocol):
    """
    Minimum surface every workflow result must expose.

    Reporting components only depend on this protocol, not on a particular
    dataclass.  This keeps holdout / k-fold / inference results
    interchangeable from the writers' point of view.
    """

    plan: WorkflowPlan
    summary_rows: List[Dict[str, Any]]
    created_at: str

    def to_legacy_results(self) -> Dict[str, Any]:
        """Expose a dict-style payload compatible with legacy reporting code."""
        ...
