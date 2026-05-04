"""
Runner context shared between workflow and execution layers.

A :class:`RunnerContext` bundles the *minimum* set of collaborators a runner
needs in order to execute a training/evaluation loop.  Bundling them here
removes the previous reverse dependency where ``BaseRunner`` reached back
into ``self.workflow`` for ``data_manager``, ``pipeline_builder``, the
logger and the protected ``_train_with_optional_sampling`` helper.

With an explicit context object:

* the runner depends on a small, named contract (``RunnerContext``) instead
  of ``Any``;
* runners are independently constructible/testable: tests can build a
  context with stubs without instantiating a full workflow;
* dependency direction stays one-way: ``workflow -> context -> runner``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..data_manager import DataManager
from ..pipeline_utils import PipelineBuilder
from ..resampling import Resampler


@dataclass
class RunnerContext:
    """
    Bundle of collaborators injected into a runner.

    Attributes
    ----------
    data_manager:
        Source of features/labels and split logic.
    pipeline_builder:
        Builder that produces a fresh sklearn pipeline per model.
    resampler:
        Optional resampling adapter.  Pass a :class:`Resampler` even when no
        resampling is configured; it will pass the data through untouched.
    logger:
        Logger used for status messages.
    config:
        Validated ML configuration object (``MLConfig``).  Stored as ``Any``
        to avoid a circular import; runners only read from it.
    """

    data_manager: DataManager
    pipeline_builder: PipelineBuilder
    resampler: Resampler
    logger: logging.Logger
    config: Any
