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
"""Run-scoped persistence and figures, declared as Python objects.

This package is the API counterpart of a habitat study's presentation:
construct a :class:`Report`, pass it to
:meth:`~habit.recipes.study.Study.fit_predict`, and each completed subject
is persisted and drawn before the next one accumulates in memory.

It is deliberately not part of :class:`~habit.spec.specs.HabitatSpec`.
"""

from __future__ import annotations

from habit.report.api import (
    FIGURE_LAYOUTS,
    PERSIST_HABITAT_MAP,
    PERSIST_KINDS,
    PERSIST_SUBJECT_MODEL,
    RETAIN_MODES,
    FigureAtom,
    Report,
    SubjectContext,
    coerce_report,
)
from habit.report.figures import (
    ClusterValidation,
    GraphNetwork2D,
    GraphSlice,
    ITH,
    MSI,
    Overlay,
    VolumeFractions,
)

__all__ = [
    "PERSIST_HABITAT_MAP",
    "PERSIST_SUBJECT_MODEL",
    "PERSIST_KINDS",
    "RETAIN_MODES",
    "FIGURE_LAYOUTS",
    "FigureAtom",
    "Report",
    "SubjectContext",
    "coerce_report",
    "Overlay",
    "VolumeFractions",
    "MSI",
    "ITH",
    "ClusterValidation",
    "GraphSlice",
    "GraphNetwork2D",
]
