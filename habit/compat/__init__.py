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
"""Ecosystem interop adapters (optional integrations).

Each submodule bridges HABIT to one third-party ecosystem:

- :mod:`habit.compat.sklearn` -- domain protocols and table-ML components as
  genuine ``sklearn.base.BaseEstimator`` adapters, usable inside
  ``sklearn.pipeline.Pipeline`` / ``GridSearchCV``.
- :mod:`habit.compat.monai` -- ``Subject`` <-> MONAI-style dict conversion so
  HABIT subject-level operators slot into ``monai.transforms.Compose`` and
  torch ``DataLoader`` pipelines. Works without MONAI installed.
- :mod:`habit.compat.nnunet` -- :class:`NnUNetDataSource` reading nnU-Net raw
  datasets (``imagesTr`` / ``labelsTr`` + ``dataset.json``) directly.

Submodules load lazily on first attribute access so that
``import habit.compat`` itself stays free of heavy third-party imports.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["sklearn", "monai", "nnunet"]


def __getattr__(name: str) -> Any:
    """Resolve a compat submodule on first access, keeping imports lazy."""
    if name in __all__:
        module = importlib.import_module(f"habit.compat.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
