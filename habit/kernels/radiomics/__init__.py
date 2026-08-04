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
"""L0 radiomics kernels: batched per-supervoxel feature extraction.

Describing thousands of supervoxels with full PyRadiomics feature vectors is
the numerical heart of habitat analysis, and the most expensive part of a run.
This subpackage holds that machinery -- the union-mask binning, the batched
texture-matrix path (native C extension, with a PyRadiomics fallback), and the
optional TorchRadiomics GPU backend -- as arrays-in, table-out functions.

Living at L0 is what lets the v0.1 extractors and the v1.0 domain operators
call the same code, so a supervoxel feature table cannot depend on which layer
requested it. Imports here stay inside :mod:`habit.kernels`.

Submodules
----------
``supervoxel_batch``
    Batched extraction entry points.
``settings``
    Merging habit's supervoxel keys into a PyRadiomics settings mapping.
``cext``
    Native batched texture matrices, with a pure-PyRadiomics fallback.
``torchradiomics``
    Torch reimplementations of the PyRadiomics feature classes.

The heavy submodules are imported on demand rather than eagerly, so importing
:mod:`habit.kernels` does not pull in SimpleITK, PyRadiomics or torch.
"""

from __future__ import annotations

__all__ = [
    "extract_batched_supervoxel_features",
    "extract_supervoxel_features_pyradiomics",
    "extract_batched_supervoxel_firstorder",
    "merge_supervoxel_settings",
    "SUPERVOXEL_SETTING_KEYS",
]

_LAZY_EXPORTS = {
    "extract_batched_supervoxel_features": "habit.kernels.radiomics.supervoxel_batch",
    "extract_supervoxel_features_pyradiomics": "habit.kernels.radiomics.supervoxel_batch",
    "extract_batched_supervoxel_firstorder": "habit.kernels.radiomics.supervoxel_batch",
    "merge_supervoxel_settings": "habit.kernels.radiomics.settings",
    "SUPERVOXEL_SETTING_KEYS": "habit.kernels.radiomics.settings",
}


def __getattr__(name: str):
    """
    Resolve a public name from its submodule on first access.

    Args:
        name: Attribute requested on this package.

    Returns:
        The requested object.

    Raises:
        AttributeError: If ``name`` is not a public export.
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list:
    """Return the public exports for tab completion."""
    return sorted(__all__)
