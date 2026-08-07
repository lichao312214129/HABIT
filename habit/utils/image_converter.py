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
"""Image format conversion helpers with optional ANTs / PyTorch backends.

ANTs (antspyx) and torch are optional. They must not be imported at module
import time: on Windows, a broken or version-mismatched torch native DLL can
raise ``OSError`` (e.g. WinError 1114 loading ``c10.dll``) rather than
``ImportError``. Preprocessing paths that only need SimpleITK must keep working
when torch is missing or unloadable.

Lazy loaders also avoid a fixed module-level ``import ants`` then ``import
torch`` order, which has been observed to trigger that DLL failure even when
torch alone imports successfully.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from habit.exceptions import OptionalDependencyError

# Cached optional backends. ``None`` after a failed probe means unavailable.
_ants_module: Optional[ModuleType] = None
_ants_probed: bool = False
_torch_module: Optional[ModuleType] = None
_torch_probed: bool = False


def _probe_optional_import(module_name: str) -> Optional[ModuleType]:
    """Import an optional dependency, treating native DLL failures as missing.

    Args:
        module_name: Fully-qualified module name to import (e.g. ``"torch"``).

    Returns:
        The imported module, or ``None`` if import failed for any expected
        reason (missing package, broken wheel, Windows DLL load error, etc.).
    """
    try:
        module = __import__(module_name)
    except Exception:
        # ImportError: package not installed.
        # OSError / WindowsError: native DLL load failures (WinError 1114, 126, ...).
        # Other Exception subclasses: rare init failures that must not crash
        # unrelated callers that only import this module.
        return None
    return module


def _get_ants() -> Optional[ModuleType]:
    """Lazily import ANTs (antspyx) on first use.

    Returns:
        The ``ants`` module if importable, otherwise ``None``.
    """
    global _ants_module, _ants_probed
    if not _ants_probed:
        _ants_probed = True
        _ants_module = _probe_optional_import("ants")
    return _ants_module


def _get_torch() -> Optional[ModuleType]:
    """Lazily import PyTorch on first use.

    Returns:
        The ``torch`` module if importable, otherwise ``None``.
    """
    global _torch_module, _torch_probed
    if not _torch_probed:
        _torch_probed = True
        _torch_module = _probe_optional_import("torch")
    return _torch_module


def is_ants_available() -> bool:
    """Return whether ANTs can be imported in this process.

    Returns:
        ``True`` when ``import ants`` succeeds (lazy probe).
    """
    return _get_ants() is not None


def is_torch_available() -> bool:
    """Return whether PyTorch can be imported in this process.

    Returns:
        ``True`` when ``import torch`` succeeds (lazy probe).
    """
    return _get_torch() is not None


def __getattr__(name: str) -> bool:
    """Lazy module attributes for backward-compatible availability flags.

    ``ANTS_AVAILABLE`` / ``TORCH_AVAILABLE`` resolve on first access so that
    ``import habit.utils.image_converter`` alone never loads ants or torch.
    Prefer ``is_ants_available()`` / ``is_torch_available()`` in new code.
    """
    if name == "ANTS_AVAILABLE":
        return is_ants_available()
    if name == "TORCH_AVAILABLE":
        return is_torch_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ImageConverter:
    """Utility class for converting between different image formats."""

    @staticmethod
    def get_metadata(meta_dict: Dict[str, Any], ndim: int) -> Tuple[tuple, tuple, tuple]:
        """Extract and validate metadata from dictionary.

        Args:
            meta_dict: Metadata dictionary.
            ndim: Number of dimensions.

        Returns:
            Validated spacing, origin, and direction as tuples.
        """
        # Default values
        default_spacing = tuple([1.0] * ndim)
        default_origin = tuple([0.0] * ndim)
        default_direction = tuple(
            [1.0 if i == j else 0.0 for i in range(ndim) for j in range(ndim)]
        )

        # Get metadata with defaults
        spacing = meta_dict.get("spacing", default_spacing)
        origin = meta_dict.get("origin", default_origin)
        direction = meta_dict.get("direction", default_direction)

        # Convert to tuples if necessary
        if not isinstance(spacing, tuple):
            spacing = tuple(spacing[:ndim])
        if not isinstance(origin, tuple):
            origin = tuple(origin[:ndim])
        if not isinstance(direction, tuple):
            direction = tuple(direction)

        # Validate direction matrix size
        direction_size = ndim * ndim
        if len(direction) != direction_size:
            direction = default_direction

        return spacing, origin, direction

    @staticmethod
    def tensor_to_numpy(tensor: Any) -> np.ndarray:
        """Convert torch tensor to numpy array.

        Args:
            tensor: Input tensor in format [C,Z,Y,X] or [C,H,W].

        Returns:
            Numpy array with channel dimension removed if single channel.

        Raises:
            OptionalDependencyError: If torch is not installed or cannot load.
        """
        torch = _get_torch()
        if torch is None:
            raise OptionalDependencyError(
                "tensor_to_numpy requires the optional torch dependency; "
                'install with pip install "habitat-analysis[torch]" to use it.'
            )
        array = tensor.cpu().numpy()
        if array.shape[0] == 1:  # If single channel
            array = array.squeeze(0)  # Remove channel dimension
        return array

    @staticmethod
    def numpy_to_tensor(
        array: np.ndarray,
        dtype: Any = None,
        device: Any = None,
    ) -> Any:
        """Convert numpy array to torch tensor.

        Args:
            array: Input array in [Z,Y,X] format.
            dtype: Target tensor dtype (requires torch).
            device: Target tensor device (requires torch).

        Returns:
            Torch tensor with added channel dimension [1,Z,Y,X].

        Raises:
            OptionalDependencyError: If torch is not installed or cannot load.
        """
        torch = _get_torch()
        if torch is None:
            raise OptionalDependencyError(
                "numpy_to_tensor requires the optional torch dependency; "
                'install with pip install "habitat-analysis[torch]" to use it.'
            )
        if array.ndim == 2:
            array = array[np.newaxis, ...]  # Add channel dim for 2D
        elif array.ndim == 3:
            array = array[np.newaxis, ...]  # Add channel dim for 3D

        tensor = torch.from_numpy(array)
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype, device=device)
        return tensor

    @staticmethod
    def ants_2_itk(image: Any) -> sitk.Image:
        """Convert an ANTs image to a SimpleITK image.

        Args:
            image: ANTs image instance.

        Returns:
            SimpleITK image with matching geometry.

        Raises:
            OptionalDependencyError: If antspyx is not installed or cannot load.
        """
        ants = _get_ants()
        if ants is None:
            raise OptionalDependencyError(
                "ANTs<->ITK conversion requires the optional antspyx dependency; "
                "install 'habitat-analysis[registration]' to use it."
            )
        imageITK = sitk.GetImageFromArray(image.numpy().transpose(2, 1, 0))
        imageITK.SetOrigin(image.origin)
        imageITK.SetSpacing(image.spacing)
        imageITK.SetDirection(image.direction.reshape(9))
        return imageITK

    @staticmethod
    def itk_2_ants(image: sitk.Image) -> Any:
        """Convert a SimpleITK image to an ANTs image.

        Args:
            image: SimpleITK image.

        Returns:
            ANTs image with matching geometry.

        Raises:
            OptionalDependencyError: If antspyx is not installed or cannot load.
        """
        ants = _get_ants()
        if ants is None:
            raise OptionalDependencyError(
                "ITK<->ANTs conversion requires the optional antspyx dependency; "
                "install 'habitat-analysis[registration]' to use it."
            )
        image_ants = ants.from_numpy(
            sitk.GetArrayFromImage(image).transpose(2, 1, 0),
            origin=image.GetOrigin(),
            spacing=image.GetSpacing(),
            direction=np.array(image.GetDirection()).reshape(3, 3),
        )
        return image_ants
