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
from typing import Dict, Any, Union, List

from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

# Optional torch import (only needed if your custom preprocessor uses PyTorch)
# Uncomment the following lines if you need torch:
# try:
#     import torch
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False
#     torch = None


@PreprocessorFactory.register("custom_preprocessor")
class CustomPreprocessor(BasePreprocessor):
    """Template for creating custom preprocessors.

    Copy this file, rename the class, register with ``PreprocessorFactory``,
    and implement ``__call__``. See the customization user guide for a full example.
    """

    def __init__(self, keys: Union[str, List[str]], allow_missing_keys: bool = False, **kwargs: Any):
        """Initialize the custom preprocessor.

        Args:
            keys (Union[str, List[str]]): Keys of items to transform.
            allow_missing_keys (bool): If True, missing keys in ``data`` are allowed.
            **kwargs: Preprocessor-specific parameters (document in your subclass).
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

        # Add any additional initialization here
        # Example:
        # self.param1 = kwargs.pop('param1', default_value)
        # self.param2 = kwargs.pop('param2', default_value)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom preprocessing to ``data``.

        Args:
            data (Dict[str, Any]): Input dict (SimpleITK images, arrays, etc.).

        Returns:
            Dict[str, Any]: Processed dict with the same keys as input.
        """
        # Implement your custom preprocessing logic here
        pass

    def _process_item(self, item: Any) -> Any:
        """Process a single item (image or tensor).

        Args:
            item (Any): One entry from ``data`` (type depends on your pipeline).

        Returns:
            Any: Processed item in the same format as input.
        """
        # Add your preprocessing logic here
        return item
