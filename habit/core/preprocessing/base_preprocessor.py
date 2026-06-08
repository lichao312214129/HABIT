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
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List

class BasePreprocessor(ABC):
    """Base class for all image preprocessors in HABIT.
    
    This class defines the basic interface that all preprocessors must implement.
    """
    
    def __init__(self, keys: Union[str, List[str]], allow_missing_keys: bool = False):
        """Initialize the preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the corresponding items to be transformed.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
        """
        self.keys = [keys] if isinstance(keys, str) else keys
        self.allow_missing_keys = allow_missing_keys

    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing image and metadata.
            
        Returns:
            Dict[str, Any]: Processed data dictionary.
        """
        pass

    def _check_keys(self, data: Dict[str, Any]) -> None:
        """Check if all required keys are present in the input data.
        
        Args:
            data (Dict[str, Any]): Input data dictionary.
            
        Raises:
            KeyError: If a required key is missing and allow_missing_keys is False.
        """
        for key in self.keys:
            if key not in data and not self.allow_missing_keys:
                raise KeyError(f"Key {key} not found in data dictionary") 