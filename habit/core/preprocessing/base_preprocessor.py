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