from typing import Dict, Any, Union, List, Optional, Sequence, Tuple
import torch
import numpy as np
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("custom_preprocessor")
class CustomPreprocessor(BasePreprocessor):
    """Template for creating custom preprocessors.
    
    This class serves as a template for users to create their own preprocessors.
    Users should:
    1. Copy this file and rename it
    2. Change the class name and registration name
    3. Implement the __call__ method with their custom preprocessing logic
    4. Add any necessary helper methods for preprocessing
    
    Example:
        @PreprocessorFactory.register("my_preprocessor")
        class MyPreprocessor(BasePreprocessor):
            def __init__(self, keys, param1, param2, **kwargs):
                super().__init__(keys=keys)
                self.param1 = param1
                self.param2 = param2
                
            def __call__(self, data):
                self._check_keys(data)
                for key in self.keys:
                    data[key] = self._process_image(data[key])
                return data
    """
    
    def __init__(self, keys: Union[str, List[str]], allow_missing_keys: bool = False, **kwargs: Any):
        """Initialize the custom preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the corresponding items to be transformed.
                If a single string is provided, it will be converted to a list.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional arguments specific to the custom preprocessor.
                These should be documented in the class docstring.
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
        """Process the input data with custom preprocessing logic.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing image and metadata.
                The values for keys should be in the expected format (e.g., SimpleITK Image objects,
                numpy arrays, or torch tensors).
            
        Returns:
            Dict[str, Any]: Processed data dictionary with the same structure as input.
            
        Example:
            # 1. Check if required keys are present
            self._check_keys(data)
            
            # 2. Process each key
            for key in self.keys:
                # Get the data
                item = data[key]
                
                # Process the data
                processed_item = self._process_item(item)
                
                # Update the data dictionary
                data[key] = processed_item
                
                # Update metadata if necessary
                meta_key = f"{key}_meta_dict"
                if meta_key in data:
                    data[meta_key]["processed"] = True
                    data[meta_key]["processor"] = self.__class__.__name__
                
            return data
        """
        # Implement your custom preprocessing logic here
        pass
        
    def _process_item(self, item: Any) -> Any:
        """Process a single item with custom preprocessing logic.
        
        Args:
            item (Any): Input item to be processed. The type should be documented
                based on your specific use case (e.g., torch.Tensor, np.ndarray, sitk.Image).
            
        Returns:
            Any: Processed item in the same format as input.
            
        Example:
            # Add your preprocessing logic here
            # Example for image processing:
            # 1. Convert to appropriate format if needed
            # 2. Apply transformations
            # 3. Return processed item
            return item
        """
        # Add your preprocessing logic here
        return item 