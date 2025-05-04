from typing import Dict, Any, Union, List
import torch
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
    """
    
    def __init__(self, keys: Union[str, List[str]], allow_missing_keys: bool = False, **kwargs: Any):
        """Initialize the custom preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the corresponding items to be transformed.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional arguments specific to the custom preprocessor.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        # Add any additional initialization here
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data with custom preprocessing logic.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing image and metadata.
            
        Returns:
            Dict[str, Any]: Processed data dictionary.
            
        Example:
            # 1. Check if required keys are present
            self._check_keys(data)
            
            # 2. Get the image data
            image = data[self.keys[0]]
            
            # 3. Apply preprocessing
            processed_image = self._custom_preprocessing(image)
            
            # 4. Update the data dictionary
            data[self.keys[0]] = processed_image
            
            # 5. Update metadata if necessary
            if f"{self.keys[0]}_meta_dict" in data:
                data[f"{self.keys[0]}_meta_dict"]["processed"] = True
                
            return data
        """
        # Implement your custom preprocessing logic here
        pass
        
    def _custom_preprocessing(self, image: torch.Tensor) -> torch.Tensor:
        """Implement your custom preprocessing logic here.
        
        Args:
            image (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Processed image tensor.
        """
        # Add your preprocessing logic here
        return image 