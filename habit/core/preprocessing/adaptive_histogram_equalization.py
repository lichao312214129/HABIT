from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import SimpleITK as sitk
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.progress_utils import CustomTqdm
from ...utils.log_utils import get_module_logger

# Get module logger
logger = get_module_logger(__name__)


@PreprocessorFactory.register("adaptive_histogram_equalization")
class AdaptiveHistogramEqualization(BasePreprocessor):
    """Apply adaptive histogram equalization to images using SimpleITK.
    
    This preprocessor uses SimpleITK's AdaptiveHistogramEqualizationImageFilter
    to perform contrast-limited adaptive histogram equalization (CLAHE).
    It enhances local contrast while limiting noise amplification.
    
    The filter divides the image into regions (controlled by radius parameter)
    and performs histogram equalization in each region, with bilinear interpolation
    to eliminate boundary artifacts.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        alpha: float = 0.3,
        beta: float = 0.3,
        radius: Union[int, Tuple[int, int, int]] = 5,
        allow_missing_keys: bool = False,
    ):
        """Initialize the adaptive histogram equalization preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be processed.
            alpha (float): Controls how much the filter acts like the classical 
                histogram equalization method. Range [0, 1].
                - alpha=0: Unsharp mask (edge enhancement)
                - alpha=1: Classical histogram equalization
                Default is 0.3.
            beta (float): Controls how much the filter acts like an unsharp mask.
                Range [0, 1].
                - beta=0: No window adaptation (global equalization)
                - beta=1: Full window adaptation (local equalization)
                Default is 0.3.
            radius (Union[int, Tuple[int, int, int]]): Radius of the local region
                in pixels for each dimension. Larger values result in smoother
                output but less local contrast enhancement.
                Can be a single int (same for all dimensions) or a tuple (x, y, z).
                Default is 5.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Validate and set alpha parameter
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in range [0, 1], got {alpha}")
        self.alpha = alpha
        
        # Validate and set beta parameter
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in range [0, 1], got {beta}")
        self.beta = beta
        
        # Set radius parameter
        if isinstance(radius, int):
            self.radius = (radius, radius, radius)
        else:
            self.radius = tuple(radius)
            
    def _apply_adaptive_histogram_equalization(
        self, 
        input_image: sitk.Image,
        subj: Optional[str] = None,
        key: Optional[str] = None
    ) -> sitk.Image:
        """Apply adaptive histogram equalization to a SimpleITK image.
        
        Args:
            input_image (sitk.Image): Input SimpleITK image to be processed.
            subj (Optional[str]): Subject identifier for logging.
            key (Optional[str]): Image key for logging.
            
        Returns:
            sitk.Image: Processed SimpleITK image with enhanced local contrast.
        """
        subj_info = f"[{subj}] " if subj else ""
        key_info = f"({key}) " if key else ""
        
        logger.debug(f"{subj_info}{key_info}AHE (α={self.alpha}, β={self.beta}, r={self.radius})")
        
        # Cast image to float32 for processing
        input_image = sitk.Cast(input_image, sitk.sitkFloat32)
        
        # Create adaptive histogram equalization filter
        ahe_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
        
        # Set filter parameters
        ahe_filter.SetAlpha(self.alpha)
        ahe_filter.SetBeta(self.beta)
        ahe_filter.SetRadius(self.radius)
        
        # Apply filter
        output_image = ahe_filter.Execute(input_image)
        
        
        return output_image
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive histogram equalization to the specified images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with processed images.
        """
        self._check_keys(data)
        
        subj = data.get('subj', 'unknown')
        logger.debug(f"[{subj}] Adaptive histogram equalization")
        
        # Initialize progress bar
        progress_bar = CustomTqdm(
            total=len(self.keys), 
            desc=f"[{subj}] Adaptive Histogram Equalization"
        )
        
        # Process each image
        for key in self.keys:
            # Get SimpleITK image from data
            if key not in data:
                if not self.allow_missing_keys:
                    raise KeyError(f"[{subj}] Key {key} not found in data dictionary")
                logger.warning(f"[{subj}] Key {key} not found, skipping")
                progress_bar.update()
                continue
                
            input_image = data[key]
            
            # Ensure we have a SimpleITK image object
            if not isinstance(input_image, sitk.Image):
                logger.warning(f"[{subj}] Warning: {key} is not a SimpleITK Image object. Skipping.")
                progress_bar.update()
                continue
            
            try:
                # Apply adaptive histogram equalization
                processed_image = self._apply_adaptive_histogram_equalization(
                    input_image, subj, key
                )
                
                # Store the processed image
                data[key] = processed_image
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["adaptive_histogram_equalization"] = True
                data[meta_key]["ahe_alpha"] = self.alpha
                data[meta_key]["ahe_beta"] = self.beta
                data[meta_key]["ahe_radius"] = self.radius
                
                
            except Exception as e:
                logger.error(f"[{subj}] Error applying adaptive histogram equalization to {key}: {e}")
                if not self.allow_missing_keys:
                    raise
            
            progress_bar.update()
        
        return data
