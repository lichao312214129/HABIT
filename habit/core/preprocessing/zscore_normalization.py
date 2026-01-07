from typing import Dict, Any, Optional, Union, List, Tuple
import SimpleITK as sitk
import numpy as np
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.log_utils import get_module_logger

# Get module logger
logger = get_module_logger(__name__)

@PreprocessorFactory.register("zscore_normalization")
class ZScoreNormalization(BasePreprocessor):
    """Apply Z-score normalization to medical images.
    
    This preprocessor normalizes image intensities by subtracting the mean and 
    dividing by the standard deviation, resulting in a distribution with
    zero mean and unit variance.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        only_inmask: bool = False,
        mask_key: Optional[str] = None,
        clip_values: Optional[Tuple[float, float]] = None,
        allow_missing_keys: bool = False,
        **kwargs
    ):
        """Initialize the Z-score normalization preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be normalized.
            only_inmask (bool): If True, only calculate statistics within the mask.
            mask_key (Optional[str]): Key of the mask to use when only_inmask is True.
            clip_values (Optional[Tuple[float, float]]): Optional tuple of (min, max) values to clip 
                normalized results. Useful to prevent extreme values, e.g. (-3, 3).
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional parameters.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Handle mask settings
        self.only_inmask = only_inmask
        self.mask_key = mask_key
            
        # Clipping values
        self.clip_values = clip_values
    
    def _apply_zscore_normalization(
        self, 
        sitk_image: sitk.Image,
        sitk_mask: Optional[sitk.Image] = None,
        subj: Optional[str] = None
    ) -> sitk.Image:
        """Apply Z-score normalization to a SimpleITK image.
        
        Args:
            sitk_image (sitk.Image): Input SimpleITK image to normalize
            sitk_mask (Optional[sitk.Image]): Optional mask for computing statistics
            subj (Optional[str]): Subject identifier for logging
            
        Returns:
            sitk.Image: Normalized SimpleITK image
        """
        # Calculate statistics using SimpleITK
        stats_filter = sitk.StatisticsImageFilter()
        
        # If mask is provided, use it for statistics calculation
        if sitk_mask is not None:
            # Create a version of the image with mask applied
            masked_image = sitk.Mask(sitk_image, sitk_mask)
            stats_filter.Execute(masked_image)
        else:
            stats_filter.Execute(sitk_image)
        
        # Get mean and standard deviation
        mean_val = stats_filter.GetMean()
        std_val = stats_filter.GetSigma()  # Use GetSigma() instead of GetStandardDeviation()
        
        subj_info = f"[{subj}] " if subj else ""
        logger.info(f"{subj_info}Calculated mean: {mean_val}, std: {std_val}")
        
        # Avoid division by zero or very small values
        if std_val < 1e-10:
            logger.warning(f"{subj_info}Warning: Standard deviation is very small ({std_val}). Using std=1 to avoid division issues.")
            std_val = 1.0
        
        # Create mean image (same size as input, all pixels = mean value)
        mean_image = sitk.Image(sitk_image.GetSize(), sitk_image.GetPixelID())
        mean_image.CopyInformation(sitk_image)  # Copy metadata
        mean_image = sitk.Add(mean_image, mean_val)  # Fill with mean value
        
        # Subtract mean (step 1 of z-score)
        centered_image = sitk.Subtract(sitk_image, mean_image)
        
        # Divide by standard deviation (step 2 of z-score)
        normalized_image = sitk.Divide(centered_image, std_val)
        
        # Get sample values for logging
        sample_array = sitk.GetArrayFromImage(normalized_image)
        logger.info(f"{subj_info}Normalized image min: {np.min(sample_array)}, max: {np.max(sample_array)}")
        
        # Clip values if specified
        if self.clip_values is not None:
            # Create threshold filter
            threshold_filter = sitk.ClampImageFilter()
            threshold_filter.SetLowerBound(self.clip_values[0])
            threshold_filter.SetUpperBound(self.clip_values[1])
            normalized_image = threshold_filter.Execute(normalized_image)
        
        return normalized_image
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Z-score normalization to the specified images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with normalized images.
        """
        logger.info("Applying Z-score normalization...")
        self._check_keys(data)

        subj = data.get('subj', 'unknown')
        logger.info(f"Processing subject: {subj}")
        
        # Process each image
        for key in self.keys:
            # Get SimpleITK image from data
            sitk_image = data[key]
            
            # Ensure we have a SimpleITK image object
            if not isinstance(sitk_image, sitk.Image):
                logger.warning(f"[{subj}] Warning: {key} is not a SimpleITK Image object. Skipping.")
                continue
            
            # Get mask if specified and only_inmask is True
            sitk_mask = None
            if self.only_inmask and self.mask_key is not None:
                if self.mask_key in data:
                    sitk_mask = data[self.mask_key]
                    if not isinstance(sitk_mask, sitk.Image):
                        logger.warning(f"[{subj}] Warning: {self.mask_key} is not a SimpleITK Image object. Using no mask.")
                        sitk_mask = None
            
            try:
                # Apply Z-score normalization
                normalized_image = self._apply_zscore_normalization(sitk_image, sitk_mask, subj)
                
                # Check normalized image
                normalized_array = sitk.GetArrayFromImage(normalized_image)
                logger.info(f"[{subj}] Final normalized image pixel type: {normalized_image.GetPixelID()}")
                logger.info(f"[{subj}] Final array min: {np.min(normalized_array)}, max: {np.max(normalized_array)}")
                
                # Calculate one sample manually for verification
                orig_array = sitk.GetArrayFromImage(sitk_image)
                # Find point with maximum deviation from mean for clearer verification
                sample_idx = np.unravel_index(np.argmax(np.abs(orig_array - np.mean(orig_array))), orig_array.shape)
                
                # Convert to physical point for comparison
                # (SimpleITK and NumPy index order is different)
                # NumPy array is [z,y,x] while SimpleITK is [x,y,z]
                sitk_idx = sample_idx[::-1]  # Reverse the indices
                
                logger.info(f"[{subj}] Sample voxel at array index {sample_idx}:")
                logger.info(f"[{subj}]   Original value: {orig_array[sample_idx]}")
                logger.info(f"[{subj}]   Mean: {np.mean(orig_array)}, Std: {np.std(orig_array)}")
                logger.info(f"[{subj}]   Manual z-score: {(orig_array[sample_idx] - np.mean(orig_array)) / np.std(orig_array)}")
                logger.info(f"[{subj}]   Normalized value: {normalized_array[sample_idx]}")

                # Store the normalized image
                data[key] = normalized_image
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["zscore_normalized"] = True
                
            except Exception as e:
                logger.error(f"[{subj}] Error applying Z-score normalization to {key}: {e}")
                if not self.allow_missing_keys:
                    raise
        
        return data 