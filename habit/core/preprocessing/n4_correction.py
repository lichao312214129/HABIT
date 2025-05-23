from typing import Dict, Any, Optional, Union, List, Tuple, Sequence
import numpy as np
import SimpleITK as sitk
import logging
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.image_converter import ImageConverter

# 配置日志记录器
logger = logging.getLogger(__name__)

@PreprocessorFactory.register("n4_correction")
class N4BiasFieldCorrection(BasePreprocessor):
    """Apply N4 bias field correction to images using SimpleITK.
    
    This preprocessor applies N4 bias field correction to correct for intensity
    inhomogeneity in medical images.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        mask_keys: Optional[Union[str, List[str]]] = None,
        num_fitting_levels: int = 4,
        num_iterations: List[int] = None,
        convergence_threshold: float = 0.001,
        shrink_factor: int = 4,
        allow_missing_keys: bool = False,
        **kwargs
    ):
        """Initialize the N4 bias field correction preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be corrected.
            mask_keys (Optional[Union[str, List[str]]]): Keys of the masks to use for correction.
                If None, no mask will be used.
            num_fitting_levels (int): Number of fitting levels for the bias field correction.
            num_iterations (List[int]): Number of iterations at each fitting level.
                If None, will use [50] * num_fitting_levels.
            convergence_threshold (float): Convergence threshold for the correction.
            shrink_factor (int): Shrink factor to accelerate computation (default 4).
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Additional parameters for N4 correction.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Handle mask keys
        if mask_keys is None:
            self.mask_keys = None
        else:
            self.mask_keys = [mask_keys] if isinstance(mask_keys, str) else mask_keys
        
        # Set N4 parameters
        self.num_fitting_levels = num_fitting_levels
        self.num_iterations = num_iterations if num_iterations is not None else [50] * num_fitting_levels
        self.convergence_threshold = convergence_threshold
        self.shrink_factor = shrink_factor
        
    def _apply_n4_correction(self, 
                           sitk_image: sitk.Image,
                           sitk_mask: Optional[sitk.Image] = None,
                           subj: Optional[str] = None) -> sitk.Image:
        """Apply N4 bias field correction to a SimpleITK image.
        
        Args:
            sitk_image (sitk.Image): Input SimpleITK image to correct
            sitk_mask (Optional[sitk.Image]): Optional mask for the correction
            subj (Optional[str]): Subject identifier for logging
            
        Returns:
            sitk.Image: Corrected SimpleITK image
        """
        subj_info = f"[{subj}] " if subj else ""
        
        # Cast image to float32
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        
        # Create original image copy for full resolution correction
        original_image = sitk_image
        
        # Apply shrinking to speed up computation if shrink_factor > 1
        if self.shrink_factor > 1:
            logger.info(f"{subj_info}Applying shrinking with factor {self.shrink_factor} to speed up computation")
            sitk_image = sitk.Shrink(original_image, [self.shrink_factor] * original_image.GetDimension())
            if sitk_mask is not None:
                sitk_mask = sitk.Shrink(sitk_mask, [self.shrink_factor] * original_image.GetDimension())
        
        # Create and configure N4 corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(self.num_iterations)
        corrector.SetConvergenceThreshold(self.convergence_threshold)
        
        logger.info(f"{subj_info}Executing N4 correction with {self.num_fitting_levels} fitting levels")
        
        # Execute the correction
        if sitk_mask is not None:
            corrected_image = corrector.Execute(sitk_image, sitk_mask)
        else:
            corrected_image = corrector.Execute(sitk_image)
        
        # If we used shrinking, apply correction to the full resolution image
        if self.shrink_factor > 1:
            logger.info(f"{subj_info}Applying correction to full resolution image")
            # Get the log bias field and apply to full resolution image
            log_bias_field = corrector.GetLogBiasFieldAsImage(original_image)
            corrected_image = original_image / sitk.Exp(log_bias_field)
        
        return corrected_image
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply N4 bias field correction to the specified images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with corrected images.
        """
        logger.info("Applying N4 bias field correction...")
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
            
            # Get corresponding mask if specified
            sitk_mask = None
            if self.mask_keys is not None:
                mask_key = f"mask_{key}"
                if mask_key in data:
                    sitk_mask = data[mask_key]
                    if not isinstance(sitk_mask, sitk.Image):
                        logger.warning(f"[{subj}] Warning: {mask_key} is not a SimpleITK Image object. Using no mask.")
                        sitk_mask = None
            
            try:
                # Apply N4 correction
                corrected_image = self._apply_n4_correction(sitk_image, sitk_mask, subj)
                
                # Store the corrected image
                data[key] = corrected_image
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["n4_corrected"] = True
                
                logger.info(f"[{subj}] Successfully applied N4 correction to {key}")
                
            except Exception as e:
                logger.error(f"[{subj}] Error applying N4 correction to {key}: {e}")
                if not self.allow_missing_keys:
                    raise
        
        return data 