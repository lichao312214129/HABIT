from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import SimpleITK as sitk
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.progress_utils import CustomTqdm
from ...utils.log_utils import get_module_logger

# Get module logger
logger = get_module_logger(__name__)


@PreprocessorFactory.register("histogram_standardization")
class HistogramStandardization(BasePreprocessor):
    """Apply Nyúl histogram standardization to images.
    
    This preprocessor implements the Nyúl histogram standardization method,
    which maps image intensities to a standard scale using percentile landmarks.
    Unlike histogram matching, this method does not require a reference image,
    but instead maps intensities to predefined target values.
    
    Reference:
        Nyúl, L.G., Udupa, J.K., Zhang, X., 2000. New variants of a method of 
        MRI scale standardization. IEEE Trans. Med. Imaging 19, 143-150.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        percentiles: Optional[List[float]] = None,
        target_min: float = 0.0,
        target_max: float = 100.0,
        mask_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ):
        """Initialize the histogram standardization preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be standardized.
            percentiles (Optional[List[float]]): Percentile landmarks for standardization.
                Default is [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99].
            target_min (float): Target minimum value for standardization. Default is 0.0.
            target_max (float): Target maximum value for standardization. Default is 100.0.
            mask_key (Optional[str]): Key of the mask image to use for computing percentiles.
                If None, uses all non-zero voxels. Default is None.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Set default percentiles if not provided
        # These are the standard landmarks used in Nyúl method
        if percentiles is None:
            self.percentiles = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
        else:
            self.percentiles = sorted(percentiles)
        
        # Target range for standardization
        self.target_min = target_min
        self.target_max = target_max
        
        # Optional mask key
        self.mask_key = mask_key
        
        # Compute target landmarks (linearly spaced between target_min and target_max)
        self._compute_target_landmarks()
        
    def _compute_target_landmarks(self) -> None:
        """Compute target landmark values based on percentiles and target range.
        
        The target landmarks are linearly interpolated between target_min and target_max
        based on the percentile values.
        """
        # Map percentiles to target range
        # percentile 0 -> target_min, percentile 100 -> target_max
        self.target_landmarks = [
            self.target_min + (p / 100.0) * (self.target_max - self.target_min)
            for p in self.percentiles
        ]
        logger.debug(f"Target landmarks: {self.target_landmarks}")
        
    def _compute_percentile_landmarks(
        self, 
        image_array: np.ndarray,
        mask_array: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute percentile landmarks from an image.
        
        Args:
            image_array (np.ndarray): Input image array.
            mask_array (Optional[np.ndarray]): Optional mask array. If provided,
                only voxels within the mask are used. If None, all non-zero voxels are used.
                
        Returns:
            np.ndarray: Array of intensity values at each percentile landmark.
        """
        # Get voxels to use for percentile computation
        if mask_array is not None:
            # Use voxels within mask
            voxels = image_array[mask_array > 0]
        else:
            # Use all non-zero voxels (common approach for medical images)
            voxels = image_array[image_array > 0]
        
        if len(voxels) == 0:
            logger.warning("No valid voxels found for percentile computation, using all voxels")
            voxels = image_array.flatten()
        
        # Compute percentile values
        landmarks = np.percentile(voxels, self.percentiles)
        
        return landmarks
    
    def _apply_piecewise_linear_mapping(
        self,
        image_array: np.ndarray,
        source_landmarks: np.ndarray,
        target_landmarks: List[float]
    ) -> np.ndarray:
        """Apply piecewise linear mapping to transform image intensities.
        
        This function maps intensity values from source landmarks to target landmarks
        using piecewise linear interpolation.
        
        Args:
            image_array (np.ndarray): Input image array to be transformed.
            source_landmarks (np.ndarray): Intensity values at percentile landmarks in source image.
            target_landmarks (List[float]): Target intensity values for each landmark.
            
        Returns:
            np.ndarray: Transformed image array with standardized intensities.
        """
        # Create output array
        output_array = np.zeros_like(image_array, dtype=np.float32)
        
        # Handle values below the minimum landmark
        mask_below = image_array <= source_landmarks[0]
        if source_landmarks[0] != 0:
            # Linear extrapolation for values below minimum
            scale = target_landmarks[0] / source_landmarks[0] if source_landmarks[0] != 0 else 1.0
            output_array[mask_below] = image_array[mask_below] * scale
        else:
            output_array[mask_below] = image_array[mask_below]
        
        # Apply piecewise linear mapping between landmarks
        for i in range(len(source_landmarks) - 1):
            src_low = source_landmarks[i]
            src_high = source_landmarks[i + 1]
            tgt_low = target_landmarks[i]
            tgt_high = target_landmarks[i + 1]
            
            # Create mask for this segment
            mask = (image_array > src_low) & (image_array <= src_high)
            
            # Linear interpolation within this segment
            if src_high - src_low > 0:
                # Compute slope and intercept for linear mapping
                slope = (tgt_high - tgt_low) / (src_high - src_low)
                intercept = tgt_low - slope * src_low
                output_array[mask] = image_array[mask] * slope + intercept
            else:
                # If source range is zero, map to target midpoint
                output_array[mask] = (tgt_low + tgt_high) / 2.0
        
        # Handle values above the maximum landmark
        mask_above = image_array > source_landmarks[-1]
        if np.any(mask_above):
            # Linear extrapolation for values above maximum
            src_range = source_landmarks[-1] - source_landmarks[-2]
            tgt_range = target_landmarks[-1] - target_landmarks[-2]
            if src_range > 0:
                slope = tgt_range / src_range
                intercept = target_landmarks[-1] - slope * source_landmarks[-1]
                output_array[mask_above] = image_array[mask_above] * slope + intercept
            else:
                output_array[mask_above] = target_landmarks[-1]
        
        return output_array
        
    def _apply_histogram_standardization(
        self, 
        input_image: sitk.Image,
        mask_image: Optional[sitk.Image] = None,
        subj: Optional[str] = None,
        key: Optional[str] = None
    ) -> sitk.Image:
        """Apply Nyúl histogram standardization to a SimpleITK image.
        
        Args:
            input_image (sitk.Image): Input SimpleITK image to be standardized.
            mask_image (Optional[sitk.Image]): Optional mask image for percentile computation.
            subj (Optional[str]): Subject identifier for logging.
            key (Optional[str]): Image key for logging.
            
        Returns:
            sitk.Image: Histogram-standardized SimpleITK image.
        """
        subj_info = f"[{subj}] " if subj else ""
        key_info = f"({key}) " if key else ""
        
        logger.info(f"{subj_info}{key_info}Applying Nyul histogram standardization")
        
        # Convert to numpy array
        image_array = sitk.GetArrayFromImage(input_image).astype(np.float32)
        
        # Get mask array if provided
        mask_array = None
        if mask_image is not None:
            mask_array = sitk.GetArrayFromImage(mask_image)
        
        # Compute source landmarks from input image
        source_landmarks = self._compute_percentile_landmarks(image_array, mask_array)
        logger.debug(f"{subj_info}{key_info}Source landmarks: {source_landmarks}")
        
        # Apply piecewise linear mapping
        standardized_array = self._apply_piecewise_linear_mapping(
            image_array, source_landmarks, self.target_landmarks
        )
        
        # Convert back to SimpleITK image
        standardized_image = sitk.GetImageFromArray(standardized_array)
        
        # Copy spatial information from original image
        standardized_image.SetOrigin(input_image.GetOrigin())
        standardized_image.SetSpacing(input_image.GetSpacing())
        standardized_image.SetDirection(input_image.GetDirection())
        
        logger.info(f"{subj_info}{key_info}Histogram standardization completed "
                   f"(target range: [{self.target_min}, {self.target_max}])")
        
        return standardized_image
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply histogram standardization to the specified images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with histogram-standardized images.
        """
        self._check_keys(data)
        
        subj = data.get('subj', 'unknown')
        logger.info(f"Processing subject: {subj}")
        
        # Get mask image if specified
        mask_image = None
        if self.mask_key is not None:
            if self.mask_key in data:
                mask_image = data[self.mask_key]
                if not isinstance(mask_image, sitk.Image):
                    logger.warning(f"[{subj}] Mask key {self.mask_key} is not a SimpleITK Image. "
                                 "Proceeding without mask.")
                    mask_image = None
                else:
                    logger.info(f"[{subj}] Using {self.mask_key} as mask for histogram standardization")
            else:
                logger.warning(f"[{subj}] Mask key {self.mask_key} not found. Proceeding without mask.")
        
        # Initialize progress bar
        progress_bar = CustomTqdm(total=len(self.keys), desc=f"[{subj}] Histogram Standardization")
        
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
                # Apply histogram standardization
                standardized_image = self._apply_histogram_standardization(
                    input_image, mask_image, subj, key
                )
                
                # Store the standardized image
                data[key] = standardized_image
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["histogram_standardized"] = True
                data[meta_key]["standardization_method"] = "nyul"
                data[meta_key]["percentiles"] = self.percentiles
                data[meta_key]["target_range"] = [self.target_min, self.target_max]
                
                logger.info(f"[{subj}] Successfully standardized image {key}")
                
            except Exception as e:
                logger.error(f"[{subj}] Error applying histogram standardization to {key}: {e}")
                if not self.allow_missing_keys:
                    raise
            
            progress_bar.update()
        
        return data
