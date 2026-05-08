import SimpleITK as sitk
from typing import Dict, Any, Union, List, Optional
import logging

from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

logger = logging.getLogger(__name__)

@PreprocessorFactory.register("reorientation")
class ReorientationPreprocessor(BasePreprocessor):
    """Preprocessor for reorienting images to a target coordinate system (e.g., LPS).
    
    This preprocessor allows adjusting the image orientation to a specific canonical
    direction. It supports two modes:
    1. 'strict': Exact spatial resampling to a perfect canonical grid (uses interpolation).
                 Linear/BSpline for images, Nearest Neighbor for masks.
    2. 'closest': Only flips and permutes axes to get as close to the target
                  orientation as possible without interpolation.
    """
    
    def __init__(self, 
                 keys: Union[str, List[str]], 
                 target_orientation: str = "LPS",
                 mode: str = "closest",
                 is_label: Union[bool, List[bool]] = False,
                 allow_missing_keys: bool = False):
        """Initialize the reorientation preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the corresponding items to be transformed.
            target_orientation (str): Target anatomical orientation (e.g., 'LPS', 'RAS').
            mode (str): 'closest' (no interpolation, just flip/permute) or 'strict' (resampling with interpolation).
            is_label (Union[bool, List[bool]]): Whether the corresponding key is a label/mask image.
                                                Used to determine interpolation method in 'strict' mode.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
        """
        super().__init__(keys, allow_missing_keys)
        self.target_orientation = target_orientation.upper()
        self.mode = mode.lower()
        
        if self.mode not in ["closest", "strict"]:
            raise ValueError("mode must be either 'closest' or 'strict'")
            
        # Ensure is_label is a list of same length as keys
        if isinstance(is_label, bool):
            self.is_label = [is_label] * len(self.keys)
        elif isinstance(is_label, list):
            if len(is_label) != len(self.keys):
                raise ValueError("Length of is_label list must match length of keys list")
            self.is_label = is_label
        else:
            raise TypeError("is_label must be a boolean or a list of booleans")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data by reorienting images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK images.
            
        Returns:
            Dict[str, Any]: Processed data dictionary with reoriented images.
        """
        self._check_keys(data)
        
        for key, is_lbl in zip(self.keys, self.is_label):
            if key not in data:
                continue
                
            image = data[key]
            
            if not isinstance(image, sitk.Image):
                logger.warning(f"Data for key '{key}' is not a SimpleITK Image. Skipping reorientation.")
                continue
                
            if self.mode == "closest":
                # 'closest' mode: only permute and flip, no interpolation
                orient_filter = sitk.DICOMOrientImageFilter()
                orient_filter.SetDesiredCoordinateOrientation(self.target_orientation)
                try:
                    reoriented_image = orient_filter.Execute(image)
                    data[key] = reoriented_image
                    logger.debug(f"Reoriented '{key}' to {self.target_orientation} using 'closest' mode.")
                except Exception as e:
                    logger.error(f"Failed to reorient '{key}' using 'closest' mode: {e}")
                    raise
                    
            elif self.mode == "strict":
                # 'strict' mode: full resampling to orthogonal grid (needs interpolation)
                # First, find the closest orthogonal direction cosine matrix
                orient_filter = sitk.DICOMOrientImageFilter()
                orient_filter.SetDesiredCoordinateOrientation(self.target_orientation)
                
                # We can use the orient_filter on an empty image to get the target direction matrix
                dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
                dummy.SetDirection(image.GetDirection())
                dummy_reoriented = orient_filter.Execute(dummy)
                target_direction = dummy_reoriented.GetDirection()
                
                # Set up the resampler
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(image) # Use original physical space limits
                
                # Override direction to perfectly orthogonal
                resampler.SetOutputDirection(target_direction)
                
                # Determine appropriate interpolation
                if is_lbl:
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler.SetDefaultPixelValue(0)
                else:
                    resampler.SetInterpolator(sitk.sitkLinear) # Or BSpline
                    # Use minimum value of original image for padding
                    f = sitk.MinimumMaximumImageFilter()
                    f.Execute(image)
                    resampler.SetDefaultPixelValue(f.GetMinimum())
                    
                # We need to compute the new origin and size to encompass the whole image
                # This can be complex for arbitrary rotations, but sitk allows automatic bounding box computation
                # Here we use an identity transform
                transform = sitk.Transform()
                resampler.SetTransform(transform)
                
                try:
                    reoriented_image = resampler.Execute(image)
                    data[key] = reoriented_image
                    logger.debug(f"Reoriented '{key}' to {self.target_orientation} using 'strict' mode (interpolated).")
                except Exception as e:
                    logger.error(f"Failed to reorient '{key}' using 'strict' mode: {e}")
                    raise

        return data
