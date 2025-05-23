from typing import Dict, Any, Optional, Union, List, Tuple, Sequence
import numpy as np
import SimpleITK as sitk
import ants
from habit.utils.image_converter import ImageConverter
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("registration")
class RegistrationPreprocessor(BasePreprocessor):
    """Register images to a reference image using ANTs.
    
    This preprocessor performs image registration using ANTs (Advanced Normalization Tools).
    It supports various registration methods including SyN, Rigid, Affine, etc.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        fixed_image: str,
        mask_keys: Optional[Union[str, List[str]]] = None,
        type_of_transform: str = "SyN",
        metric: str = "MI",
        optimizer: str = "gradient_descent",
        use_mask: bool = False,
        allow_missing_keys: bool = False,
        replace_by_fixed_image_mask: bool = True,
        **kwargs
    ):
        """Initialize the registration preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be registered.
            fixed_image (str): Key of the reference image to register to.
            mask_keys (Optional[Union[str, List[str]]]): Keys of the masks to use for registration.
            type_of_transform (str): Type of transform to use. Options: 
                - "Rigid": Rigid transformation
                - "Affine": Affine transformation
                - "SyN": Symmetric normalization (deformable)
                - "SyNRA": SyN + Rigid + Affine
                - "SyNOnly": SyN without initial rigid/affine
                - "TRSAA": Translation + Rotation + Scaling + Affine
                - "Elastic": Elastic transformation
                - "SyNCC": SyN with cross-correlation metric
                - "SyNabp": SyN with mutual information metric
                - "SyNBold": SyN optimized for BOLD images
                - "SyNBoldAff": SyN + Affine for BOLD images
                - "SyNAggro": SyN with aggressive optimization
                - "TVMSQ": Time-varying diffeomorphism with mean square metric
            metric (str): Similarity metric to use. Options:
                - "CC": Cross-correlation
                - "MI": Mutual information
                - "MeanSquares": Mean squares
                - "Demons": Demons metric
            optimizer (str): Optimizer to use. Options:
                - "gradient_descent": Gradient descent
                - "lbfgsb": L-BFGS-B
                - "amoeba": Amoeba
            use_mask (bool): If True, use mask for registration.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            replace_by_fixed_image_mask (bool): If True, use fixed image's mask to replace moving image's mask after registration.
            **kwargs: Additional parameters for registration.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.fixed_image = fixed_image
        
        # Handle mask keys
        if mask_keys is None:
            self.mask_keys = None
        else:
            self.mask_keys = [mask_keys] if isinstance(mask_keys, str) else mask_keys
        
        # Set registration parameters
        self.type_of_transform = type_of_transform
        self.metric = metric
        self.optimizer = optimizer
        self.use_mask = use_mask
        self.replace_by_fixed_image_mask = replace_by_fixed_image_mask
        
        # Store additional parameters
        self.reg_params = kwargs
        
    def _register_image(self,
                       fixed_image: ants.ANTsImage,
                       moving_image: ants.ANTsImage,
                       fixed_mask: Optional[ants.ANTsImage] = None,
                       moving_mask: Optional[ants.ANTsImage] = None) -> Tuple[ants.ANTsImage, List[str]]:
        """Register a moving image to a fixed image using ANTs.
        
        Args:
            fixed_image (ants.ANTsImage): Reference image
            moving_image (ants.ANTsImage): Image to be registered
            fixed_mask (Optional[ants.ANTsImage]): Mask for reference image
            moving_mask (Optional[ants.ANTsImage]): Mask for moving image
            
        Returns:
            Tuple[ants.ANTsImage, List[str]]: 
                - Registered image
                - List of transform files
        """
        
        # Prepare registration parameters
        reg_params = {
            'metric': self.metric,
            'optimizer': self.optimizer,
            **self.reg_params
        }
        
        # Add masks if provided
        if fixed_mask is not None:
            reg_params['mask'] = fixed_mask
        if moving_mask is not None:
            reg_params['moving_mask'] = moving_mask
            
        # Perform registration
        reg_result = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform=self.type_of_transform,
            **reg_params
        )
        
        return reg_result['warpedmovout'], reg_result['fwdtransforms']
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Register the specified images to the reference image.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing ANTs image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with registered images.
        """
        print(f"Registering images to {self.fixed_image}...")
        self._check_keys(data)
        
        # Get reference image
        if self.fixed_image not in data:
            raise KeyError(f"Reference key {self.fixed_image} not found in data dictionary")
        
        fixed_image = data[self.fixed_image]
        
        # to float32 for sitk
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

        # 将SimpleITK图像转换为ANTs图像
        fixed_image = ImageConverter.itk_2_ants(fixed_image)
        
        # Get reference mask if specified
        fixed_mask = None
        if self.use_mask:
            mask_key = f"mask_{self.fixed_image}"
            if mask_key in data:
                fixed_mask = data[mask_key]
                fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)
                fixed_mask = ImageConverter.itk_2_ants(fixed_mask)
        
        # Process each image
        for key in self.keys:
            if key == self.fixed_image:
                continue

            # Get fixed image
            fixed_image = data[self.fixed_image]
            fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
            fixed_image = ImageConverter.itk_2_ants(fixed_image)

            # Get moving image
            moving_image = data[key]
            moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
            moving_image = ImageConverter.itk_2_ants(moving_image)
                
            # Get moving mask if specified
            moving_mask = None
            if self.use_mask:
                mask_key = f"mask_{key}"
                if mask_key in data:
                    moving_mask = data[mask_key]
                    moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)
                    moving_mask = ImageConverter.itk_2_ants(moving_mask)
            
            try:
                # Register image
                registered_image, transform_files = self._register_image(
                    fixed_image, moving_image, fixed_mask, moving_mask
                )
                
                # Convert ANTs image to SimpleITK image
                registered_sitk = ImageConverter.ants_2_itk(registered_image)
                # sitk.GetArrayFromImage(registered_sitk)
                
                # Store the registered image
                data[key] = registered_sitk
                
                # Store the transform files
                transform_key = f"{key}_transform_files"
                data[transform_key] = transform_files
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["registered"] = True
                data[meta_key]["fixed_image"] = self.fixed_image
                data[meta_key]["type_of_transform"] = self.type_of_transform
                data[meta_key]["metric"] = self.metric
                data[meta_key]["optimizer"] = self.optimizer
                # Update the image path to indicate it's registered
                data[meta_key]["image_path"] = data[meta_key]["image_path"].replace(".nii.gz", "_registered.nii.gz")
                
            except Exception as e:
                print(f"Error registering image {key}: {e}")
                if not self.allow_missing_keys:
                    raise
        
        # ============================
        # Process each mask image
        for key in self.keys:
            if key == self.fixed_image:
                continue
            
            mask_key = f"mask_{key}"
            fixed_mask_key = f"mask_{self.fixed_image}"
            transform_key = f"{key}_transform_files"
            
            # Skip if no mask for moving image
            if mask_key not in data:
                continue
                
            # Skip if no mask for fixed image and replace option is enabled
            if self.replace_by_fixed_image_mask and fixed_mask_key not in data:
                print(f"Warning: Cannot replace mask for {key} because fixed mask {fixed_mask_key} not found.")
                continue
                
            # If user chose to replace moving mask with fixed mask
            if self.replace_by_fixed_image_mask:
                print(f"Replacing mask for {key} with fixed image mask as requested.")
                # Get the fixed mask and make a copy for the moving image
                fixed_mask = data[fixed_mask_key]
                data[mask_key] = sitk.Cast(fixed_mask, sitk.sitkUInt8)
                
                # Update metadata
                meta_key = f"{mask_key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["registered"] = True
                data[meta_key]["fixed_image"] = self.fixed_image
                data[meta_key]["replaced_by_fixed_mask"] = True
                continue
                
            # Normal mask registration process
            # Skip if no transform files (which means image registration failed)
            if transform_key not in data:
                print(f"Warning: No transform files found for {key}. Skipping mask registration.")
                continue
                
            # Get fixed image for transform application
            fixed_image = data[self.fixed_image]
            fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
            fixed_image = ImageConverter.itk_2_ants(fixed_image)
                
            # Get the mask image and convert to ANTs
            moving_mask = data[mask_key]
            moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)
            moving_mask_ants = ImageConverter.itk_2_ants(moving_mask)
            
            # Get the transform files from previous registration
            transform_files = data[transform_key]
            
            try:
                # Apply the transform to the mask
                transformed_mask = ants.apply_transforms(
                    fixed=fixed_image,
                    moving=moving_mask_ants,
                    transformlist=transform_files,
                    interpolator="nearestNeighbor"  # Use nearest neighbor for masks
                )
                
                # Convert back to SimpleITK
                transformed_mask_sitk = ImageConverter.ants_2_itk(transformed_mask)
                
                # Ensure it's binary (uint8)
                transformed_mask_sitk = sitk.Cast(transformed_mask_sitk, sitk.sitkUInt8)
                
                # Store the transformed mask
                data[mask_key] = transformed_mask_sitk
                
                # Update metadata
                meta_key = f"{mask_key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["registered"] = True
                data[meta_key]["fixed_image"] = self.fixed_image
                data[meta_key]["type_of_transform"] = self.type_of_transform
                data[meta_key]["metric"] = self.metric
                data[meta_key]["optimizer"] = self.optimizer
                
            except Exception as e:
                print(f"Error applying transform to mask {mask_key}: {e}")
                # Continue even if error occurs for one mask
        
        print(f"Registration completed for {self.keys}.")
