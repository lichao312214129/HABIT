#!/usr/bin/env python
"""
Habitat Radiomics Features Extraction
This module provides functionality for extracting habitat-specific radiomics features:
1. Radiomic features of raw images within different habitats
2. Radiomic features of habitats within the entire ROI
"""

import logging
import SimpleITK as sitk
import numpy as np
from typing import Dict, Optional
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

# Lazy import radiomics to avoid scipy.fft compatibility issues at startup
# Only import when actually needed
def _get_featureextractor():
    """Lazy import radiomics featureextractor to avoid startup errors"""
    try:
        from radiomics import featureextractor
        return featureextractor
    except ImportError as e:
        logger.error(f"Failed to import radiomics: {e}")
        raise ImportError(
            "PyRadiomics is required for radiomics feature extraction. "
            "Please install it with: pip install pyradiomics"
        ) from e

class HabitatRadiomicsExtractor:
    """Extractor class for habitat radiomics features"""
    
    @staticmethod
    def extract_radiomics_features_for_whole_habitat(habitat_img, params_file):
        """
        Extract radiomics features from the whole habitat map within the ROI
        
        Args:
            habitat_img: SimpleITK image or path to habitat map file
            params_file: Parameter file for PyRadiomics feature extraction
            
        Returns:
            Dict: Dictionary containing extracted radiomics features
        """
        try:
            featureextractor = _get_featureextractor()
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
            
            if isinstance(habitat_img, str):
                habitat_img = sitk.ReadImage(habitat_img)
            elif not isinstance(habitat_img, sitk.Image):
                raise ValueError("habitat_img must be a SimpleITK image or a file path.")

            label_filter = sitk.LabelStatisticsImageFilter()
            label_filter.Execute(habitat_img, habitat_img)
            labels = label_filter.GetLabels()
            labels = [int(label) for label in labels if label != 0]

            # Make a binary image
            habitat_img_binary = sitk.BinaryThreshold(
                habitat_img, 
                lowerThreshold=1, 
                upperThreshold=np.max(labels).astype(np.double), 
                insideValue=1, 
                outsideValue=0
            )

            return extractor.execute(
                imageFilepath=habitat_img,
                maskFilepath=habitat_img_binary, 
                label=1
            )
        except Exception as e:
            logger.error(f"Error extracting radiomics features from whole habitat: {str(e)}")
            return {"error": f"Feature extraction error: {str(e)}"}
    
    @staticmethod
    def extract_radiomics_features_from_each_habitat(habitat_path, image_path, subject_id, params_file):
        """
        Extract radiomics features from original images within each habitat
        
        Args:
            habitat_path: Path to habitat map file
            image_path: Path to original image file
            subject_id: Subject identifier
            params_file: Parameter file for PyRadiomics feature extraction
            
        Returns:
            Dict: Dictionary containing extracted radiomics features for each habitat
        """
        featureextractor = _get_featureextractor()
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
        
        try:
            habitat_img = sitk.ReadImage(habitat_path)
            raw_img = sitk.ReadImage(image_path)

            # Check and adjust image metadata for consistency
            if raw_img.GetDirection() != habitat_img.GetDirection():
                logger.info(f"Raw and mask direction is different: {subject_id}")
                habitat_img.SetDirection(raw_img.GetDirection())

            if raw_img.GetOrigin() != habitat_img.GetOrigin():
                logger.info(f"Raw and mask origin is different: {subject_id}")
                habitat_img.SetOrigin(raw_img.GetOrigin())

            if raw_img.GetSpacing() != habitat_img.GetSpacing():
                logger.info(f"Raw and mask spacing is different: {subject_id}")
                habitat_img.SetSpacing(raw_img.GetSpacing())

            label = sitk.LabelStatisticsImageFilter()
            label.Execute(habitat_img, habitat_img)
            labels = label.GetLabels()
            labels = [int(label) for label in labels if label != 0]
        except Exception as e:
            logger.error(f"Error preparing habitat image data: {str(e)}")
            return {}

        feature_vector = {}
        for label in labels:
            try:
                feature_vector[label] = extractor.execute(
                    imageFilepath=raw_img, 
                    maskFilepath=habitat_img, 
                    label=label
                )
            except Exception as e:
                logger.error(f"Error extracting radiomics features for habitat {label}: {str(e)}")
                feature_vector[label] = {}
               
        return feature_vector
    
    @staticmethod
    def extract_tranditional_radiomics(image_path, habitat_path, subject_id, params_file):
        """
        Extract traditional radiomics features from original images
        
        Args:
            image_path: Path to original image file
            habitat_path: Path to habitat map file
            subject_id: Subject identifier
            params_file: Parameter file for PyRadiomics feature extraction
            
        Returns:
            Dict: Dictionary containing extracted traditional radiomics features
        """
        try:
            featureextractor = _get_featureextractor()
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
            
            habitat_img = sitk.ReadImage(habitat_path)
            raw_img = sitk.ReadImage(image_path)

            # Check and adjust image metadata for consistency
            if raw_img.GetDirection() != habitat_img.GetDirection():
                logger.info(f"Raw and mask direction is different: {subject_id}")
                habitat_img.SetDirection(raw_img.GetDirection())

            if raw_img.GetOrigin() != habitat_img.GetOrigin():
                logger.info(f"Raw and mask origin is different: {subject_id}")
                habitat_img.SetOrigin(raw_img.GetOrigin())

            if raw_img.GetSpacing() != habitat_img.GetSpacing():
                logger.info(f"Raw and mask spacing is different: {subject_id}")
                habitat_img.SetSpacing(raw_img.GetSpacing())

            label_filter = sitk.LabelStatisticsImageFilter()
            label_filter.Execute(habitat_img, habitat_img)
            labels = label_filter.GetLabels()
            labels = [int(label) for label in labels if label != 0]

            mask = sitk.BinaryThreshold(
                habitat_img, 
                lowerThreshold=1, 
                upperThreshold=np.max(labels).astype(np.double), 
                insideValue=1, 
                outsideValue=0
            )

            return extractor.execute(
                imageFilepath=raw_img,
                maskFilepath=mask, 
                label=1
            )
        except Exception as e:
            logger.error(f"Error extracting traditional radiomics features: {str(e)}")
            return {"error": f"Feature extraction error: {str(e)}"} 