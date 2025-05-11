#!/usr/bin/env python
"""
Basic Habitat Features Extraction
This module provides functionality for extracting basic features from habitat maps:
1. Number of disconnected regions for each habitat
2. Volume percentage for each habitat
"""

import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict

class BasicFeatureExtractor:
    """Extractor class for basic habitat features"""
    
    @staticmethod
    def get_non_radiomics_features(habitat_img):
        """
        Calculate number of disconnected regions and volume ratio for each habitat
        
        Args:
            habitat_img: SimpleITK image or path to habitat map file
            
        Returns:
            Dict: Dictionary containing basic features for each habitat
        """
        try:
            if isinstance(habitat_img, str):
                habitat_img = sitk.ReadImage(habitat_img)
            elif not isinstance(habitat_img, sitk.Image):
                raise ValueError("habitat_img must be a SimpleITK image or a file path.")

            results = {}
            
            # Calculate total volume of the habitat map
            stats_filter = sitk.StatisticsImageFilter()
            stats_filter.Execute(habitat_img != 0)
            total_voxels = int(stats_filter.GetSum())

            label_filter = sitk.LabelStatisticsImageFilter()
            label_filter.Execute(habitat_img, habitat_img)
            labels = label_filter.GetLabels()
            labels = [int(label) for label in labels if label != 0]
            
            for label in labels:
                try:
                    binary_img = sitk.BinaryThreshold(habitat_img, lowerThreshold=label, upperThreshold=label)
                    
                    stats_filter.Execute(binary_img)
                    habitat_voxels = int(stats_filter.GetSum())
                    volume_ratio = habitat_voxels / total_voxels if total_voxels > 0 else 0.0
                    
                    cc_filter = sitk.ConnectedComponentImageFilter()
                    cc_filter.SetFullyConnected(False)
                    labeled_img = cc_filter.Execute(binary_img)
                    num_regions = cc_filter.GetObjectCount()
                    
                    results[label] = {
                        'num_regions': num_regions,
                        'volume_ratio': volume_ratio
                    }
                except Exception as e:
                    logging.error(f"Error processing habitat label {label}: {str(e)}")
                    results[label] = {
                        'num_regions': 0,
                        'volume_ratio': 0.0,
                        'error': str(e)
                    }
                    
            results['num_habitats'] = len(labels)
            
            return results
        except Exception as e:
            logging.error(f"Error calculating basic habitat features: {str(e)}")
            return {"error": str(e), "num_habitats": 0} 