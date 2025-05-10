#!/usr/bin/env python
"""
Intratumoral Heterogeneity (ITH) Score Calculation
This module provides functionality for calculating ITH scores from habitat maps
based on the methodology described in literature for quantifying tumor heterogeneity.
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, Union
import logging

class ITHFeatureExtractor:
    """Extractor class for Intratumoral Heterogeneity (ITH) scores"""
    
    def __init__(self):
        """
        Initialize ITH feature extractor
        """
        pass
    
    def extract_ith_features(self, habitat_img: Union[str, sitk.Image]) -> Dict:
        """
        Calculate ITH score directly from a habitat image
        
        Args:
            habitat_img: SimpleITK image or path to habitat map file
            
        Returns:
            Dict: Dictionary containing ITH score and related features
        """
        try:
            # Load image if path is provided
            if isinstance(habitat_img, str):
                habitat_img = sitk.ReadImage(habitat_img)
            elif not isinstance(habitat_img, sitk.Image):
                raise ValueError("habitat_img must be a SimpleITK image or a file path.")
            
            # Calculate total area using SimpleITK
            binary_mask = sitk.BinaryThreshold(habitat_img, 1, 100000, 1, 0)  # large upper bound
            stat_filter = sitk.StatisticsImageFilter()
            stat_filter.Execute(binary_mask)
            total_area = stat_filter.GetSum()
            
            if total_area == 0:
                return {"ith_score": 0.0, "error": "Empty habitat map"}
            
            # Get unique habitats using SimpleITK
            label_stats = sitk.LabelIntensityStatisticsImageFilter()
            mask = sitk.BinaryThreshold(habitat_img, 1, 100000, 1, 0)  # large upper bound
            label_stats.Execute(habitat_img, habitat_img)  # Using the image itself as both inputs
            
            # Get all labels (habitats) excluding background
            habitats = [label for label in label_stats.GetLabels() if label > 0]
            
            if not habitats:
                return {"ith_score": 0.0, "error": "No habitats found"}
            
            summation = 0.0
            habitat_stats = {}
            
            for habitat in habitats:
                # Create binary mask for this habitat using SimpleITK
                threshold_filter = sitk.BinaryThresholdImageFilter()
                threshold_filter.SetLowerThreshold(habitat)
                threshold_filter.SetUpperThreshold(habitat)
                threshold_filter.SetInsideValue(1)
                threshold_filter.SetOutsideValue(0)
                habitat_mask_sitk = threshold_filter.Execute(habitat_img)
                
                # Label connected components using SimpleITK
                connected_components = sitk.ConnectedComponent(habitat_mask_sitk)
                
                # Get stats of connected components
                label_stats = sitk.LabelShapeStatisticsImageFilter()
                label_stats.Execute(connected_components)
                
                # Get number of regions (labels)
                num_regions = label_stats.GetNumberOfLabels()
                
                # Skip if no regions found
                if num_regions == 0:
                    habitat_stats[f'habitat_{habitat}_regions'] = 0
                    habitat_stats[f'habitat_{habitat}_largest_area'] = 0
                    habitat_stats[f'habitat_{habitat}_area_ratio'] = 0
                    continue
                
                # Find largest region area
                largest_area = 0
                for label in label_stats.GetLabels():
                    area = label_stats.GetNumberOfPixels(label)
                    if area > largest_area:
                        largest_area = area
                
                # Add to summation for ITH score
                summation += largest_area / num_regions
                
                # Store habitat statistics
                habitat_stats[f'habitat_{habitat}_regions'] = num_regions
                habitat_stats[f'habitat_{habitat}_largest_area'] = int(largest_area)
                habitat_stats[f'habitat_{habitat}_area_ratio'] = largest_area / num_regions
            
            # Calculate ITH score: 1 - (1/S_total) * Σ(S_i,max / n_i)
            ith_score = 1.0 - (1.0 / total_area) * summation
            
            # Prepare result dictionary
            result = {
                'ith_score': ith_score,
                'num_habitats': len(habitats),
                'total_area': int(total_area)
            }
            
            # Add habitat statistics
            result.update(habitat_stats)
            
            return result
            
        except Exception as e:
            logging.error(f"Error calculating ITH score from habitat image: {str(e)}")
            return {"error": str(e), "ith_score": 0.0}