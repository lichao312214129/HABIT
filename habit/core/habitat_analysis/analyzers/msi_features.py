#!/usr/bin/env python
"""
MSI (Mutual Spatial Integrity) Features Extraction
This module provides functionality for extracting MSI features from habitat maps
"""

import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

class MSIFeatureExtractor:
    """Extractor class for MSI features"""
    
    def __init__(self, voxel_cutoff=10):
        """
        Initialize MSI feature extractor
        
        Args:
            voxel_cutoff: Voxel threshold for filtering small regions in MSI feature calculation
        """
        self.voxel_cutoff = voxel_cutoff
    
    def calculate_MSI_matrix(self, habitat_array: np.ndarray, unique_class: int) -> np.ndarray:
        """
        Calculate the MSI matrix from habitat array
        
        Args:
            habitat_array: Array representing the habitat map
            unique_class: Number of habitat classes (including background)
            
        Returns:
            msi_matrix: Calculated MSI matrix
        """
        # Find the minimum bounding box of non-zero regions in habitat_array
        roi_z, roi_y, roi_x = np.where(habitat_array != 0)
        
        if len(roi_z) == 0:
            logger.warning("No non-zero elements found in habitat array")
            return np.zeros((unique_class, unique_class), dtype=np.int64)
            
        z_min, z_max = np.min(roi_z), np.max(roi_z)
        y_min, y_max = np.min(roi_y), np.max(roi_y)
        x_min, x_max = np.min(roi_x), np.max(roi_x)

        # Extract data within the bounding box
        box_of_VOI = habitat_array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        
        # Add a layer of zeros around the box to capture boundary information
        box_of_VOI = np.pad(box_of_VOI, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)

        # Define 3D neighborhood (face-connected only)
        neighborhood_3d_cube_only = [
            (-1, 0, 0), (1, 0, 0),  # Up and down neighbors
            (0, -1, 0), (0, 1, 0),  # Left and right neighbors
            (0, 0, -1), (0, 0, 1)   # Front and back neighbors
        ]

        # Initialize MSI matrix
        msi_matrix = np.zeros((unique_class, unique_class), dtype=np.int64)
        
        # Traverse the 3D image and count neighbor relationships
        for z in range(box_of_VOI.shape[0]):  
            for y in range(box_of_VOI.shape[1]):  
                for x in range(box_of_VOI.shape[2]): 
                    # Get current voxel value
                    current_voxel_value = box_of_VOI[z, y, x]

                    # Check all neighbors
                    for dz, dy, dx in neighborhood_3d_cube_only:
                        neighbor_z = z + dz
                        neighbor_y = y + dy
                        neighbor_x = x + dx
                        
                        # Check if neighbor is within image bounds
                        if 0 <= neighbor_z < box_of_VOI.shape[0] and \
                        0 <= neighbor_y < box_of_VOI.shape[1] and \
                        0 <= neighbor_x < box_of_VOI.shape[2]:
                            
                            neighbor_voxel_value = box_of_VOI[neighbor_z, neighbor_y, neighbor_x]
                            
                            # Update MSI matrix
                            msi_matrix[current_voxel_value, neighbor_voxel_value] += 1

        return msi_matrix

    def calculate_MSI_features(self, msi_matrix: np.ndarray, name: str) -> Dict:
        """
        Calculate MSI features from the MSI matrix
        
        Args:
            msi_matrix: MSI matrix
            name: Prefix for feature names
            
        Returns:
            Dict: Calculated MSI features
        """
        # Assert that msi_matrix is square and contains no negative values
        assert msi_matrix.shape[0] == msi_matrix.shape[1], f'msi_matrix of {name} is not a square matrix'
        assert np.all(msi_matrix >= 0), f'msi_matrix of {name} has negative value'
        
        # First-order features: Volume of each subregion (diagonal) and borders of two differing subregions (off-diagonal)
        firstorder_feature = {}
        for i in range(0, msi_matrix.shape[0]):
            for j in range(i+1, msi_matrix.shape[0]):
                firstorder_feature['firstorder_{}_and_{}'.format(i, j)] = msi_matrix[i, j]

        # Calculate diagonal elements, excluding background
        for i in range(1, msi_matrix.shape[0]):
            firstorder_feature['firstorder_{}_and_{}'.format(i, i)] = msi_matrix[i, i]

        # Normalized first-order features, denominator includes only the lower triangular part excluding the first element
        denominator_mat = np.tril(msi_matrix, k=0)
        denominator_mat[0] = 0
        denominator = np.sum(denominator_mat)
        
        if denominator == 0:
            logger.warning(f"MSI matrix denominator is 0 for {name}, cannot calculate normalized features")
            normal_msi_matrix = np.zeros_like(msi_matrix, dtype=float)
        else:
            normal_msi_matrix = msi_matrix / denominator
            
        firstorder_feature_normalized = {}
        for i in range(0, normal_msi_matrix.shape[0]):
            for j in range(i+1, normal_msi_matrix.shape[1]):
                firstorder_feature_normalized['firstorder_normalized_{}_and_{}'.format(i, j)] = normal_msi_matrix[i, j]

        for i in range(1, normal_msi_matrix.shape[0]):
            firstorder_feature_normalized['firstorder_normalized_{}_and_{}'.format(i, i)] = normal_msi_matrix[i, i]
        
        # Second-order features based on normalized MSI matrix
        p = normal_msi_matrix.copy()
        
        # Calculate contrast
        i_indices, j_indices = np.indices(p.shape)
        contrast = np.sum((i_indices - j_indices)**2 * p)
        
        # Calculate homogeneity
        homogeneity = np.sum(p / (1.0 + (i_indices - j_indices)**2))
        
        # Calculate correlation
        px = np.sum(p, axis=1)
        py = np.sum(p, axis=0)
        
        ux = np.sum(px * np.arange(len(px)))
        uy = np.sum(py * np.arange(len(py)))
        
        sigmax = np.sqrt(np.sum(px * (np.arange(len(px)) - ux)**2))
        sigmay = np.sqrt(np.sum(py * (np.arange(len(py)) - uy)**2))
        
        if sigmax > 0 and sigmay > 0:
            sum_p_ij = np.sum(p * i_indices * j_indices)
            correlation = (sum_p_ij - ux * uy) / (sigmax * sigmay)
        else:
            correlation = 1.0
        
        # Calculate energy
        energy = np.sum(p**2)
        
        secondorder_feature = { 
            'contrast': contrast,
            'homogeneity': homogeneity,
            'correlation': correlation,
            'energy': energy
        }

        # Combine all features
        msi_feature = {**firstorder_feature, **firstorder_feature_normalized, **secondorder_feature}
        return msi_feature

    def extract_MSI_features(self, habitat_path: str, n_habitats: int, subj: str) -> Dict:
        """
        Extract MSI features from a single habitat map
        
        Args:
            habitat_path: Path to the habitat map file
            n_habitats: Number of habitats
            subj: Subject ID
            
        Returns:
            Dict: Extracted MSI features
        """
        try:
            img = sitk.ReadImage(habitat_path)
            array = sitk.GetArrayFromImage(img)
            
            unique_class = n_habitats+1  # Number of habitats + 1 (including background)

            # Calculate MSI matrix
            msi_matrix = self.calculate_MSI_matrix(array, unique_class)

            # Calculate MSI features
            msi_feature = self.calculate_MSI_features(msi_matrix, subj)
            
            return msi_feature
        except Exception as e:
            logger.error(f"Error extracting MSI features for subject {subj}: {str(e)}")
            return {"error": str(e)} 