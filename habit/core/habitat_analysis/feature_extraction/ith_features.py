#!/usr/bin/env python
"""
Intratumoral Heterogeneity (ITH) Score Calculation
This module provides functionality for calculating ITH scores from habitat maps
based on the methodology described in literature for quantifying tumor heterogeneity.
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, Tuple, List, Optional, Union
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from radiomics import featureextractor
import os
from scipy import ndimage
import matplotlib.pyplot as plt

class ITHFeatureExtractor:
    """Extractor class for Intratumoral Heterogeneity (ITH) scores"""
    
    def __init__(self, params_file=None, window_size=3, margin_size=3, voxel_cutoff=10):
        """
        Initialize ITH feature extractor
        
        Args:
            params_file: Parameter file for PyRadiomics feature extraction
            window_size: Size of sliding window in mm for local feature extraction
            margin_size: Size of margin in mm to expand tumor region
            voxel_cutoff: Minimum number of voxels for a cluster to be considered
        """
        self.params_file = params_file
        self.window_size = window_size
        self.margin_size = margin_size
        self.voxel_cutoff = voxel_cutoff
        
    def expand_tumor_region(self, mask_img: sitk.Image, margin_mm: int = 3) -> sitk.Image:
        """
        Expand tumor region by specified margin
        
        Args:
            mask_img: Binary mask image of tumor
            margin_mm: Margin size in mm
            
        Returns:
            sitk.Image: Expanded tumor mask
        """
        # Convert margin from mm to voxels based on image spacing
        spacing = mask_img.GetSpacing()
        margin_voxels = [int(margin_mm / s) for s in spacing]
        
        # Create a structuring element for dilation
        se = sitk.BinaryDilateImageFilter()
        se.SetKernelRadius(margin_voxels)
        se.SetForegroundValue(1)
        
        # Dilate the mask
        expanded_mask = se.Execute(mask_img)
        
        return expanded_mask
    
    def extract_local_features(self, image: sitk.Image, mask: sitk.Image, 
                              window_size_mm: int = 3) -> np.ndarray:
        """
        Extract local radiomic features using sliding window
        
        Args:
            image: Original image
            mask: Tumor mask (can be expanded)
            window_size_mm: Window size in mm
            
        Returns:
            np.ndarray: Feature matrix where each row represents features of a voxel
        """
        # Initialize PyRadiomics feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(self.params_file)
        
        # Convert window size from mm to voxels
        spacing = image.GetSpacing()
        window_size_voxels = [int(window_size_mm / s) for s in spacing]
        
        # Get mask array and find foreground voxels
        mask_array = sitk.GetArrayFromImage(mask)
        foreground_voxels = np.argwhere(mask_array > 0)
        
        # Initialize feature matrix
        n_voxels = len(foreground_voxels)
        
        # Get the number of features from a sample extraction
        sample_patch = self._extract_patch(image, mask, foreground_voxels[0], window_size_voxels)
        sample_features = extractor.execute(sample_patch[0], sample_patch[1], label=1)
        # Remove diagnostic features
        sample_features = {k: v for k, v in sample_features.items() if not k.startswith('diagnostic')}
        n_features = len(sample_features)
        
        feature_matrix = np.zeros((n_voxels, n_features))
        feature_names = list(sample_features.keys())
        
        # Extract features for each foreground voxel
        for i, voxel in enumerate(foreground_voxels):
            patch = self._extract_patch(image, mask, voxel, window_size_voxels)
            try:
                features = extractor.execute(patch[0], patch[1], label=1)
                # Remove diagnostic features
                features = {k: v for k, v in features.items() if not k.startswith('diagnostic')}
                feature_matrix[i, :] = [features[name] for name in feature_names]
            except Exception as e:
                logging.warning(f"Failed to extract features for voxel {voxel}: {str(e)}")
                # Use zeros for failed extractions
                feature_matrix[i, :] = 0
        
        return feature_matrix, foreground_voxels, feature_names
    
    def _extract_patch(self, image: sitk.Image, mask: sitk.Image, 
                     voxel: np.ndarray, window_size: List[int]) -> Tuple[sitk.Image, sitk.Image]:
        """
        Extract a patch centered at the specified voxel
        
        Args:
            image: Original image
            mask: Tumor mask
            voxel: Center voxel coordinates (z, y, x)
            window_size: Window size in voxels [z, y, x]
            
        Returns:
            Tuple[sitk.Image, sitk.Image]: Patch image and mask
        """
        # Get image arrays
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Calculate patch boundaries
        z, y, x = voxel
        z_min = max(0, z - window_size[0] // 2)
        z_max = min(image_array.shape[0], z + window_size[0] // 2 + 1)
        y_min = max(0, y - window_size[1] // 2)
        y_max = min(image_array.shape[1], y + window_size[1] // 2 + 1)
        x_min = max(0, x - window_size[2] // 2)
        x_max = min(image_array.shape[2], x + window_size[2] // 2 + 1)
        
        # Extract patches
        image_patch = image_array[z_min:z_max, y_min:y_max, x_min:x_max]
        mask_patch = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Convert back to SimpleITK images
        patch_image = sitk.GetImageFromArray(image_patch)
        patch_image.SetSpacing(image.GetSpacing())
        patch_image.SetOrigin(image.GetOrigin())
        patch_image.SetDirection(image.GetDirection())
        
        patch_mask = sitk.GetImageFromArray(mask_patch)
        patch_mask.SetSpacing(mask.GetSpacing())
        patch_mask.SetOrigin(mask.GetOrigin())
        patch_mask.SetDirection(mask.GetDirection())
        
        return patch_image, patch_mask
    
    def determine_optimal_clusters(self, feature_matrix: np.ndarray, 
                                 max_clusters: int = 10) -> int:
        """
        Determine optimal number of clusters using elbow method
        
        Args:
            feature_matrix: Feature matrix
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            int: Optimal number of clusters
        """
        # Calculate distortion (sum of squared distances) for different k values
        distortions = []
        silhouette_scores = []
        k_values = range(2, min(max_clusters + 1, feature_matrix.shape[0]))
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(feature_matrix)
            distortions.append(kmeans.inertia_)
            
            # Calculate silhouette score if possible
            if k > 1 and k < feature_matrix.shape[0]:
                try:
                    cluster_labels = kmeans.predict(feature_matrix)
                    silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                except:
                    silhouette_scores.append(0)
        
        # Find elbow point using the second derivative
        if len(distortions) > 2:
            deltas = np.diff(distortions)
            delta_deltas = np.diff(deltas)
            elbow_index = np.argmax(delta_deltas) + 2  # +2 because we have two derivatives and start at k=2
            optimal_k = k_values[elbow_index]
        else:
            # If not enough points to find elbow, choose based on silhouette score
            if silhouette_scores:
                optimal_k = k_values[np.argmax(silhouette_scores)]
            else:
                optimal_k = 2  # Default
        
        return optimal_k
    
    def calculate_ith_score(self, clustered_mask: np.ndarray) -> float:
        """
        Calculate ITH score based on the clustered mask
        
        Args:
            clustered_mask: Mask with cluster labels
            
        Returns:
            float: ITH score
        """
        # Get unique clusters (excluding background - 0)
        clusters = np.unique(clustered_mask)
        clusters = clusters[clusters > 0]
        
        if len(clusters) == 0:
            return 0.0
        
        # Calculate total tumor area
        total_area = np.sum(clustered_mask > 0)
        
        summation = 0.0
        for cluster in clusters:
            # Create binary mask for this cluster
            cluster_mask = (clustered_mask == cluster).astype(np.uint8)
            
            # Label connected components
            labeled_array, num_regions = ndimage.label(cluster_mask)
            
            # Skip if no regions found
            if num_regions == 0:
                continue
            
            # Calculate area of each connected region
            region_areas = np.zeros(num_regions + 1)
            for i in range(1, num_regions + 1):
                region_areas[i] = np.sum(labeled_array == i)
            
            # Get largest region area and number of regions
            largest_area = np.max(region_areas[1:])
            
            # Add to summation
            summation += largest_area / num_regions
        
        # Calculate ITH score: 1 - (1/S_total) * Σ(S_i,max / n_i)
        ith_score = 1.0 - (1.0 / total_area) * summation
        
        return ith_score
    
    def extract_ith_features(self, image_path: str, mask_path: str, 
                             out_dir: Optional[str] = None) -> Dict:
        """
        Extract ITH features from image and mask
        
        Args:
            image_path: Path to original image
            mask_path: Path to tumor mask
            out_dir: Optional output directory for saving visualizations
            
        Returns:
            Dict: Dictionary containing ITH score and related features
        """
        try:
            # Load images
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            
            # Ensure mask is binary
            mask_array = sitk.GetArrayFromImage(mask)
            binary_mask = (mask_array > 0).astype(np.uint8)
            binary_mask_img = sitk.GetImageFromArray(binary_mask)
            binary_mask_img.CopyInformation(mask)
            
            # Expand tumor region
            expanded_mask = self.expand_tumor_region(binary_mask_img, self.margin_size)
            
            # Extract local features using sliding window
            feature_matrix, voxel_coords, feature_names = self.extract_local_features(
                image, expanded_mask, self.window_size)
            
            # Determine optimal number of clusters
            optimal_k = self.determine_optimal_clusters(feature_matrix)
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            # Create clustered mask
            clustered_mask = np.zeros_like(mask_array)
            for label, voxel in zip(cluster_labels, voxel_coords):
                z, y, x = voxel
                clustered_mask[z, y, x] = label + 1  # Add 1 to avoid 0 (background)
            
            # Calculate ITH score
            ith_score = self.calculate_ith_score(clustered_mask)
            
            # Generate visualization if output directory is provided
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                subject_id = os.path.basename(image_path).split('.')[0]
                self._save_visualization(mask_array, clustered_mask, subject_id, out_dir)
            
            # Create result dictionary
            result = {
                'ith_score': ith_score,
                'num_clusters': optimal_k,
                'num_voxels': len(voxel_coords)
            }
            
            # Add cluster statistics
            for i in range(optimal_k):
                cluster_mask = (clustered_mask == i+1)
                labeled_array, num_regions = ndimage.label(cluster_mask)
                region_areas = np.zeros(num_regions + 1)
                for j in range(1, num_regions + 1):
                    region_areas[j] = np.sum(labeled_array == j)
                
                if num_regions > 0:
                    largest_area = np.max(region_areas[1:])
                    result[f'cluster_{i+1}_regions'] = num_regions
                    result[f'cluster_{i+1}_largest_area'] = largest_area
                    result[f'cluster_{i+1}_area_ratio'] = largest_area / num_regions if num_regions > 0 else 0
            
            return result
            
        except Exception as e:
            logging.error(f"Error extracting ITH features: {str(e)}")
            return {"error": str(e), "ith_score": 0.0}
    
    def _save_visualization(self, original_mask: np.ndarray, 
                          clustered_mask: np.ndarray, 
                          subject_id: str, 
                          out_dir: str):
        """
        Save visualization of original mask and clustered mask
        
        Args:
            original_mask: Original tumor mask
            clustered_mask: Mask with cluster labels
            subject_id: Subject identifier
            out_dir: Output directory
        """
        # Find central slice for visualization
        non_zero_indices = np.where(original_mask > 0)
        if len(non_zero_indices[0]) > 0:
            central_slice = int(np.median(non_zero_indices[0]))
        else:
            central_slice = original_mask.shape[0] // 2
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original mask
        axes[0].imshow(original_mask[central_slice], cmap='gray')
        axes[0].set_title('Original Tumor Mask')
        axes[0].axis('off')
        
        # Clustered mask
        cluster_img = axes[1].imshow(clustered_mask[central_slice], cmap='viridis')
        axes[1].set_title('Tumor Clusters')
        axes[1].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(cluster_img, ax=axes[1], shrink=0.7)
        cbar.set_label('Cluster ID')
        
        # Save figure
        plt.suptitle(f'Subject: {subject_id}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{subject_id}_ith_clusters.png'), dpi=150)
        plt.close()