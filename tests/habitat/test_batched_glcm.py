import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor, imageoperations, cMatrices

from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLCM import TorchRadiomicsGLCM

def test_batched_glcm():
    # Create a dummy image and a supervoxel map with 3 supervoxels
    image_arr = np.random.rand(20, 20, 20).astype(np.float32) * 100
    sv_map_arr = np.zeros((20, 20, 20), dtype=np.uint8)
    sv_map_arr[2:8, 2:8, 2:8] = 1
    sv_map_arr[10:15, 10:15, 10:15] = 2
    sv_map_arr[2:8, 10:15, 10:15] = 3
    
    image = sitk.GetImageFromArray(image_arr)
    sv_map = sitk.GetImageFromArray(sv_map_arr)
    
    settings = {'binWidth': 25, 'force2D': False}
    
    # 1. Global discretization
    # We use imageoperations.binImage with a mask of ALL supervoxels
    mask_all = sitk.GetImageFromArray((sv_map_arr > 0).astype(np.uint8))
    mask_all_arr = sitk.GetArrayFromImage(mask_all) > 0
    binned_image_arr, _ = imageoperations.binImage(image_arr, mask_all_arr, **settings)
    
    # Ng is the maximum gray level in the whole binned image (within mask)
    Ng = int(np.max(binned_image_arr[sv_map_arr > 0]))
    print("Global Ng:", Ng)
    
    # 2. Build batched P_glcm on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    binned_tensor = torch.tensor(binned_image_arr, dtype=torch.long, device=device)
    sv_tensor = torch.tensor(sv_map_arr, dtype=torch.long, device=device)
    
    angles = cMatrices.generate_angles(
        np.array(image_arr.shape), np.array([1]), True, False, 0
    )
    angles_tensor = torch.tensor(angles, dtype=torch.long, device=device)
    
    batch_labels = [1, 2, 3]
    B = len(batch_labels)
    
    # P_glcm shape: (B, Ng, Ng, Angles)
    P_glcm = torch.zeros((B, Ng, Ng, len(angles)), dtype=torch.float64, device=device)
    
    # Map label to batch index
    # We can create a lookup tensor for fast mapping
    max_label = max(batch_labels)
    label_to_idx = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
    for i, lbl in enumerate(batch_labels):
        label_to_idx[lbl] = i
        
    # For each angle, shift and accumulate
    for a_idx, angle in enumerate(angles):
        dz, dy, dx = angle
        
        # Shift tensors
        # We can use slicing to get the overlapping regions
        z_start_1 = max(0, -dz); z_end_1 = min(image_arr.shape[0], image_arr.shape[0] - dz)
        y_start_1 = max(0, -dy); y_end_1 = min(image_arr.shape[1], image_arr.shape[1] - dy)
        x_start_1 = max(0, -dx); x_end_1 = min(image_arr.shape[2], image_arr.shape[2] - dx)
        
        z_start_2 = max(0, dz); z_end_2 = min(image_arr.shape[0], image_arr.shape[0] + dz)
        y_start_2 = max(0, dy); y_end_2 = min(image_arr.shape[1], image_arr.shape[1] + dy)
        x_start_2 = max(0, dx); x_end_2 = min(image_arr.shape[2], image_arr.shape[2] + dx)
        
        sv_1 = sv_tensor[z_start_1:z_end_1, y_start_1:y_end_1, x_start_1:x_end_1]
        sv_2 = sv_tensor[z_start_2:z_end_2, y_start_2:y_end_2, x_start_2:x_end_2]
        
        val_1 = binned_tensor[z_start_1:z_end_1, y_start_1:y_end_1, x_start_1:x_end_1]
        val_2 = binned_tensor[z_start_2:z_end_2, y_start_2:y_end_2, x_start_2:x_end_2]
        
        # Valid pairs: same supervoxel, and supervoxel > 0
        valid = (sv_1 == sv_2) & (sv_1 > 0)
        
        if not valid.any():
            continue
            
        sv_valid = sv_1[valid]
        v1_valid = val_1[valid] - 1 # 0-based for indexing
        v2_valid = val_2[valid] - 1
        
        batch_idx = label_to_idx[sv_valid]
        
        # Filter out labels not in this batch
        batch_mask = batch_idx >= 0
        batch_idx = batch_idx[batch_mask]
        v1_valid = v1_valid[batch_mask]
        v2_valid = v2_valid[batch_mask]
        
        if len(batch_idx) == 0:
            continue
            
        # Accumulate into P_glcm
        # We need to flatten the indices for scatter_add
        flat_indices = batch_idx * (Ng * Ng) + v1_valid * Ng + v2_valid
        
        # Use bincount or scatter_add
        counts = torch.bincount(flat_indices, minlength=B * Ng * Ng).to(torch.float64)
        P_glcm[:, :, :, a_idx] += counts.view(B, Ng, Ng)
        
    # Make symmetrical
    P_glcm = P_glcm + P_glcm.transpose(1, 2)
    
    # Normalize
    sumP_glcm = P_glcm.sum(dim=(1, 2))
    emptyAngles = sumP_glcm == 0
    sumP_glcm[emptyAngles] = torch.nan
    P_glcm /= sumP_glcm.unsqueeze(1).unsqueeze(2)
    
    print("Batched P_glcm shape:", P_glcm.shape)
    
    # 3. Inject into TorchRadiomicsGLCM
    tr_glcm = TorchRadiomicsGLCM(image, mask_all, **settings)
    tr_glcm.coefficients['Ng'] = Ng
    tr_glcm.coefficients['grayLevels'] = np.arange(1, Ng + 1)
    tr_glcm.P_glcm = P_glcm
    tr_glcm._calculateCoefficients()
    
    batch_contrast = tr_glcm.getContrastFeatureValue()
    print("Batched Contrast:", batch_contrast)
    
    # 4. Compare with PyRadiomics one-by-one
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glcm')
    
    for lbl in batch_labels:
        # Create mask for this label
        mask_lbl = sitk.GetImageFromArray((sv_map_arr == lbl).astype(np.uint8))
        
        # To match the global binning, we must use the globally binned image!
        # PyRadiomics extractor will re-bin the image.
        # So we pass the PRE-BINNED image and set binCount/binWidth so it doesn't re-bin?
        # Actually, PyRadiomics doesn't easily allow passing pre-binned images.
        # But we can just create a TorchRadiomicsGLCM for this label using the global binned image!
        
        # Let's just use TorchRadiomicsGLCM with the global binned image
        # Wait, TorchRadiomicsGLCM will re-bin it.
        # Let's bypass binning for the single ROI test.
        tr_single = TorchRadiomicsGLCM(image, mask_lbl, **settings)
        tr_single.imageArray = binned_image_arr
        tr_single.coefficients['grayLevels'] = np.arange(1, Ng + 1)
        tr_single.coefficients['Ng'] = Ng
        
        # Calculate matrix
        P_single = tr_single._calculateMatrix()
        tr_single.P_glcm = P_single
        tr_single._calculateCoefficients()
        print(f"Label {lbl} Contrast:", tr_single.getContrastFeatureValue())

if __name__ == "__main__":
    test_batched_glcm()
