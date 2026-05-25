import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor, imageoperations, cMatrices
from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLRLM import TorchRadiomicsGLRLM

def shift(tensor, dz, dy, dx, fill_value=-1):
    # Shift a 3D tensor by dz, dy, dx
    # Positive dz means shift down (elements move to higher indices)
    # So the new tensor at [z, y, x] comes from [z-dz, y-dy, x-dx]
    result = torch.full_like(tensor, fill_value)
    
    z_start = max(0, dz); z_end = min(tensor.shape[0], tensor.shape[0] + dz)
    y_start = max(0, dy); y_end = min(tensor.shape[1], tensor.shape[1] + dy)
    x_start = max(0, dx); x_end = min(tensor.shape[2], tensor.shape[2] + dx)
    
    src_z_start = max(0, -dz); src_z_end = min(tensor.shape[0], tensor.shape[0] - dz)
    src_y_start = max(0, -dy); src_y_end = min(tensor.shape[1], tensor.shape[1] - dy)
    src_x_start = max(0, -dx); src_x_end = min(tensor.shape[2], tensor.shape[2] - dx)
    
    result[z_start:z_end, y_start:y_end, x_start:x_end] = tensor[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
    return result

def test_glrlm():
    image_arr = np.random.rand(10, 10, 10).astype(np.float32) * 100
    sv_map_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    sv_map_arr[2:8, 2:8, 2:8] = 1
    
    image = sitk.GetImageFromArray(image_arr)
    sv_map = sitk.GetImageFromArray(sv_map_arr)
    
    settings = {'binWidth': 25, 'force2D': False}
    
    mask_all = sitk.GetImageFromArray((sv_map_arr > 0).astype(np.uint8))
    mask_all_arr = sitk.GetArrayFromImage(mask_all) > 0
    binned_image_arr, _ = imageoperations.binImage(image_arr, mask_all_arr, **settings)
    Ng = int(np.max(binned_image_arr[sv_map_arr > 0]))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    binned_tensor = torch.tensor(binned_image_arr, dtype=torch.long, device=device)
    sv_tensor = torch.tensor(sv_map_arr, dtype=torch.long, device=device)
    
    angles = cMatrices.generate_angles(np.array(image_arr.shape), np.array([1]), False, False, 0) # GLRLM uses bidirectional=False? Wait, PyRadiomics GLRLM uses 13 angles!
    # Let's check cMatrices.calculate_glrlm
    # Actually, PyRadiomics generates 13 angles for GLRLM.
    
    # We will just use TorchRadiomicsGLRLM to get the angles
    tr_glrlm = TorchRadiomicsGLRLM(image, mask_all, **settings)
    tr_glrlm.imageArray = binned_image_arr
    tr_glrlm.coefficients['grayLevels'] = np.arange(1, Ng + 1)
    tr_glrlm.coefficients['Ng'] = Ng
    P_single = tr_glrlm._calculateMatrix()
    
    print("PyRadiomics P_glrlm shape:", P_single.shape)
    
    # Now our GPU builder
    # angles used by cMatrices.calculate_glrlm are 13 angles.
    # We can get them from cMatrices
    _, angles_c = cMatrices.calculate_glrlm(binned_image_arr, (sv_map_arr>0).astype(np.int8), Ng, np.max(image_arr.shape), False, 0)
    
    Nr = np.max(image_arr.shape)
    P_glrlm = torch.zeros((1, Ng, Nr, len(angles_c)), dtype=torch.float64, device=device)
    
    for a_idx, angle in enumerate(angles_c):
        dz, dy, dx = angle
        
        # A run starts if the PREVIOUS pixel is different
        shifted_binned_neg = shift(binned_tensor, -dz, -dy, -dx, fill_value=-1)
        shifted_sv_neg = shift(sv_tensor, -dz, -dy, -dx, fill_value=-1)
        
        is_start = (binned_tensor != shifted_binned_neg) | (sv_tensor != shifted_sv_neg)
        is_start &= (sv_tensor > 0)
        
        run_lengths = torch.ones_like(binned_tensor, dtype=torch.long)
        
        # We only care about start pixels
        active = is_start.clone()
        
        for l in range(1, Nr):
            if not active.any():
                break
            
            shifted_binned = shift(binned_tensor, l*dz, l*dy, l*dx, fill_value=-1)
            shifted_sv = shift(sv_tensor, l*dz, l*dy, l*dx, fill_value=-1)
            
            match = (shifted_binned == binned_tensor) & (shifted_sv == sv_tensor)
            
            active &= match
            run_lengths[active] += 1
            
        # Now scatter add
        start_pixels = is_start
        sv_valid = sv_tensor[start_pixels]
        gl_valid = binned_tensor[start_pixels] - 1
        rl_valid = run_lengths[start_pixels] - 1
        
        batch_idx = sv_valid - 1 # Since we only have label 1
        
        flat_indices = batch_idx * (Ng * Nr) + gl_valid * Nr + rl_valid
        counts = torch.bincount(flat_indices, minlength=1 * Ng * Nr).to(torch.float64)
        P_glrlm[:, :, :, a_idx] = counts.view(1, Ng, Nr)
        
    print("GPU P_glrlm shape:", P_glrlm.shape)
    print("Difference:", torch.abs(P_single - P_glrlm).max().item())

if __name__ == "__main__":
    test_glrlm()
