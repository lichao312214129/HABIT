import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor, imageoperations, cMatrices
from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLDM import TorchRadiomicsGLDM

def shift(tensor, dz, dy, dx, fill_value=-1):
    result = torch.full_like(tensor, fill_value)
    z_start = max(0, dz); z_end = min(tensor.shape[0], tensor.shape[0] + dz)
    y_start = max(0, dy); y_end = min(tensor.shape[1], tensor.shape[1] + dy)
    x_start = max(0, dx); x_end = min(tensor.shape[2], tensor.shape[2] + dx)
    
    src_z_start = max(0, -dz); src_z_end = min(tensor.shape[0], tensor.shape[0] - dz)
    src_y_start = max(0, -dy); src_y_end = min(tensor.shape[1], tensor.shape[1] - dy)
    src_x_start = max(0, -dx); src_x_end = min(tensor.shape[2], tensor.shape[2] - dx)
    
    result[z_start:z_end, y_start:y_end, x_start:x_end] = tensor[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
    return result

def test_gldm():
    image_arr = np.random.rand(10, 10, 10).astype(np.float32) * 100
    sv_map_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    sv_map_arr[2:8, 2:8, 2:8] = 1
    
    image = sitk.GetImageFromArray(image_arr)
    sv_map = sitk.GetImageFromArray(sv_map_arr)
    
    settings = {'binWidth': 25, 'force2D': False, 'gldm_a': 0}
    
    mask_all = sitk.GetImageFromArray((sv_map_arr > 0).astype(np.uint8))
    mask_all_arr = sitk.GetArrayFromImage(mask_all) > 0
    binned_image_arr, _ = imageoperations.binImage(image_arr, mask_all_arr, **settings)
    Ng = int(np.max(binned_image_arr[sv_map_arr > 0]))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    binned_tensor = torch.tensor(binned_image_arr, dtype=torch.long, device=device)
    sv_tensor = torch.tensor(sv_map_arr, dtype=torch.long, device=device)
    
    tr_gldm = TorchRadiomicsGLDM(image, mask_all, **settings)
    tr_gldm.imageArray = binned_image_arr
    tr_gldm.coefficients['grayLevels'] = np.arange(1, Ng + 1)
    tr_gldm.coefficients['Ng'] = Ng
    P_single = tr_gldm._calculateMatrix()
    
    print("PyRadiomics P_gldm shape:", P_single.shape)
    
    # GPU builder
    # GLDM uses 26 angles (all neighbors)
    angles_c = cMatrices.generate_angles(np.array(image_arr.shape), np.array([1]), True, False, 0)
    
    alpha = settings.get('gldm_a', 0)
    
    dependence = torch.zeros_like(binned_tensor, dtype=torch.long)
    
    for angle in angles_c:
        dz, dy, dx = angle
        shifted_binned = shift(binned_tensor, dz, dy, dx, fill_value=-1000)
        shifted_sv = shift(sv_tensor, dz, dy, dx, fill_value=-1)
        
        valid = (sv_tensor == shifted_sv) & (torch.abs(binned_tensor - shifted_binned) <= alpha)
        dependence[valid] += 1
        
    # P_gldm shape: (BatchSize, Ng, Nd, 1)
    # Nd is max dependence + 1 (since dependence can be 0? wait, dependence includes the pixel itself?)
    # PyRadiomics GLDM dependence counts the pixel itself?
    # Let's check PyRadiomics GLDM Nd.
    # PyRadiomics GLDM uses Nd = number of neighbors + 1 (so max 27 for 3D).
    # Wait, does PyRadiomics count the pixel itself?
    # Let's see the shape of P_single.
    Nd = P_single.shape[2]
    P_gldm = torch.zeros((1, Ng, Nd), dtype=torch.float64, device=device)
    
    valid_mask = sv_tensor > 0
    sv_valid = sv_tensor[valid_mask]
    gl_valid = binned_tensor[valid_mask] - 1
    
    dep_valid = dependence[valid_mask]
    
    batch_idx = sv_valid - 1
    
    flat_indices = batch_idx * (Ng * Nd) + gl_valid * Nd + (dep_valid + 1)
    counts = torch.bincount(flat_indices, minlength=1 * Ng * Nd).to(torch.float64)
    P_gldm[:, :, :] = counts.view(1, Ng, Nd)
    
    print("GPU P_gldm shape:", P_gldm.shape)
    diff = torch.abs(P_single - P_gldm).max().item()
    print("Difference (+1):", diff)
    
    if diff > 0:
        flat_indices = batch_idx * (Ng * Nd) + gl_valid * Nd + dep_valid
        counts = torch.bincount(flat_indices, minlength=1 * Ng * Nd).to(torch.float64)
        P_gldm[:, :, :] = counts.view(1, Ng, Nd)
        print("Difference (+0):", torch.abs(P_single - P_gldm).max().item())

if __name__ == "__main__":
    test_gldm()
