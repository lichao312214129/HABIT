import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor

from habit.core.habitat_analysis.clustering_features.torchradiomics.TorchRadiomicsGLCM import TorchRadiomicsGLCM

def test_injection():
    # Create a dummy image and mask
    image = sitk.GetImageFromArray(np.random.rand(20, 20, 20).astype(np.float32) * 100)
    mask = sitk.GetImageFromArray((np.random.rand(20, 20, 20) > 0.5).astype(np.uint8))
    
    # Extract using PyRadiomics
    settings = {'binWidth': 25, 'force2D': False}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glcm')
    result = extractor.execute(image, mask)
    
    print("PyRadiomics Contrast:", result.get('original_glcm_Contrast'))
    
    # Now try to use TorchRadiomicsGLCM with tensor injection
    # Wait, we need to build P_glcm first.
    # Let's just let TorchRadiomicsGLCM build it for the single ROI, then we see if we can replicate it.
    tr_glcm = TorchRadiomicsGLCM(image, mask, **settings)
    tr_glcm._initCalculation()
    print("TorchRadiomics Contrast:", tr_glcm.getContrastFeatureValue())

if __name__ == "__main__":
    test_injection()
