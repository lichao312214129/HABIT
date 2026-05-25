import SimpleITK as sitk
import numpy as np
from radiomics.firstorder import RadiomicsFirstOrder

img_arr = np.random.randint(0, 100, size=(10, 10, 10)).astype(np.float32)
mask_arr = np.zeros((10, 10, 10), dtype=np.uint8)
mask_arr[5:8, 5:8, 5:8] = 1
# set a value outside the mask to -50
img_arr[0,0,0] = -50

img = sitk.GetImageFromArray(img_arr)
mask = sitk.GetImageFromArray(mask_arr)

ex = RadiomicsFirstOrder(img, mask, binWidth=25)
ex._initCalculation()

print('min of bounding box:', np.min(ex.imageArray))
print('min of ROI:', np.min(ex.imageArray[ex.maskArray]))
