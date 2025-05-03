import os

import SimpleITK as sitk
import six

from radiomics import featureextractor, getTestCase

dataDir = 'tests/data'

imageName, maskName = getTestCase('brain1', dataDir)

params = os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")