# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------------------------------------------------
# Third-party attribution
#
# Every module in this ``torchradiomics`` sub-package is vendored from the
# PyTorchRadiomics project and adapted for HABIT:
#
#     https://github.com/lyhyl/pytorchradiomics
#     Copyright (c) lyhyl and PyTorchRadiomics contributors
#     Licensed under the MIT License
#
# The upstream notice is retained here and in the NOTICE file at the project
# root, as required by that license.
# ---------------------------------------------------------------------------
"""
GPU-accelerated PyRadiomics feature classes vendored from PyTorchRadiomics.

The classes mirror the PyRadiomics feature-class interface so that
``inject_torch_radiomics`` can substitute them for the CPU implementations
inside an existing extraction workflow.
"""

import radiomics
from radiomics import firstorder, glcm, gldm, glrlm, glszm, ngtdm

from .TorchRadiomicsFirstOrder import TorchRadiomicsFirstOrder
from .TorchRadiomicsGLCM import TorchRadiomicsGLCM
from .TorchRadiomicsGLDM import TorchRadiomicsGLDM
from .TorchRadiomicsGLRLM import TorchRadiomicsGLRLM
from .TorchRadiomicsGLSZM import TorchRadiomicsGLSZM
from .TorchRadiomicsNGTDM import TorchRadiomicsNGTDM


def inject_torch_radiomics():
    radiomics._featureClasses["firstorder"] = TorchRadiomicsFirstOrder
    radiomics._featureClasses["glcm"] = TorchRadiomicsGLCM
    radiomics._featureClasses["gldm"] = TorchRadiomicsGLDM
    radiomics._featureClasses["glrlm"] = TorchRadiomicsGLRLM
    radiomics._featureClasses["glszm"] = TorchRadiomicsGLSZM
    radiomics._featureClasses["ngtdm"] = TorchRadiomicsNGTDM


def restore_radiomics():
    radiomics._featureClasses["firstorder"] = firstorder.RadiomicsFirstOrder
    radiomics._featureClasses["glcm"] = glcm.RadiomicsGLCM
    radiomics._featureClasses["gldm"] = gldm.RadiomicsGLDM
    radiomics._featureClasses["glrlm"] = glrlm.RadiomicsGLRLM
    radiomics._featureClasses["glszm"] = glszm.RadiomicsGLSZM
    radiomics._featureClasses["ngtdm"] = ngtdm.RadiomicsNGTDM
