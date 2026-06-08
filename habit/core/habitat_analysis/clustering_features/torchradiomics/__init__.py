# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
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
