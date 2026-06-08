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
"""
Runner layer for machine-learning workflows.

Concrete runners encapsulate the training/evaluation/prediction loops.
Workflows delegate heavy logic to runners so the public workflow classes
stay thin orchestration shells.
"""

from .context import RunnerContext
from .holdout import HoldoutRunner
from .inference import InferenceRunner
from .kfold import KFoldRunner

__all__ = [
    "RunnerContext",
    "HoldoutRunner",
    "InferenceRunner",
    "KFoldRunner",
]
