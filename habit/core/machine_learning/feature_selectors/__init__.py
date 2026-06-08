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
Feature Selector Package

Provides a unified interface for feature selection methods
"""

# 导入注册机制
from .selector_registry import (
    register_selector,
    get_selector,
    get_available_selectors,
    run_selector
)
import logging

# 导入各个选择器模块
from .correlation_selector import *
from .icc_selector import *
from .lasso_selector import *
try:
    from .mrmr_selector import *
except Exception as e:
    logging.warning(f"Error importing mrmr_selector: {e}")
    pass
from .python_stepwise_selector import *
from .rfecv_selector import *
from .anova_selector import *
from .chi2_selector import *
from .variance_selector import *
from .statistical_test_selector import *

try:
    from .stepwise_selector import *
except Exception as e:
    logging.warning(f"Error importing stepwise_selector: {e}")
    pass

from .univariate_logistic_selector import *
from .vif_selector import *

# 导出接口
__all__ = [
    'register_selector',
    'get_selector',
    'get_available_selectors',
    'run_selector'
]




    