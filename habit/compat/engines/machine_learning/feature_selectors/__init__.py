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
"""
Feature Selector Package

Provides a unified interface for feature selection methods
"""

# 导入注册机制
from .selector_registry import (
    SelectorRegistry,
    run_selector,
    SelectorContext,
    SelectorResult,
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
from .univariate_logistic_selector import *
from .vif_selector import *

# 导出接口
__all__ = [
    'SelectorRegistry',
    'run_selector',
    'SelectorContext',
    'SelectorResult',
]




    