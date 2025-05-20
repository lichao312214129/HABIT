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

# 导入各个选择器模块
from .anova_selector import *
from .correlation_selector import *
from .icc_selector import *
from .lasso_selector import *
from .mrmr_selector import *
from .python_stepwise_selector import *
from .rfecv_selector import *

try:
    from .stepwise_selector import *
except Exception as e:
    print(f"Error importing stepwise_selector: {e}")
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




    