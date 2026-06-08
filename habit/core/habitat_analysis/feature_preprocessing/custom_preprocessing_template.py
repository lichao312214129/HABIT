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
Template for registering a custom habitat feature preprocessing method.

Steps to add a new method:
1. Subclass ``BaseFeaturePreprocessing`` below.
2. Set ``changes_columns=True`` when the method drops or reorders feature columns.
3. Apply ``@register_preprocessing("your_method")`` with the YAML ``method`` key.
4. Add the name to ``PreprocessingMethod.method`` Literal in ``config_schemas.py``.
5. Import your module once at startup (or place it under ``feature_preprocessing/``
   so it is discovered alongside ``builtin_methods``).

Example YAML::

    preprocessing_for_group_level:
      methods:
        - method: custom_template
          scale: 2.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd
from pydantic import BaseModel

from .base_preprocessing import BaseFeaturePreprocessing, BaselineStats, register_preprocessing
from .method_config_utils import read_method_field


@register_preprocessing("custom_template")
class CustomTemplatePreprocessing(BaseFeaturePreprocessing):
    """
    Example handler: multiply all feature columns by a configurable scale factor.

    Replace this logic with your own algorithm; keep the DataFrame in/out contract.
    """

    changes_columns = False

    @classmethod
    def method_name(cls) -> str:
        return "custom_template"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        # Return any JSON/pickle-friendly state needed at transform time.
        return {"scale": float(read_method_field(method_config, "scale", 1.0))}

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        scale = state["scale"] if state else float(read_method_field(method_config, "scale", 1.0))
        return feature_df * scale
