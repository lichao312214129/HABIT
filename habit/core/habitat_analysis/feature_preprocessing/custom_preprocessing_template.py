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
Template for registering a custom habitat feature preprocessing method.

Steps to add a new method:
1. Subclass ``BaseFeaturePreprocessing`` below.
2. Set ``changes_columns=True`` when the method drops or reorders feature columns.
3. Apply ``@PreprocessingMethodFactory.register("your_method")`` with the YAML ``method`` key.
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

from .base_preprocessing import BaseFeaturePreprocessing, BaselineStats, PreprocessingMethodFactory
from .method_config_utils import read_method_field


@PreprocessingMethodFactory.register("custom_template")
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
