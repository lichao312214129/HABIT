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
Test file for feature expression parser
"""

import os
import sys

# 直接导入parser类
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..'))

from habit.compat.engines.habitat_analysis.clustering_features.feature_expression_parser import FeatureExpressionParser

def test_parser():
    """Test feature expression parser with different formats"""
    parser = FeatureExpressionParser()
    
    # Test format with parameters
    config_with_params = "kinetic(raw(pre_contrast, p1), raw(LAP, p1), raw(PVP, p2), raw(delay_3min, p3), timestamps)"
    params = {
        "p1": 1.0,
        "p2": True,
        "p3": False,
        "timestamps": "path/to/timestamps.xlsx"
    }
    
    # Parse with parameters
    cross_image_method, cross_image_params, processing_steps = parser.parse(config_with_params, params)
    
    print("== Test with parameters ==")
    print(f"Cross-image method: {cross_image_method}")
    print(f"Cross-image params: {cross_image_params}")
    print(f"Processing steps:")
    for step in processing_steps:
        print(f"  Method: {step['method']}, Image: {step['image']}, Params: {step['params']}")
    
    # Test format without parameters
    config_without_params = "kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)"
    
    # Parse without parameters
    cross_image_method, cross_image_params, processing_steps = parser.parse(config_without_params, params)
    
    print("\n== Test without parameters ==")
    print(f"Cross-image method: {cross_image_method}")
    print(f"Cross-image params: {cross_image_params}")
    print(f"Processing steps:")
    for step in processing_steps:
        print(f"  Method: {step['method']}, Image: {step['image']}, Params: {step['params']}")
    
    # Test raw outer function without nested function
    config_raw_only = "raw(pre_contrast)"
    
    # Parse raw outer function
    cross_image_method, cross_image_params, processing_steps = parser.parse(config_raw_only, {})
    
    print("\n== Test raw outer function ==")
    print(f"Cross-image method: {cross_image_method}")
    print(f"Cross-image params: {cross_image_params}")
    print(f"Processing steps:")
    for step in processing_steps:
        print(f"  Method: {step['method']}, Image: {step['image']}, Params: {step['params']}")

if __name__ == "__main__":
    test_parser() 