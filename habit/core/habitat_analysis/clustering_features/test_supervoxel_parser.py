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
Test the feature expression parser with supervoxel level configurations
"""

from habit.core.habitat_analysis.clustering_features.feature_expression_parser import FeatureExpressionParser

def test_supervoxel_parser():
    """Test feature expression parser with supervoxel level configurations"""
    parser = FeatureExpressionParser()

    # Test supervoxel level configuration
    config = {
        "method": "concat(supervoxel_radiomics(pre_contrast, parameter), supervoxel_radiomics(LAP, parameter), supervoxel_radiomics(PVP, parameter), supervoxel_radiomics(delay_3min, parameter))",
        "params": {
            "parameter": "./config/radiomics/parameter.yaml"
        }
    }

    # Parse the configuration
    cross_image_method, cross_image_params, processing_steps = parser.parse(config)

    print("== Test with supervoxel level configuration ==")
    print(f"Cross-image method: {cross_image_method}")
    print(f"Cross-image params: {cross_image_params}")
    print(f"Processing steps:")
    for step in processing_steps:
        print(f"  Method: {step['method']}, Image: {step['image']}, Params: {step['params']}")

    # Verify the results
    assert cross_image_method == "concat", f"Expected cross_image_method to be 'concat', got '{cross_image_method}'"
    assert len(processing_steps) == 4, f"Expected 4 processing steps, got {len(processing_steps)}"

    # Check each processing step
    for i, (image_name, expected_params) in enumerate([
        ("pre_contrast", {"parameter": "./config/radiomics/parameter.yaml"}),
        ("LAP", {"parameter": "./config/radiomics/parameter.yaml"}),
        ("PVP", {"parameter": "./config/radiomics/parameter.yaml"}),
        ("delay_3min", {"parameter": "./config/radiomics/parameter.yaml"})
    ]):
        step = processing_steps[i]
        assert step["method"] == "supervoxel_radiomics", f"Expected method to be 'supervoxel_radiomics', got '{step['method']}'"
        assert step["image"] == image_name, f"Expected image to be '{image_name}', got '{step['image']}'"

        # Check if parameter is correctly passed
        assert "parameter" in step["params"], f"Expected 'parameter' in params, got {step['params']}"
        assert step["params"]["parameter"] == expected_params["parameter"], f"Expected parameter value to be '{expected_params['parameter']}', got '{step['params']['parameter']}'"

    print("All tests passed!")

if __name__ == "__main__":
    test_supervoxel_parser()
