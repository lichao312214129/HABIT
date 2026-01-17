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
            "parameter": "./config/parameter.yaml"
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
        ("pre_contrast", {"parameter": "./config/parameter.yaml"}),
        ("LAP", {"parameter": "./config/parameter.yaml"}),
        ("PVP", {"parameter": "./config/parameter.yaml"}),
        ("delay_3min", {"parameter": "./config/parameter.yaml"})
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
