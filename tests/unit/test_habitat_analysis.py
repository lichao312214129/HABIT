"""
Unit tests for habitat analysis functionality.
"""

import pytest
from pathlib import Path
import numpy as np
from habit.core.habitat_analysis import HabitatAnalysis

@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration file for testing."""
    config_path = tmp_path / "test_config.yaml"
    config = {
        "feature_extraction": {
            "parameters": {
                "binWidth": 25,
                "resampledPixelSpacing": [1, 1, 1]
            }
        },
        "clustering": {
            "n_clusters": 3,
            "method": "kmeans"
        }
    }
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path

def test_habitat_analysis_initialization(sample_config):
    """Test HabitatAnalysis initialization."""
    analyzer = HabitatAnalysis(sample_config)
    assert analyzer.config is not None
    assert analyzer.feature_extractor is None
    assert analyzer.clustering_model is None

def test_config_loading(sample_config):
    """Test configuration loading."""
    analyzer = HabitatAnalysis(sample_config)
    assert "feature_extraction" in analyzer.config
    assert "clustering" in analyzer.config
    assert analyzer.config["clustering"]["n_clusters"] == 3

@pytest.mark.parametrize("invalid_config", [
    "nonexistent_config.yaml",
    "invalid_yaml.yaml"
])
def test_invalid_config_handling(tmp_path, invalid_config):
    """Test handling of invalid configuration files."""
    config_path = tmp_path / invalid_config
    if invalid_config == "invalid_yaml.yaml":
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content")
    
    with pytest.raises((FileNotFoundError, ValueError)):
        HabitatAnalysis(config_path) 