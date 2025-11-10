# conftest.py
"""Pytest configuration and shared fixtures"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def demo_data_dir(project_root) -> Path:
    """Get demo data directory"""
    return project_root / "demo_image_data"


@pytest.fixture(scope="session")
def test_output_dir():
    """Create temporary output directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="habit_test_")
    yield Path(temp_dir)
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_image_3d() -> np.ndarray:
    """Create a sample 3D image for testing"""
    # Create a simple 3D image (e.g., 64x64x32)
    image = np.random.randn(64, 64, 32).astype(np.float32)
    return image


@pytest.fixture
def sample_mask_3d() -> np.ndarray:
    """Create a sample 3D mask for testing"""
    # Create a simple binary mask
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    mask[16:48, 16:48, 8:24] = 1  # Create a box-shaped mask
    return mask


@pytest.fixture
def sample_features() -> np.ndarray:
    """Create sample feature matrix for testing"""
    # Create feature matrix: 100 samples, 50 features
    X = np.random.randn(100, 50)
    return X


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Create sample labels for testing"""
    # Create binary labels: 100 samples
    y = np.random.randint(0, 2, size=100)
    return y


@pytest.fixture
def sample_config_dict() -> dict:
    """Create sample configuration dictionary"""
    config = {
        'data_dir': 'path/to/data',
        'out_dir': 'path/to/output',
        'processes': 1,
        'random_state': 42,
    }
    return config


@pytest.fixture
def mock_config_file(tmp_path, sample_config_dict):
    """Create a mock YAML configuration file"""
    import yaml
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config_dict, f)
    return config_file

