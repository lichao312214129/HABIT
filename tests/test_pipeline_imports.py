"""
Test script to verify pipeline imports work correctly.

This script tests that all pipeline classes can be imported successfully.
"""

def test_pipeline_imports():
    """Test that all pipeline classes can be imported."""
    print("Testing pipeline imports...")
    
    try:
        # Test importing from pipelines module
        from habit.core.habitat_analysis.pipelines import (
            BasePipelineStep,
            HabitatPipeline,
            build_habitat_pipeline
        )
        print("✓ Successfully imported from pipelines module")
        
        # Test importing steps
        from habit.core.habitat_analysis.pipelines.steps import (
            GroupPreprocessingStep,
            PopulationClusteringStep
        )
        print("✓ Successfully imported pipeline steps")
        
        # Test importing from main module
        from habit.core.habitat_analysis import (
            BasePipelineStep as BPS2,
            HabitatPipeline as HP2,
            GroupPreprocessingStep as GPS2,
            PopulationClusteringStep as PCS2
        )
        print("✓ Successfully imported from habitat_analysis module")
        
        print("\nAll imports successful! ✓")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline_imports()
    exit(0 if success else 1)
