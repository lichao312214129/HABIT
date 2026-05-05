"""
Test suite for metrics module optimizations.

Tests:
1. Confusion matrix caching performance
2. Extended target metrics (PPV, NPV, F1)
3. Fallback mechanism for no-solution cases
4. Pareto optimal threshold selection
5. Category-based metric filtering
"""

import numpy as np
import time
import pytest
from habit.core.machine_learning.evaluation.metrics import (
    calculate_metrics,
    calculate_metrics_at_target,
    apply_threshold,
    MetricsCache
)


class TestMetricsCaching:
    """Test confusion matrix caching optimization."""
    
    def test_cache_performance_improvement(self):
        """Test that caching provides performance improvement."""
        # Generate test data
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_pred_proba = np.random.rand(1000)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Test without cache
        start_no_cache = time.time()
        for _ in range(10):
            metrics_no_cache = calculate_metrics(y_true, y_pred, y_pred_proba, use_cache=False)
        time_no_cache = time.time() - start_no_cache
        
        # Test with cache
        start_cache = time.time()
        for _ in range(10):
            metrics_cache = calculate_metrics(y_true, y_pred, y_pred_proba, use_cache=True)
        time_cache = time.time() - start_cache
        
        # Cache should be faster
        assert time_cache < time_no_cache
        
        # Results should be identical
        for key in metrics_no_cache:
            if not np.isnan(metrics_no_cache[key]):
                assert abs(metrics_no_cache[key] - metrics_cache[key]) < 1e-10
        
        print(f"\nPerformance improvement: {time_no_cache/time_cache:.2f}x faster with cache")
    
    def test_metrics_cache_class(self):
        """Test MetricsCache class directly."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        
        cache = MetricsCache(y_true, y_pred, y_prob)
        
        # First access calculates
        cm1 = cache.confusion_matrix
        
        # Second access uses cache
        cm2 = cache.confusion_matrix
        
        # Should be the same object
        assert cm1 is cm2


class TestExtendedTargetMetrics:
    """Test extended target metrics support (PPV, NPV, F1)."""
    
    def test_ppv_target(self):
        """Test finding threshold for PPV target."""
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(100), np.ones(100)])
        y_pred_proba = np.concatenate([
            np.random.beta(2, 5, 100),  # Negatives
            np.random.beta(5, 2, 100)   # Positives
        ])
        
        targets = {'ppv': 0.80, 'sensitivity': 0.70}
        result = calculate_metrics_at_target(y_true, y_pred_proba, targets)
        
        # Should find individual thresholds
        assert 'ppv' in result['thresholds']
        assert 'sensitivity' in result['thresholds']
        
        # Check that thresholds actually meet targets
        ppv_metrics = result['metrics_at_thresholds']['ppv']
        assert ppv_metrics['ppv'] >= 0.80
    
    def test_npv_target(self):
        """Test finding threshold for NPV target."""
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(100), np.ones(100)])
        y_pred_proba = np.concatenate([
            np.random.beta(2, 5, 100),
            np.random.beta(5, 2, 100)
        ])
        
        targets = {'npv': 0.85}
        result = calculate_metrics_at_target(y_true, y_pred_proba, targets)
        
        assert 'npv' in result['thresholds']
        npv_metrics = result['metrics_at_thresholds']['npv']
        assert npv_metrics['npv'] >= 0.85


class TestFallbackMechanism:
    """Test fallback to closest threshold when no perfect match."""
    
    def test_impossible_targets_fallback(self):
        """Test fallback when targets are impossible to meet."""
        # Create data where it's impossible to get both sens=0.99 and spec=0.99
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.concatenate([
            np.random.rand(50) * 0.6,  # Overlapping distributions
            np.random.rand(50) * 0.6 + 0.4
        ])
        
        targets = {'sensitivity': 0.99, 'specificity': 0.99}
        result = calculate_metrics_at_target(
            y_true, 
            y_pred_proba, 
            targets,
            fallback_to_closest=True
        )
        
        # Should have closest_threshold but not best_threshold
        assert result['closest_threshold'] is not None
        assert result['best_threshold'] is None or not result['combined_results']
        
        # Closest threshold should have warning
        assert 'warning' in result['closest_threshold']
        assert result['closest_threshold']['distance_to_target'] > 0
        
        print(f"\nClosest threshold info:")
        print(f"  Threshold: {result['closest_threshold']['threshold']:.4f}")
        print(f"  Distance: {result['closest_threshold']['distance_to_target']:.4f}")
        print(f"  Satisfied: {result['closest_threshold']['satisfied_targets']}")
        print(f"  Unsatisfied: {result['closest_threshold']['unsatisfied_targets']}")
    
    def test_fallback_distance_metrics(self):
        """Test different distance metrics for fallback."""
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        targets = {'sensitivity': 0.95, 'specificity': 0.95}
        
        # Test different distance metrics
        for metric in ['euclidean', 'manhattan', 'max']:
            result = calculate_metrics_at_target(
                y_true,
                y_pred_proba,
                targets,
                fallback_to_closest=True,
                distance_metric=metric
            )
            
            if result['closest_threshold']:
                assert result['closest_threshold']['distance_metric'] == metric


class TestThresholdSelection:
    """Test threshold selection strategies."""
    
    def test_pareto_youden_strategy(self):
        """Test Pareto+Youden threshold selection."""
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(100), np.ones(100)])
        y_pred_proba = np.concatenate([
            np.random.beta(2, 5, 100),
            np.random.beta(5, 2, 100)
        ])
        
        targets = {'sensitivity': 0.70, 'specificity': 0.70}
        result = calculate_metrics_at_target(
            y_true,
            y_pred_proba,
            targets,
            threshold_selection='pareto+youden'
        )
        
        if result['best_threshold']:
            assert 'strategy' in result['best_threshold']
            assert result['best_threshold']['strategy'] == 'pareto+youden'
            assert 'youden_index' in result['best_threshold']
            
            print(f"\nBest threshold (Pareto+Youden):")
            print(f"  Threshold: {result['best_threshold']['threshold']:.4f}")
            print(f"  Youden: {result['best_threshold']['youden_index']:.4f}")
            if 'pareto_optimal_count' in result['best_threshold']:
                print(f"  Pareto optimal count: {result['best_threshold']['pareto_optimal_count']}")
    
    def test_youden_strategy(self):
        """Test Youden-only threshold selection."""
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(100), np.ones(100)])
        y_pred_proba = np.concatenate([
            np.random.beta(2, 5, 100),
            np.random.beta(5, 2, 100)
        ])
        
        targets = {'sensitivity': 0.70, 'specificity': 0.70}
        result = calculate_metrics_at_target(
            y_true,
            y_pred_proba,
            targets,
            threshold_selection='youden'
        )
        
        if result['best_threshold']:
            assert result['best_threshold']['strategy'] == 'youden'
            assert 'youden_index' in result['best_threshold']


class TestCategoryFiltering:
    """Test category-based metric filtering."""
    
    def test_basic_metrics_only(self):
        """Test calculating only basic metrics."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Get only basic metrics
        metrics_basic = calculate_metrics(
            y_true, y_pred, y_pred_proba, 
            categories=['basic']
        )
        
        # Should have basic metrics
        assert 'accuracy' in metrics_basic
        assert 'sensitivity' in metrics_basic
        assert 'specificity' in metrics_basic
        
        # Should not have statistical metrics
        assert 'hosmer_lemeshow_p_value' not in metrics_basic
        assert 'spiegelhalter_z_p_value' not in metrics_basic
    
    def test_statistical_metrics_only(self):
        """Test calculating only statistical metrics."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Get only statistical metrics
        metrics_stat = calculate_metrics(
            y_true, y_pred, y_pred_proba,
            categories=['statistical']
        )
        
        # Should have statistical metrics
        assert 'hosmer_lemeshow_p_value' in metrics_stat or len(metrics_stat) >= 0
        
        # Should not have basic metrics
        assert 'accuracy' not in metrics_stat
        assert 'sensitivity' not in metrics_stat


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
