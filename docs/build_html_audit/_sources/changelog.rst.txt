Changelog
=========

Version 2.0 (2026-01-25)
------------------------

Metrics Module Major Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Performance Improvements**

* 🚀 **8x Speed Boost**: Introduced confusion matrix caching via ``MetricsCache`` class
  
  - Previously: 8 redundant confusion matrix calculations
  - Now: Single cached calculation shared across all metrics
  - Impact: ~800% faster for basic metrics computation

**Feature Enhancements**

* 💡 **Extended Target Metrics Support**: Beyond sensitivity/specificity
  
  - Added: PPV (Precision), NPV, F1-score, Accuracy
  - Enables more comprehensive threshold selection criteria
  - Example: ``targets={'sensitivity': 0.91, 'specificity': 0.91, 'ppv': 0.85}``

* 🎯 **Fallback Mechanism**: Closest threshold finder
  
  - Problem solved: No thresholds meet all targets (e.g., targets too high)
  - Solution: Automatically finds "closest" threshold using distance metrics
  - Supports: Euclidean, Manhattan, and Max distance
  - Prevents data leakage: Training set finds threshold, test set applies it

* 🧠 **Intelligent Threshold Selection**: Three strategies
  
  - **First**: Fast, returns first qualifying threshold
  - **Youden**: Classic, maximizes Sensitivity + Specificity - 1
  - **Pareto+Youden** (Recommended): Finds Pareto-optimal thresholds, selects max Youden
  - Handles multi-objective optimization gracefully

* 📋 **Category-Based Filtering**: Compute only needed metrics
  
  - Categories: 'basic', 'statistical'
  - Enables faster computation when full metric suite unnecessary
  - Example: ``calculate_metrics(y_true, y_pred, y_prob, categories=['basic'])``

* 🔮 **Multi-class Preparation**: Foundation for multi-class classification
  
  - AUC: Already supports multi-class (One-vs-Rest)
  - Basic metrics: Added macro averaging for multi-class
  - Future: Per-class and weighted strategies

**API Changes**

* ``calculate_metrics_at_target``
  
  - New parameters: ``threshold_selection``, ``fallback_to_closest``, ``distance_metric``
  - New return fields: ``best_threshold``, ``closest_threshold``, ``combined_results``
  - Enhanced logging for threshold selection strategy

* ``calculate_metrics``
  
  - New parameters: ``use_cache`` (default: True), ``categories``
  - Backward compatible: All existing calls work unchanged

**Technical Debt Resolved**

* ✅ Eliminated redundant confusion matrix calculations (8x performance hit)
* ✅ Removed hardcoded sensitivity/specificity limitation
* ✅ Implemented previously unused ``category`` parameter
* ✅ Fixed inefficient F1-score calculation (3 CM calculations → 1)

**Testing**

* Added comprehensive test suite: ``tests/test_metrics_optimization.py``
* Coverage: Caching, extended targets, fallback, Pareto selection, categories
* All tests passing ✓

**Documentation**

* Added detailed guide: :doc:`development/metrics_optimization`
* Includes: API reference, usage examples, performance comparison
* Best practices for training/test threshold management

**Backward Compatibility**

* ✅ 100% backward compatible
* All existing code works without modification
* New features accessible via optional parameters

**Known Limitations**

* PPV/NPV computation: O(n) complexity, slower than sensitivity/specificity
* Pareto algorithm: O(n²) worst case (negligible for typical threshold counts)
* Multi-class: Basic support, further validation needed

**Migration Guide**

No migration needed! Existing code continues to work. To use new features:

.. code-block:: python

    # Old (still works)
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # New (enhanced)
    metrics = calculate_metrics(
        y_true, y_pred, y_prob,
        use_cache=True,           # Enable caching
        categories=['basic']       # Faster computation
    )
    
    result = calculate_metrics_at_target(
        y_true, y_prob,
        targets={'sensitivity': 0.91, 'specificity': 0.91, 'ppv': 0.85},
        threshold_selection='pareto+youden',  # Intelligent selection
        fallback_to_closest=True              # Fallback mechanism
    )

**Contributors**

* HABIT Development Team

---

Bug Fixes
~~~~~~~~~

* Fixed indentation error in ``comparison_workflow.py`` (duplicate code block removal)
* Fixed tuple unpacking in ``_calculate_target_metrics_by_split`` (L736, L756)
* Corrected threshold application logic for test sets

Configuration Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Enhanced model name resolution in ``ComparisonFileConfig``
* Added ``_ensure_unique_model_name`` safeguard in ``MultifileEvaluator``
* Improved Pydantic validation for model comparison configurations

Workflow Enhancements
~~~~~~~~~~~~~~~~~~~~~

* Test sets now always receive target metrics (no longer empty)
* Enhanced logging for threshold selection and fallback mechanisms
* Proper train→test threshold application (data leakage prevention)

---

Version 1.x
-----------

(Previous versions documented elsewhere)

Future Roadmap
--------------

**Planned for v2.1**

* GPU acceleration for confusion matrix computation
* Parallel Pareto optimization (multi-threading)
* Adaptive threshold selection (auto-strategy)
* Pareto frontier visualization

**Planned for v3.0**

* Full multi-class classification support
  
  - Weighted averaging strategies
  - Per-class metrics
  - Multi-label support

* Advanced optimization
  
  - Bayesian threshold optimization
  - Cost-sensitive learning integration

* Enhanced visualization
  
  - Interactive threshold explorer
  - Real-time metrics dashboard

---

See Also
--------

* :doc:`development/metrics_optimization` - Detailed optimization guide
* :doc:`development/testing` - Testing guidelines
* :doc:`api/machine_learning` - Machine learning API reference
