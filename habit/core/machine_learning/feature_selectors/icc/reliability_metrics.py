"""
Reliability Metrics Module

This module provides a unified framework for calculating various reliability metrics
including Intraclass Correlation Coefficient (ICC), Cohen's Kappa, Fleiss' Kappa,
and other inter-rater reliability measures.

The module uses a registry pattern to allow easy extension with new metrics.

Available Metrics:
    - ICC Types: ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k (all 6 types from pingouin)
    - Kappa Metrics: Cohen's Kappa (2 raters), Fleiss' Kappa (multiple raters)
    - Weighted Kappa: For ordinal data

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""

import pandas as pd
import numpy as np
import pingouin as pg
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
import logging

from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)


# ==================== Enums for Metric Types ====================

class ICCType(Enum):
    """
    Enumeration of all ICC types as defined in pingouin.
    
    Reference: McGraw & Wong (1996), Shrout & Fleiss (1979)
    
    Attributes:
        ICC1: Single raters, absolute agreement, one-way random model
        ICC2: Single random raters, absolute agreement, two-way random model
        ICC3: Single fixed raters, consistency, two-way mixed model
        ICC1k: Average raters, absolute agreement, one-way random model
        ICC2k: Average random raters, absolute agreement, two-way random model
        ICC3k: Average fixed raters, consistency, two-way mixed model
    """
    ICC1 = 0      # Single raters, absolute agreement
    ICC2 = 1      # Single random raters, absolute agreement
    ICC3 = 2      # Single fixed raters, consistency (most commonly used)
    ICC1k = 3     # Average raters (k raters), absolute agreement
    ICC2k = 4     # Average random raters
    ICC3k = 5     # Average fixed raters
    
    @classmethod
    def get_description(cls, icc_type: 'ICCType') -> str:
        """
        Get human-readable description of ICC type.
        
        Args:
            icc_type: ICC type enum value
            
        Returns:
            String description of the ICC type
        """
        descriptions = {
            cls.ICC1: "ICC(1,1): Single raters, absolute agreement, one-way random model",
            cls.ICC2: "ICC(2,1): Single random raters, absolute agreement, two-way random model",
            cls.ICC3: "ICC(3,1): Single fixed raters, consistency, two-way mixed model",
            cls.ICC1k: "ICC(1,k): Average raters, absolute agreement, one-way random model",
            cls.ICC2k: "ICC(2,k): Average random raters, absolute agreement, two-way random model",
            cls.ICC3k: "ICC(3,k): Average fixed raters, consistency, two-way mixed model",
        }
        return descriptions.get(icc_type, "Unknown ICC type")


class KappaType(Enum):
    """
    Enumeration of Kappa coefficient types.
    
    Attributes:
        COHEN: Cohen's Kappa for 2 raters
        FLEISS: Fleiss' Kappa for multiple raters
        WEIGHTED_LINEAR: Weighted Kappa with linear weights
        WEIGHTED_QUADRATIC: Weighted Kappa with quadratic weights
    """
    COHEN = "cohen"
    FLEISS = "fleiss"
    WEIGHTED_LINEAR = "weighted_linear"
    WEIGHTED_QUADRATIC = "weighted_quadratic"


# ==================== Metric Result Container ====================

class MetricResult:
    """
    Container for reliability metric calculation results.
    
    Attributes:
        value: The primary metric value (e.g., ICC coefficient, Kappa value)
        ci95_lower: Lower bound of 95% confidence interval (if available)
        ci95_upper: Upper bound of 95% confidence interval (if available)
        p_value: P-value for statistical significance (if available)
        metric_type: String identifier of the metric type
        additional_info: Dictionary for any additional information
    """
    
    def __init__(
        self,
        value: float,
        ci95_lower: Optional[float] = None,
        ci95_upper: Optional[float] = None,
        p_value: Optional[float] = None,
        metric_type: str = "",
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a MetricResult instance.
        
        Args:
            value: Primary metric value
            ci95_lower: Lower bound of 95% CI
            ci95_upper: Upper bound of 95% CI
            p_value: Statistical p-value
            metric_type: String identifier for the metric
            additional_info: Additional metadata or results
        """
        self.value = value
        self.ci95_lower = ci95_lower
        self.ci95_upper = ci95_upper
        self.p_value = p_value
        self.metric_type = metric_type
        self.additional_info = additional_info or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.
        
        Returns:
            Dictionary representation of the result
        """
        def convert_value(val):
            """Convert numpy types to Python native types"""
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val) if not np.isnan(val) else None
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val
        
        result = {
            "value": convert_value(self.value),
            "metric_type": self.metric_type
        }
        if self.ci95_lower is not None and self.ci95_upper is not None:
            result["ci95"] = [convert_value(self.ci95_lower), convert_value(self.ci95_upper)]
        if self.p_value is not None:
            result["p_value"] = convert_value(self.p_value)
        if self.additional_info:
            result["additional_info"] = {k: convert_value(v) for k, v in self.additional_info.items()}
        return result
    
    def __repr__(self) -> str:
        """String representation of the result."""
        ci_str = f", CI95=[{self.ci95_lower:.3f}, {self.ci95_upper:.3f}]" if self.ci95_lower is not None else ""
        p_str = f", p={self.p_value:.4f}" if self.p_value is not None else ""
        return f"MetricResult({self.metric_type}={self.value:.4f}{ci_str}{p_str})"


# ==================== Base Class for Reliability Metrics ====================

class BaseReliabilityMetric(ABC):
    """
    Abstract base class for all reliability metrics.
    
    This class defines the interface that all reliability metrics must implement.
    It provides common functionality and ensures consistency across different
    metric implementations.
    
    Subclasses must implement:
        - name (property): Return the metric name
        - calculate(): Perform the actual metric calculation
        - validate_data(): Validate input data before calculation
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name identifier for this metric.
        
        Returns:
            String name of the metric
        """
        pass
    
    @property
    def description(self) -> str:
        """
        Return a human-readable description of the metric.
        
        Returns:
            Description string
        """
        return f"Reliability metric: {self.name}"
    
    @abstractmethod
    def calculate(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str,
        **kwargs
    ) -> MetricResult:
        """
        Calculate the reliability metric.
        
        Args:
            data: Long-format DataFrame with columns for targets, raters, and ratings
            targets: Column name for the target/subject identifiers
            raters: Column name for the rater/observer identifiers
            ratings: Column name for the rating values
            **kwargs: Additional parameters specific to each metric
            
        Returns:
            MetricResult containing the calculated value and statistics
        """
        pass
    
    @abstractmethod
    def validate_data(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str
    ) -> bool:
        """
        Validate that the input data is suitable for this metric.
        
        Args:
            data: DataFrame to validate
            targets: Column name for targets
            raters: Column name for raters
            ratings: Column name for ratings
            
        Returns:
            True if data is valid, raises exception otherwise
        """
        pass


# ==================== ICC Metrics ====================

class ICCMetric(BaseReliabilityMetric):
    """
    Intraclass Correlation Coefficient (ICC) calculator.
    
    Supports all 6 types of ICC as defined by McGraw & Wong (1996):
        - ICC1, ICC2, ICC3: Single rater metrics
        - ICC1k, ICC2k, ICC3k: Average rater metrics
    
    The most commonly used type is ICC(3,1) for test-retest reliability.
    
    Args:
        icc_type: Type of ICC to calculate (default: ICC3)
        nan_policy: How to handle NaN values ('omit', 'raise', or 'propagate')
    
    Example:
        >>> metric = ICCMetric(icc_type=ICCType.ICC3)
        >>> result = metric.calculate(data, 'subject', 'rater', 'score')
        >>> print(f"ICC = {result.value:.3f}")
    """
    
    def __init__(
        self,
        icc_type: ICCType = ICCType.ICC3,
        nan_policy: str = 'omit'
    ):
        """
        Initialize ICC metric calculator.
        
        Args:
            icc_type: Type of ICC to calculate
            nan_policy: Policy for handling NaN values
        """
        self.icc_type = icc_type
        self.nan_policy = nan_policy
    
    @property
    def name(self) -> str:
        """Return the ICC type name."""
        return self.icc_type.name
    
    @property
    def description(self) -> str:
        """Return ICC type description."""
        return ICCType.get_description(self.icc_type)
    
    def validate_data(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str
    ) -> bool:
        """
        Validate input data for ICC calculation.
        
        Args:
            data: DataFrame to validate
            targets: Column name for targets
            raters: Column name for raters
            ratings: Column name for ratings
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If data is invalid
        """
        # Check required columns exist
        for col in [targets, raters, ratings]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Check for minimum number of targets and raters
        n_targets = data[targets].nunique()
        n_raters = data[raters].nunique()
        
        if n_targets < 2:
            raise ValueError(f"ICC requires at least 2 targets, got {n_targets}")
        if n_raters < 2:
            raise ValueError(f"ICC requires at least 2 raters, got {n_raters}")
        
        # Check that ratings column is numeric
        if not pd.api.types.is_numeric_dtype(data[ratings]):
            raise ValueError(f"Ratings column '{ratings}' must be numeric")
        
        return True
    
    def calculate(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str,
        **kwargs
    ) -> MetricResult:
        """
        Calculate ICC using pingouin library.
        
        Args:
            data: Long-format DataFrame
            targets: Column name for target/subject identifiers
            raters: Column name for rater identifiers
            ratings: Column name for rating values
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            MetricResult with ICC value and confidence interval
        """
        # Validate data
        self.validate_data(data, targets, raters, ratings)
        
        # Log data info for debugging
        logger.info(f"Calculating ICC with data shape: {data.shape}")
        logger.info(f"Targets column: {targets}, unique values: {data[targets].nunique()}")
        logger.info(f"Raters column: {raters}, unique values: {data[raters].nunique()}")
        logger.info(f"Ratings column: {ratings}, type: {data[ratings].dtype}")
        logger.info(f"Ratings range: [{data[ratings].min()}, {data[ratings].max()}]")
        logger.info(f"NaN values in ratings: {data[ratings].isna().sum()}")
        
        # Additional diagnostic checks
        logger.info(f"Rating variance per target:")
        target_variances = data.groupby(targets)[ratings].var()
        logger.info(f"Target variances: {target_variances}")
        logger.info(f"Targets with zero variance: {(target_variances == 0).sum()}")
        
        logger.info(f"Number of ratings per target:")
        ratings_per_target = data.groupby(targets)[ratings].count()
        logger.info(f"Ratings per target - min: {ratings_per_target.min()}, max: {ratings_per_target.max()}, avg: {ratings_per_target.mean():.2f}")
        
        logger.info(f"Number of ratings per rater:")
        ratings_per_rater = data.groupby(raters)[ratings].count()
        logger.info(f"Ratings per rater - min: {ratings_per_rater.min()}, max: {ratings_per_rater.max()}, avg: {ratings_per_rater.mean():.2f}")
        
        # Check for balanced design
        pivot_table = data.pivot_table(index=targets, columns=raters, values=ratings, aggfunc='first')
        logger.info(f"Pivot table shape: {pivot_table.shape}")
        missing_data_pct = (pivot_table.isna().sum().sum() / (pivot_table.shape[0] * pivot_table.shape[1])) * 100
        logger.info(f"Missing data in pivot table: {missing_data_pct:.2f}%")
        
        # Calculate ICC using pingouin
        icc_result = pg.intraclass_corr(
            data=data,
            targets=targets,
            raters=raters,
            ratings=ratings,
            nan_policy=self.nan_policy
        )
        
        # Extract results for the specified ICC type
        row_idx = self.icc_type.value
        icc_value = icc_result.loc[row_idx, "ICC"]
        
        # Get confidence interval
        ci95 = icc_result.loc[row_idx, "CI95%"]
        ci_lower = ci95[0] if ci95 is not None else None
        ci_upper = ci95[1] if ci95 is not None else None
        
        # Get p-value
        p_value = icc_result.loc[row_idx, "pval"]
        
        return MetricResult(
            value=icc_value,
            ci95_lower=ci_lower,
            ci95_upper=ci_upper,
            p_value=p_value,
            metric_type=self.name,
            additional_info={
                "F": icc_result.loc[row_idx, "F"],
                "df1": icc_result.loc[row_idx, "df1"],
                "df2": icc_result.loc[row_idx, "df2"],
            }
        )


class MultiICCMetric(BaseReliabilityMetric):
    """
    Calculate multiple ICC types simultaneously.
    
    This is useful when you need to compare different ICC types or report
    multiple ICC values in a study.
    
    Args:
        icc_types: List of ICC types to calculate (default: all 6 types)
        nan_policy: How to handle NaN values
    
    Example:
        >>> metric = MultiICCMetric(icc_types=[ICCType.ICC2, ICCType.ICC3])
        >>> results = metric.calculate(data, 'subject', 'rater', 'score')
    """
    
    def __init__(
        self,
        icc_types: Optional[List[ICCType]] = None,
        nan_policy: str = 'omit'
    ):
        """
        Initialize multi-ICC metric calculator.
        
        Args:
            icc_types: List of ICC types to calculate (None = all types)
            nan_policy: Policy for handling NaN values
        """
        self.icc_types = icc_types or list(ICCType)
        self.nan_policy = nan_policy
    
    @property
    def name(self) -> str:
        """Return composite name of all ICC types."""
        return "MultiICC"
    
    @property
    def description(self) -> str:
        """Return description listing all ICC types."""
        types_str = ", ".join([t.name for t in self.icc_types])
        return f"Multiple ICC types: {types_str}"
    
    def validate_data(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str
    ) -> bool:
        """Validate data for ICC calculation."""
        # Use ICCMetric validation
        single_metric = ICCMetric(ICCType.ICC1, self.nan_policy)
        return single_metric.validate_data(data, targets, raters, ratings)
    
    def calculate(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Calculate all specified ICC types.
        
        Args:
            data: Long-format DataFrame
            targets: Column name for target identifiers
            raters: Column name for rater identifiers
            ratings: Column name for rating values
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping ICC type names to MetricResult objects
        """
        self.validate_data(data, targets, raters, ratings)
        
        # Log data info for debugging
        logger.info(f"Calculating multiple ICC types with data shape: {data.shape}")
        logger.info(f"Targets column: {targets}, unique values: {data[targets].nunique()}")
        logger.info(f"Raters column: {raters}, unique values: {data[raters].nunique()}")
        logger.info(f"Ratings column: {ratings}, type: {data[ratings].dtype}")
        logger.info(f"Ratings range: [{data[ratings].min()}, {data[ratings].max()}]")
        logger.info(f"NaN values in ratings: {data[ratings].isna().sum()}")
        
        # Additional diagnostic checks
        logger.info(f"Rating variance per target:")
        target_variances = data.groupby(targets)[ratings].var()
        logger.info(f"Target variances: {target_variances}")
        logger.info(f"Targets with zero variance: {(target_variances == 0).sum()}")
        
        logger.info(f"Number of ratings per target:")
        ratings_per_target = data.groupby(targets)[ratings].count()
        logger.info(f"Ratings per target - min: {ratings_per_target.min()}, max: {ratings_per_target.max()}, avg: {ratings_per_target.mean():.2f}")
        
        logger.info(f"Number of ratings per rater:")
        ratings_per_rater = data.groupby(raters)[ratings].count()
        logger.info(f"Ratings per rater - min: {ratings_per_rater.min()}, max: {ratings_per_rater.max()}, avg: {ratings_per_rater.mean():.2f}")
        
        # Check for balanced design
        pivot_table = data.pivot_table(index=targets, columns=raters, values=ratings, aggfunc='first')
        logger.info(f"Pivot table shape: {pivot_table.shape}")
        missing_data_pct = (pivot_table.isna().sum().sum() / (pivot_table.shape[0] * pivot_table.shape[1])) * 100
        logger.info(f"Missing data in pivot table: {missing_data_pct:.2f}%")
        
        # Calculate ICC once (pingouin returns all types)
        icc_result = pg.intraclass_corr(
            data=data,
            targets=targets,
            raters=raters,
            ratings=ratings,
            nan_policy=self.nan_policy
        )
        
        results = {}
        for icc_type in self.icc_types:
            row_idx = icc_type.value
            ci95 = icc_result.loc[row_idx, "CI95%"]
            
            results[icc_type.name] = MetricResult(
                value=icc_result.loc[row_idx, "ICC"],
                ci95_lower=ci95[0] if ci95 is not None else None,
                ci95_upper=ci95[1] if ci95 is not None else None,
                p_value=icc_result.loc[row_idx, "pval"],
                metric_type=icc_type.name,
                additional_info={
                    "F": icc_result.loc[row_idx, "F"],
                    "df1": icc_result.loc[row_idx, "df1"],
                    "df2": icc_result.loc[row_idx, "df2"],
                }
            )
        
        return results


# ==================== Kappa Metrics ====================

class CohenKappaMetric(BaseReliabilityMetric):
    """
    Cohen's Kappa coefficient calculator for inter-rater agreement.
    
    Cohen's Kappa is used when there are exactly 2 raters providing categorical
    ratings. It measures the agreement beyond what would be expected by chance.
    
    Interpretation:
        - < 0: Poor agreement (less than chance)
        - 0.01 - 0.20: Slight agreement
        - 0.21 - 0.40: Fair agreement
        - 0.41 - 0.60: Moderate agreement
        - 0.61 - 0.80: Substantial agreement
        - 0.81 - 1.00: Almost perfect agreement
    
    Args:
        weights: Weighting scheme for ordinal data
                 None = unweighted (nominal)
                 'linear' = linear weights
                 'quadratic' = quadratic weights
    
    Example:
        >>> metric = CohenKappaMetric(weights='quadratic')
        >>> result = metric.calculate_from_arrays(rater1_scores, rater2_scores)
    """
    
    def __init__(self, weights: Optional[str] = None):
        """
        Initialize Cohen's Kappa calculator.
        
        Args:
            weights: Weighting scheme ('linear', 'quadratic', or None)
        """
        self.weights = weights
    
    @property
    def name(self) -> str:
        """Return the metric name."""
        if self.weights:
            return f"CohenKappa_{self.weights}"
        return "CohenKappa"
    
    @property
    def description(self) -> str:
        """Return description of the metric."""
        weight_desc = f" with {self.weights} weights" if self.weights else ""
        return f"Cohen's Kappa coefficient{weight_desc} for 2-rater agreement"
    
    def validate_data(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str
    ) -> bool:
        """
        Validate data for Cohen's Kappa calculation.
        
        Args:
            data: DataFrame to validate
            targets: Column name for targets
            raters: Column name for raters
            ratings: Column name for ratings
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If data is invalid for Cohen's Kappa
        """
        for col in [targets, raters, ratings]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        n_raters = data[raters].nunique()
        if n_raters != 2:
            raise ValueError(
                f"Cohen's Kappa requires exactly 2 raters, got {n_raters}. "
                "Use Fleiss' Kappa for more than 2 raters."
            )
        
        return True
    
    def calculate(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str,
        **kwargs
    ) -> MetricResult:
        """
        Calculate Cohen's Kappa from long-format data.
        
        Args:
            data: Long-format DataFrame
            targets: Column name for target identifiers
            raters: Column name for rater identifiers
            ratings: Column name for rating values
            **kwargs: Additional parameters
            
        Returns:
            MetricResult with Kappa value
        """
        self.validate_data(data, targets, raters, ratings)
        
        # Pivot to wide format for Cohen's Kappa
        pivot_data = data.pivot(index=targets, columns=raters, values=ratings)
        rater_columns = pivot_data.columns.tolist()
        
        rater1_ratings = pivot_data[rater_columns[0]].values
        rater2_ratings = pivot_data[rater_columns[1]].values
        
        return self.calculate_from_arrays(rater1_ratings, rater2_ratings)
    
    def calculate_from_arrays(
        self,
        rater1: np.ndarray,
        rater2: np.ndarray
    ) -> MetricResult:
        """
        Calculate Cohen's Kappa from two arrays of ratings.
        
        Args:
            rater1: Array of ratings from first rater
            rater2: Array of ratings from second rater
            
        Returns:
            MetricResult with Kappa value and confidence interval
        """
        from sklearn.metrics import cohen_kappa_score
        
        # Remove NaN pairs
        mask = ~(np.isnan(rater1) | np.isnan(rater2))
        r1_clean = rater1[mask]
        r2_clean = rater2[mask]
        
        # Check if data is categorical (required for Cohen's Kappa)
        unique_r1 = len(np.unique(r1_clean))
        unique_r2 = len(np.unique(r2_clean))
        
        # Cohen's Kappa requires categorical data
        # If data has many unique values, it's likely continuous
        if unique_r1 > 10 or unique_r2 > 10:
            error_msg = (
                f"Cohen's Kappa requires categorical (discrete) data. "
                f"Found {unique_r1} unique values in rater1 and {unique_r2} unique values in rater2. "
                f"Please discretize continuous data before calculating Cohen's Kappa, "
                f"or use ICC for continuous data."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate Kappa
        kappa = cohen_kappa_score(r1_clean, r2_clean, weights=self.weights)
        
        # Calculate standard error and CI using bootstrap or analytical method
        n = len(r1_clean)
        se = self._calculate_standard_error(r1_clean, r2_clean, kappa)
        
        ci_lower = kappa - 1.96 * se if se is not None else None
        ci_upper = kappa + 1.96 * se if se is not None else None
        
        return MetricResult(
            value=kappa,
            ci95_lower=ci_lower,
            ci95_upper=ci_upper,
            p_value=None,  # P-value not directly available from sklearn
            metric_type=self.name,
            additional_info={"n_samples": n, "weights": self.weights}
        )
    
    def _calculate_standard_error(
        self,
        rater1: np.ndarray,
        rater2: np.ndarray,
        kappa: float
    ) -> Optional[float]:
        """
        Calculate approximate standard error for Kappa.
        
        Uses analytical approximation for unweighted Kappa.
        For weighted Kappa, bootstrap would be more accurate.
        
        Args:
            rater1: First rater's ratings
            rater2: Second rater's ratings
            kappa: Calculated Kappa value
            
        Returns:
            Standard error estimate
        """
        n = len(rater1)
        if n < 10:
            return None
            
        # Simple approximation: SE ≈ sqrt(2 / n) for moderate agreement
        # This is a rough estimate; more precise methods exist
        return np.sqrt(2.0 / n)


class FleissKappaMetric(BaseReliabilityMetric):
    """
    Fleiss' Kappa coefficient calculator for multiple raters.
    
    Fleiss' Kappa extends Cohen's Kappa to any number of raters (2 or more).
    It measures the reliability of agreement between a fixed number of raters
    when assigning categorical ratings to a number of items.
    
    Note: Fleiss' Kappa assumes that each item is rated by the same number
    of raters, though not necessarily the same raters for each item.
    
    Interpretation (same as Cohen's Kappa):
        - < 0: Poor agreement
        - 0.01 - 0.20: Slight agreement
        - 0.21 - 0.40: Fair agreement
        - 0.41 - 0.60: Moderate agreement
        - 0.61 - 0.80: Substantial agreement
        - 0.81 - 1.00: Almost perfect agreement
    
    Example:
        >>> metric = FleissKappaMetric()
        >>> result = metric.calculate(data, 'subject', 'rater', 'category')
    """
    
    @property
    def name(self) -> str:
        """Return the metric name."""
        return "FleissKappa"
    
    @property
    def description(self) -> str:
        """Return description of the metric."""
        return "Fleiss' Kappa coefficient for multi-rater agreement"
    
    def validate_data(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str
    ) -> bool:
        """
        Validate data for Fleiss' Kappa calculation.
        
        Args:
            data: DataFrame to validate
            targets: Column name for targets
            raters: Column name for raters
            ratings: Column name for ratings
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If data is invalid
        """
        for col in [targets, raters, ratings]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        n_raters = data[raters].nunique()
        if n_raters < 2:
            raise ValueError(f"Fleiss' Kappa requires at least 2 raters, got {n_raters}")
        
        n_targets = data[targets].nunique()
        if n_targets < 2:
            raise ValueError(f"Fleiss' Kappa requires at least 2 targets, got {n_targets}")
        
        return True
    
    def calculate(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str,
        **kwargs
    ) -> MetricResult:
        """
        Calculate Fleiss' Kappa from long-format data.
        
        Args:
            data: Long-format DataFrame
            targets: Column name for target identifiers
            raters: Column name for rater identifiers  
            ratings: Column name for categorical ratings
            **kwargs: Additional parameters
            
        Returns:
            MetricResult with Fleiss' Kappa value
        """
        self.validate_data(data, targets, raters, ratings)
        
        # Convert to format required for Fleiss' Kappa
        # Need a matrix where rows are subjects and columns are categories
        # Each cell contains the count of raters who assigned that category
        
        # Get unique categories
        categories = sorted(data[ratings].dropna().unique())
        n_categories = len(categories)
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        # Get unique targets
        unique_targets = data[targets].unique()
        n_targets = len(unique_targets)
        
        # Build rating matrix (n_targets x n_categories)
        rating_matrix = np.zeros((n_targets, n_categories), dtype=int)
        
        for target_idx, target in enumerate(unique_targets):
            target_data = data[data[targets] == target]
            for rating in target_data[ratings].dropna():
                if rating in cat_to_idx:
                    rating_matrix[target_idx, cat_to_idx[rating]] += 1
        
        # Calculate Fleiss' Kappa using the rating matrix
        kappa, var_kappa = self._calculate_fleiss_kappa(rating_matrix)
        
        # Calculate confidence interval
        se = np.sqrt(var_kappa) if var_kappa > 0 else None
        ci_lower = kappa - 1.96 * se if se is not None else None
        ci_upper = kappa + 1.96 * se if se is not None else None
        
        # Calculate z-score and p-value
        z_score = kappa / se if se is not None and se > 0 else None
        p_value = 2 * (1 - self._norm_cdf(abs(z_score))) if z_score is not None else None
        
        return MetricResult(
            value=kappa,
            ci95_lower=ci_lower,
            ci95_upper=ci_upper,
            p_value=p_value,
            metric_type=self.name,
            additional_info={
                "n_subjects": n_targets,
                "n_categories": n_categories,
                "n_raters": int(rating_matrix.sum() / n_targets),
                "categories": categories
            }
        )
    
    def _calculate_fleiss_kappa(
        self,
        rating_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Fleiss' Kappa from a rating matrix.
        
        Args:
            rating_matrix: Matrix of shape (n_subjects, n_categories)
                          where each cell is the count of raters
                          who assigned that category to that subject
        
        Returns:
            Tuple of (kappa_value, variance_estimate)
        """
        n, k = rating_matrix.shape  # n = subjects, k = categories
        N = rating_matrix.sum(axis=1).mean()  # Number of raters per subject (assumed constant)
        
        if N < 2:
            return np.nan, np.nan
        
        # Calculate proportion of all assignments to each category
        p_j = rating_matrix.sum(axis=0) / (n * N)
        
        # Calculate extent of agreement for each subject
        P_i = np.zeros(n)
        for i in range(n):
            n_ij = rating_matrix[i, :]
            P_i[i] = (np.sum(n_ij ** 2) - N) / (N * (N - 1))
        
        P_bar = P_i.mean()
        P_e_bar = np.sum(p_j ** 2)
        
        # Calculate Fleiss' Kappa
        if P_e_bar == 1:
            kappa = 1.0
        else:
            kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
        
        # Calculate variance (Fleiss et al., 1979)
        numerator = 2 * (1 - P_e_bar) ** 2
        denominator = n * N * (N - 1)
        var_kappa = numerator / denominator if denominator > 0 else 0
        
        return kappa, var_kappa
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Calculate standard normal CDF."""
        from scipy.stats import norm
        return norm.cdf(x)


# ==================== Krippendorff's Alpha ====================

class KrippendorffAlphaMetric(BaseReliabilityMetric):
    """
    Krippendorff's Alpha coefficient for inter-rater reliability.
    
    Krippendorff's Alpha is a versatile reliability measure that:
    - Works with any number of raters
    - Works with any level of measurement (nominal, ordinal, interval, ratio)
    - Can handle missing data
    - Accounts for the level of measurement in the data
    
    Args:
        level_of_measurement: Type of data ('nominal', 'ordinal', 'interval', 'ratio')
    
    Example:
        >>> metric = KrippendorffAlphaMetric(level_of_measurement='ordinal')
        >>> result = metric.calculate(data, 'subject', 'rater', 'score')
    """
    
    def __init__(self, level_of_measurement: str = 'interval'):
        """
        Initialize Krippendorff's Alpha calculator.
        
        Args:
            level_of_measurement: One of 'nominal', 'ordinal', 'interval', 'ratio'
        """
        valid_levels = ['nominal', 'ordinal', 'interval', 'ratio']
        if level_of_measurement not in valid_levels:
            raise ValueError(f"level_of_measurement must be one of {valid_levels}")
        self.level_of_measurement = level_of_measurement
    
    @property
    def name(self) -> str:
        """Return the metric name."""
        return f"KrippendorffAlpha_{self.level_of_measurement}"
    
    @property
    def description(self) -> str:
        """Return description of the metric."""
        return f"Krippendorff's Alpha for {self.level_of_measurement} data"
    
    def validate_data(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str
    ) -> bool:
        """Validate data for Krippendorff's Alpha calculation."""
        for col in [targets, raters, ratings]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        n_raters = data[raters].nunique()
        if n_raters < 2:
            raise ValueError(f"Krippendorff's Alpha requires at least 2 raters, got {n_raters}")
        
        return True
    
    def calculate(
        self,
        data: pd.DataFrame,
        targets: str,
        raters: str,
        ratings: str,
        **kwargs
    ) -> MetricResult:
        """
        Calculate Krippendorff's Alpha from long-format data.
        
        Args:
            data: Long-format DataFrame
            targets: Column name for target identifiers
            raters: Column name for rater identifiers
            ratings: Column name for rating values
            **kwargs: Additional parameters
            
        Returns:
            MetricResult with Alpha value
        """
        self.validate_data(data, targets, raters, ratings)
        
        # Pivot to wide format (raters x targets)
        pivot_data = data.pivot(index=raters, columns=targets, values=ratings)
        reliability_data = pivot_data.values
        
        # Calculate Krippendorff's Alpha
        alpha = self._calculate_alpha(reliability_data)
        
        return MetricResult(
            value=alpha,
            ci95_lower=None,  # Bootstrap CI would require additional computation
            ci95_upper=None,
            p_value=None,
            metric_type=self.name,
            additional_info={
                "level_of_measurement": self.level_of_measurement,
                "n_raters": data[raters].nunique(),
                "n_subjects": data[targets].nunique()
            }
        )
    
    def _calculate_alpha(self, reliability_data: np.ndarray) -> float:
        """
        Calculate Krippendorff's Alpha from a reliability matrix.
        
        Args:
            reliability_data: Matrix of shape (n_raters, n_units)
                             with NaN for missing values
        
        Returns:
            Alpha value
        """
        # This is a simplified implementation
        # For production use, consider using the krippendorff package
        try:
            import krippendorff
            return krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement=self.level_of_measurement
            )
        except ImportError:
            # Fallback to simplified calculation for interval data
            logger.warning(
                "krippendorff package not installed. "
                "Using simplified calculation for interval data."
            )
            return self._calculate_alpha_simplified(reliability_data)
    
    def _calculate_alpha_simplified(self, data: np.ndarray) -> float:
        """
        Simplified Alpha calculation for interval data.
        
        This is a fallback when the krippendorff package is not available.
        """
        # Mask for non-missing values
        valid = ~np.isnan(data)
        
        # Calculate observed disagreement
        n_units = data.shape[1]
        D_o = 0
        n_pairs = 0
        
        for u in range(n_units):
            unit_ratings = data[:, u]
            unit_valid = unit_ratings[~np.isnan(unit_ratings)]
            m_u = len(unit_valid)
            if m_u >= 2:
                for i in range(m_u):
                    for j in range(i + 1, m_u):
                        D_o += (unit_valid[i] - unit_valid[j]) ** 2
                        n_pairs += 1
        
        if n_pairs == 0:
            return np.nan
        
        D_o = D_o / n_pairs
        
        # Calculate expected disagreement
        all_valid = data[valid]
        n_total = len(all_valid)
        D_e = 0
        
        if n_total >= 2:
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    D_e += (all_valid[i] - all_valid[j]) ** 2
            D_e = D_e / (n_total * (n_total - 1) / 2)
        
        if D_e == 0:
            return 1.0
        
        return 1 - D_o / D_e


# ==================== Metric Registry ====================

# Registry to hold all available metrics
_RELIABILITY_METRICS: Dict[str, type] = {}


def register_metric(name: str) -> Callable:
    """
    Decorator for registering reliability metrics.
    
    Args:
        name: Unique name for the metric
        
    Returns:
        Decorator function
        
    Example:
        >>> @register_metric("my_custom_metric")
        ... class MyMetric(BaseReliabilityMetric):
        ...     pass
    """
    def decorator(cls: type) -> type:
        if not issubclass(cls, BaseReliabilityMetric):
            raise TypeError(f"{cls.__name__} must be a subclass of BaseReliabilityMetric")
        _RELIABILITY_METRICS[name] = cls
        return cls
    return decorator


def get_available_metrics() -> List[str]:
    """
    Get list of all registered metric names.
    
    Returns:
        List of metric names
    """
    return list(_RELIABILITY_METRICS.keys())


def get_metric(name: str, **kwargs) -> BaseReliabilityMetric:
    """
    Get an instance of a registered metric.
    
    Args:
        name: Name of the metric
        **kwargs: Parameters to pass to the metric constructor
        
    Returns:
        Metric instance
        
    Raises:
        ValueError: If metric name is not registered
    """
    if name not in _RELIABILITY_METRICS:
        available = ", ".join(get_available_metrics())
        raise ValueError(f"Unknown metric '{name}'. Available: {available}")
    
    return _RELIABILITY_METRICS[name](**kwargs)


def create_metric(
    metric_type: str,
    **kwargs
) -> BaseReliabilityMetric:
    """
    Factory function to create reliability metrics.
    
    This provides a convenient way to create metric instances by name.
    
    Args:
        metric_type: Type of metric to create:
            - "icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k": Individual ICC types
            - "multi_icc": All or selected ICC types
            - "cohen_kappa": Cohen's Kappa for 2 raters
            - "fleiss_kappa": Fleiss' Kappa for multiple raters
            - "krippendorff": Krippendorff's Alpha
        **kwargs: Additional parameters for the metric
        
    Returns:
        Configured metric instance
        
    Example:
        >>> metric = create_metric("icc3")
        >>> metric = create_metric("cohen_kappa", weights="quadratic")
        >>> metric = create_metric("multi_icc", icc_types=[ICCType.ICC2, ICCType.ICC3])
    """
    metric_type_lower = metric_type.lower()
    
    # ICC types
    icc_type_map = {
        "icc1": ICCType.ICC1,
        "icc2": ICCType.ICC2,
        "icc3": ICCType.ICC3,
        "icc1k": ICCType.ICC1k,
        "icc2k": ICCType.ICC2k,
        "icc3k": ICCType.ICC3k,
    }
    
    if metric_type_lower in icc_type_map:
        return ICCMetric(icc_type=icc_type_map[metric_type_lower], **kwargs)
    
    if metric_type_lower == "multi_icc":
        return MultiICCMetric(**kwargs)
    
    if metric_type_lower in ["cohen_kappa", "cohens_kappa", "cohen"]:
        return CohenKappaMetric(**kwargs)
    
    if metric_type_lower in ["fleiss_kappa", "fleiss"]:
        return FleissKappaMetric(**kwargs)
    
    if metric_type_lower in ["krippendorff", "krippendorff_alpha"]:
        return KrippendorffAlphaMetric(**kwargs)
    
    # Check registry for custom metrics
    if metric_type in _RELIABILITY_METRICS:
        return _RELIABILITY_METRICS[metric_type](**kwargs)
    
    available = list(icc_type_map.keys()) + [
        "multi_icc", "cohen_kappa", "fleiss_kappa", "krippendorff"
    ] + get_available_metrics()
    raise ValueError(f"Unknown metric type '{metric_type}'. Available: {available}")


# ==================== Convenience Functions ====================

def calculate_reliability(
    data: pd.DataFrame,
    targets: str,
    raters: str,
    ratings: str,
    metrics: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, MetricResult]:
    """
    Calculate multiple reliability metrics at once.
    
    This is a convenience function for calculating several metrics
    on the same data in one call.
    
    Args:
        data: Long-format DataFrame with columns for targets, raters, and ratings
        targets: Column name for target/subject identifiers
        raters: Column name for rater/observer identifiers
        ratings: Column name for rating values
        metrics: List of metric types to calculate (default: ["icc3"])
        **kwargs: Additional parameters passed to all metrics
        
    Returns:
        Dictionary mapping metric names to MetricResult objects
        
    Example:
        >>> results = calculate_reliability(
        ...     data, 'subject', 'rater', 'score',
        ...     metrics=['icc2', 'icc3', 'fleiss_kappa']
        ... )
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.value:.3f}")
    """
    if metrics is None:
        metrics = ["icc3"]
    
    results = {}
    for metric_name in metrics:
        try:
            metric = create_metric(metric_name, **kwargs)
            result = metric.calculate(data, targets, raters, ratings)
            
            # Handle MultiICCMetric which returns a dict
            if isinstance(result, dict):
                results.update(result)
            else:
                results[metric.name] = result
                
        except Exception as e:
            logger.warning(f"Error calculating {metric_name}: {e}")
            results[metric_name] = MetricResult(
                value=np.nan,
                metric_type=metric_name,
                additional_info={"error": str(e)}
            )
    
    return results


# Register built-in metrics
register_metric("icc")(ICCMetric)
register_metric("multi_icc")(MultiICCMetric)
register_metric("cohen_kappa")(CohenKappaMetric)
register_metric("fleiss_kappa")(FleissKappaMetric)
register_metric("krippendorff_alpha")(KrippendorffAlphaMetric)
