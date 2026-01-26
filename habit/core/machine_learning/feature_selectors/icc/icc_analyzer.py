"""
Unified ICC Analyzer Module

This module provides a comprehensive framework for calculating and analyzing
inter-rater reliability metrics, including Intraclass Correlation Coefficient (ICC),
Cohen's Kappa, Fleiss' Kappa, and Krippendorff's Alpha.

It integrates a robust, object-oriented metric calculation engine with a
high-level analyzer that can process multiple files and features.

Key Components:
- Metric Classes (e.g., ICCMetric, CohenKappaMetric): Perform actual calculations.
- MetricResult: A data container for rich metric results (value, CI, p-value).
- analyze_features: High-level function to run analysis across files and features.

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum

import pingouin as pg
from habit.utils.log_utils import get_module_logger

# ==================== Setup Logger ====================
logger = get_module_logger(__name__)


# ==================== Data Processing Helpers ====================

def load_and_merge_data(file_paths: List[str]) -> Tuple[List[pd.DataFrame], List[str]]:
    """Load multiple CSV or Excel files and return a list of DataFrames."""
    data_frames = []
    file_names = []
    for file_path in file_paths:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if p.suffix == '.csv':
            df = pd.read_csv(file_path, index_col=0)
        elif p.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, index_col=0)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")
        
        data_frames.append(df)
        file_names.append(p.stem)
    return data_frames, file_names

def find_common_indices(data_frames: List[pd.DataFrame]) -> pd.Index:
    """Find common indices among a list of DataFrames."""
    if not data_frames:
        return pd.Index([])
    
    common_index = data_frames[0].index
    for df in data_frames[1:]:
        common_index = common_index.intersection(df.index)
    return common_index

def find_common_columns(data_frames: List[pd.DataFrame]) -> List[str]:
    """Find common columns among a list of DataFrames."""
    if not data_frames:
        return []
        
    common_cols = set(data_frames[0].columns)
    for df in data_frames[1:]:
        common_cols.intersection_update(df.columns)
    return sorted(list(common_cols))

def prepare_long_format(data_frames: List[pd.DataFrame], feature_name: str, common_index: pd.Index, file_names: List[str]) -> pd.DataFrame:
    """Prepare long-format DataFrame for a specific feature."""
    long_dfs = []
    for i, df in enumerate(data_frames):
        rater_name = file_names[i]
        temp_df = df.loc[common_index, [feature_name]].copy()
        temp_df.rename(columns={feature_name: 'value'}, inplace=True)
        temp_df['reader'] = rater_name
        temp_df['target'] = temp_df.index
        long_dfs.append(temp_df)
    return pd.concat(long_dfs, ignore_index=True)


# ==================== Enums for Metric Types ====================

class ICCType(Enum):
    """Enumeration of all ICC types as defined in pingouin."""
    ICC1 = 0
    ICC2 = 1
    ICC3 = 2
    ICC1k = 3
    ICC2k = 4
    ICC3k = 5
    
    @classmethod
    def get_description(cls, icc_type: 'ICCType') -> str:
        descriptions = {
            cls.ICC1: "ICC(1,1): Single raters, absolute agreement, one-way random model",
            cls.ICC2: "ICC(2,1): Single random raters, absolute agreement, two-way random model",
            cls.ICC3: "ICC(3,1): Single fixed raters, consistency, two-way mixed model",
            cls.ICC1k: "ICC(1,k): Average raters, absolute agreement, one-way random model",
            cls.ICC2k: "ICC(2,k): Average random raters, absolute agreement, two-way random model",
            cls.ICC3k: "ICC(3,k): Average fixed raters, consistency, two-way mixed model",
        }
        return descriptions.get(icc_type, "Unknown ICC type")

# ==================== Metric Result Container ====================

class MetricResult:
    """Container for reliability metric calculation results."""
    def __init__(
        self,
        value: float,
        ci95_lower: Optional[float] = None,
        ci95_upper: Optional[float] = None,
        p_value: Optional[float] = None,
        metric_type: str = "",
        additional_info: Optional[Dict[str, Any]] = None
    ):
        self.value = value
        self.ci95_lower = ci95_lower
        self.ci95_upper = ci95_upper
        self.p_value = p_value
        self.metric_type = metric_type
        self.additional_info = additional_info or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        def convert_value(val):
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
        ci_str = f", CI95=[{self.ci95_lower:.3f}, {self.ci95_upper:.3f}]" if self.ci95_lower is not None else ""
        p_str = f", p={self.p_value:.4f}" if self.p_value is not None else ""
        return f"MetricResult({self.metric_type}={self.value:.4f}{ci_str}{p_str})"

# ==================== Base Class for Reliability Metrics ====================

class BaseReliabilityMetric(ABC):
    """Abstract base class for all reliability metrics."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, targets: str, raters: str, ratings: str, **kwargs) -> Union[MetricResult, Dict[str, MetricResult]]:
        pass

    def validate_data(self, data: pd.DataFrame, targets: str, raters: str, ratings: str) -> bool:
        for col in [targets, raters, ratings]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        if data[targets].nunique() < 2:
            raise ValueError(f"Reliability metrics require at least 2 targets, got {data[targets].nunique()}")
        if data[raters].nunique() < 2:
            raise ValueError(f"Reliability metrics require at least 2 raters, got {data[raters].nunique()}")
        return True

# ==================== Metric Implementations ====================

class ICCMetric(BaseReliabilityMetric):
    """Intraclass Correlation Coefficient (ICC) calculator."""
    def __init__(self, icc_type: ICCType = ICCType.ICC3, nan_policy: str = 'omit'):
        self.icc_type = icc_type
        self.nan_policy = nan_policy
    
    @property
    def name(self) -> str:
        return self.icc_type.name
    
    def validate_data(self, data: pd.DataFrame, targets: str, raters: str, ratings: str) -> bool:
        super().validate_data(data, targets, raters, ratings)
        if not pd.api.types.is_numeric_dtype(data[ratings]):
            raise ValueError(f"Ratings column '{ratings}' must be numeric for ICC")
        return True

    def calculate(self, data: pd.DataFrame, targets: str, raters: str, ratings: str, **kwargs) -> MetricResult:
        self.validate_data(data, targets, raters, ratings)
        
        icc_result = pg.intraclass_corr(
            data=data, targets=targets, raters=raters, ratings=ratings, nan_policy=self.nan_policy
        ).set_index('Type')
        
        icc_name = self.icc_type.name
        if icc_name not in icc_result.index:
             raise ValueError(f"ICC type '{icc_name}' not found in pingouin results.")

        icc_series = icc_result.loc[icc_name]
        
        ci95 = icc_series["CI95%"]
        
        return MetricResult(
            value=icc_series["ICC"],
            ci95_lower=ci95[0] if ci95 is not None else None,
            ci95_upper=ci95[1] if ci95 is not None else None,
            p_value=icc_series["pval"],
            metric_type=self.name,
            additional_info={
                "F": icc_series["F"],
                "df1": icc_series["df1"],
                "df2": icc_series["df2"],
            }
        )

class MultiICCMetric(BaseReliabilityMetric):
    """Calculate multiple ICC types simultaneously."""
    def __init__(self, icc_types: Optional[List[ICCType]] = None, nan_policy: str = 'omit'):
        self.icc_types = icc_types or list(ICCType)
        self.nan_policy = nan_policy
    
    @property
    def name(self) -> str:
        return "MultiICC"

    def calculate(self, data: pd.DataFrame, targets: str, raters: str, ratings: str, **kwargs) -> Dict[str, MetricResult]:
        ICCMetric().validate_data(data, targets, raters, ratings)
        
        icc_result_df = pg.intraclass_corr(
            data=data, targets=targets, raters=raters, ratings=ratings, nan_policy=self.nan_policy
        ).set_index('Type')
        
        results = {}
        for icc_type in self.icc_types:
            icc_name = icc_type.name
            if icc_name in icc_result_df.index:
                icc_series = icc_result_df.loc[icc_name]
                ci95 = icc_series["CI95%"]
                
                results[icc_name] = MetricResult(
                    value=icc_series["ICC"],
                    ci95_lower=ci95[0] if ci95 is not None else None,
                    ci95_upper=ci95[1] if ci95 is not None else None,
                    p_value=icc_series["pval"],
                    metric_type=icc_name,
                    additional_info={
                        "F": icc_series["F"],
                        "df1": icc_series["df1"],
                        "df2": icc_series["df2"],
                    }
                )
        return results

class CohenKappaMetric(BaseReliabilityMetric):
    """Cohen's Kappa for 2-rater agreement on categorical data."""
    def __init__(self, weights: Optional[str] = None):
        self.weights = weights

    @property
    def name(self) -> str:
        return f"CohenKappa_{self.weights}" if self.weights else "CohenKappa"

    def validate_data(self, data: pd.DataFrame, targets: str, raters: str, ratings: str) -> bool:
        super().validate_data(data, targets, raters, ratings)
        n_raters = data[raters].nunique()
        if n_raters != 2:
            raise ValueError(f"Cohen's Kappa requires exactly 2 raters, but got {n_raters}")
        return True

    def calculate(self, data: pd.DataFrame, targets: str, raters: str, ratings: str, **kwargs) -> MetricResult:
        from sklearn.metrics import cohen_kappa_score
        self.validate_data(data, targets, raters, ratings)

        pivot_data = data.pivot(index=targets, columns=raters, values=ratings)
        rater1, rater2 = pivot_data.iloc[:, 0], pivot_data.iloc[:, 1]

        mask = ~pd.isna(rater1) & ~pd.isna(rater2)
        r1_clean, r2_clean = rater1[mask], rater2[mask]

        if len(r1_clean) == 0:
            return MetricResult(value=np.nan, metric_type=self.name, additional_info={"error": "No overlapping data"})

        kappa = cohen_kappa_score(r1_clean, r2_clean, weights=self.weights)
        
        # CI calculation for Kappa is complex, so we omit it for now
        return MetricResult(value=kappa, metric_type=self.name)

class FleissKappaMetric(BaseReliabilityMetric):
    """Fleiss' Kappa for multi-rater agreement on categorical data."""
    @property
    def name(self) -> str:
        return "FleissKappa"

    def validate_data(self, data: pd.DataFrame, targets: str, raters: str, ratings: str) -> bool:
        super().validate_data(data, targets, raters, ratings)
        if data[ratings].nunique() > 50: # Heuristic for categorical
             logger.warning("Fleiss' Kappa is intended for categorical data. Found >50 unique values.")
        return True

    def calculate(self, data: pd.DataFrame, targets: str, raters: str, ratings: str, **kwargs) -> MetricResult:
        self.validate_data(data, targets, raters, ratings)
        
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa
            from scipy.stats import norm

            # Create a contingency table (subjects x categories)
            ratings_matrix = pd.crosstab(data[targets], data[ratings]).values
            
            # The number of raters per subject must be consistent
            n_raters_per_subject = ratings_matrix.sum(axis=1)
            if not np.all(n_raters_per_subject == n_raters_per_subject[0]):
                error_msg = "Fleiss' Kappa requires an equal number of raters for each subject."
                logger.error(error_msg)
                return MetricResult(value=np.nan, metric_type=self.name, additional_info={"error": error_msg})
            
            n = n_raters_per_subject[0] # Number of raters per subject
            N = ratings_matrix.shape[0] # Number of subjects
            k = ratings_matrix.shape[1] # Number of categories

            if n < 2 or N < 2:
                return MetricResult(value=np.nan, metric_type=self.name, additional_info={"error": "Not enough data"})

            # Calculate Fleiss' Kappa
            kappa = fleiss_kappa(ratings_matrix, method='fleiss')

            # Calculate additional statistics based on user's snippet
            p_j = np.sum(ratings_matrix, axis=0) / (N * n)  # Proportion of each category
            P_i = (np.sum(ratings_matrix**2, axis=1) - n) / (n * (n - 1)) # Agreement for each subject
            
            P_bar = np.mean(P_i) # Mean agreement
            P_e_bar = np.sum(p_j**2) # Expected agreement

            # Standard Error calculation
            var_num = 2 * (np.sum(p_j**2) - np.sum(p_j**3))**2 + \
                        (P_e_bar - 4 * np.sum(p_j**3) + 3 * P_e_bar**2) * (1 - P_e_bar)**2
            var_den = (N * n * (n - 1) * (1 - P_e_bar)**2)
            
            # A simpler approximation is often used, let's use the one from the user's snippet's logic
            # which seems to be a variant of SE for Cohen's Kappa applied to Fleiss'
            # Let's use a more standard formula for Fleiss' Kappa SE for robustness.
            # Fleiss, J. L., Levin, B., & Paik, M. C. (2003). Statistical Methods for Rates and Proportions.
            var_0 = (2 / (N * n * (n - 1))) * (P_e_bar - (2*n - 3)*P_e_bar**2 + 2*(n-2)*np.sum(p_j**3))
            se_kappa = np.sqrt(var_0) if var_0 > 0 else 0

            # Z-value and p-value
            z_kappa = kappa / se_kappa if se_kappa > 0 else np.inf
            p_value = 2 * (1 - norm.cdf(np.abs(z_kappa)))

            # 95% CI
            ci_lower = kappa - 1.96 * se_kappa
            ci_upper = kappa + 1.96 * se_kappa
            
            return MetricResult(
                value=kappa,
                ci95_lower=ci_lower,
                ci95_upper=ci_upper,
                p_value=p_value,
                metric_type=self.name,
                additional_info={
                    "z_value": z_kappa,
                    "se": se_kappa
                }
            )
        except ImportError:
            error_msg = "statsmodels is not installed. Please install it via 'pip install statsmodels'."
            logger.error(error_msg)
            return MetricResult(value=np.nan, metric_type=self.name, additional_info={"error": error_msg})
        except Exception as e:
            logger.error(f"Error calculating Fleiss' Kappa: {e}", exc_info=True)
            return MetricResult(value=np.nan, metric_type=self.name, additional_info={"error": str(e)})


class KrippendorffAlphaMetric(BaseReliabilityMetric):
    """Krippendorff's Alpha for various data types."""
    def __init__(self, level_of_measurement: str = 'interval'):
        self.level_of_measurement = level_of_measurement

    @property
    def name(self) -> str:
        return f"KrippendorffAlpha_{self.level_of_measurement}"

    def calculate(self, data: pd.DataFrame, targets: str, raters: str, ratings: str, **kwargs) -> MetricResult:
        self.validate_data(data, targets, raters, ratings)
        
        try:
            import krippendorff
            
            # Pivot to (raters x subjects) format
            pivot_data = data.pivot(index=raters, columns=targets, values=ratings)
            
            alpha = krippendorff.alpha(
                reliability_data=pivot_data.values,
                level_of_measurement=self.level_of_measurement
            )
            return MetricResult(value=alpha, metric_type=self.name)
        except ImportError:
            error_msg = "krippendorff package is not installed. Please install it via 'pip install krippendorff'."
            logger.error(error_msg)
            return MetricResult(value=np.nan, metric_type=self.name, additional_info={"error": error_msg})
        except Exception as e:
            return MetricResult(value=np.nan, metric_type=self.name, additional_info={"error": str(e)})


# ==================== Metric Factory ====================

def create_metric(metric_type: str, **kwargs) -> BaseReliabilityMetric:
    """Factory function to create reliability metrics."""
    metric_type_lower = metric_type.lower()
    
    icc_type_map = {
        "icc1": ICCType.ICC1, "icc2": ICCType.ICC2, "icc3": ICCType.ICC3,
        "icc1k": ICCType.ICC1k, "icc2k": ICCType.ICC2k, "icc3k": ICCType.ICC3k,
    }
    
    if metric_type_lower in icc_type_map:
        return ICCMetric(icc_type=icc_type_map[metric_type_lower], **kwargs)
    
    if metric_type_lower in ["all_icc", "multi_icc"]:
        return MultiICCMetric(**kwargs)
    
    if metric_type_lower in ["cohen", "cohen_kappa"]:
        return CohenKappaMetric(**kwargs)

    if metric_type_lower in ["fleiss", "fleiss_kappa"]:
        return FleissKappaMetric(**kwargs)

    if metric_type_lower in ["krippendorff", "krippendorff_alpha"]:
        return KrippendorffAlphaMetric(**kwargs)
    
    available = list(icc_type_map.keys()) + ["all_icc", "cohen_kappa", "fleiss_kappa", "krippendorff_alpha"]
    raise ValueError(f"Unknown metric type '{metric_type}'. Available: {', '.join(available)}")

# ==================== High-Level Analyzer ====================

def analyze_features(
    file_paths: List[str],
    metrics: Optional[List[str]] = None,
    selected_features: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze reliability metrics for features across multiple files.
    """
    metrics = metrics or ['icc2', 'icc3']
    
    logger.info(f"Loading {len(file_paths)} files...")
    data_frames, file_names = load_and_merge_data(file_paths)
    
    group_name = "_vs_".join(file_names)
    logger.info(f"Analyzing group: {group_name}")
    
    common_index = find_common_indices(data_frames)
    if len(common_index) == 0:
        logger.error("No common indices found. Aborting analysis.")
        return {group_name: {}}
        
    common_columns = find_common_columns(data_frames)
    if selected_features:
        common_columns = [col for col in common_columns if col in selected_features]
    
    if not common_columns:
        logger.error("No common features to analyze. Aborting.")
        return {group_name: {}}

    logger.info(f"Found {len(common_index)} common samples and {len(common_columns)} common features.")

    for i in range(len(data_frames)):
        data_frames[i] = data_frames[i].loc[common_index]

    feature_results = {}
    total = len(common_columns)
    
    for i, feature_name in enumerate(common_columns):
        try:
            long_data = prepare_long_format(data_frames, feature_name, common_index, file_names)
            
            ft_results = {}
            for metric_name in metrics:
                try:
                    calculator = create_metric(metric_name)
                    result = calculator.calculate(long_data, 'target', 'reader', 'value')
                    
                    if isinstance(result, dict): # For MultiICCMetric
                        for k, v in result.items():
                            ft_results[k] = v.to_dict()
                    else:
                        ft_results[calculator.name] = result.to_dict()

                except Exception as e:
                    logger.warning(f"Could not calculate metric '{metric_name}' for feature '{feature_name}': {e}")
                    ft_results[metric_name] = {"error": str(e)}

            feature_results[feature_name] = ft_results
            
            if (i + 1) % 20 == 0 or (i + 1) == total:
                logger.info(f"Progress: {(i + 1) / total:.1%}")

        except Exception as e:
            logger.error(f"Error analyzing feature '{feature_name}': {e}")
            feature_results[feature_name] = {"error": str(e)}
            
    logger.info("Analysis complete.")
    return {group_name: feature_results}

def save_results(results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """Save analysis results to a JSON file."""
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results successfully saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise

def _is_nan(value) -> bool:
    """Helper to check if a value is NaN."""
    import math
    return isinstance(value, float) and math.isnan(value)

def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Prints a brief summary of the analysis results."""
    for group_name, group_results in results.items():
        logger.info(f"\nGroup: {group_name}")
        
        total_features = len(group_results)
        valid_features = sum(
            1 for ft_results in group_results.values()
            if ft_results and not ft_results.get("error")
        )
        
        logger.info(f"  - Total features analyzed: {total_features}")
        logger.info(f"  - Features with valid results: {valid_features}")
        
        # Print sample results
        for i, (feature_name, ft_results) in enumerate(group_results.items()):
            if i >= 3:
                break
            logger.info(f"  - Sample feature '{feature_name}': {ft_results}")
        
        if total_features > 3:
            logger.info(f"  ... and {total_features - 3} more features.")

def print_statistics(results: Dict[str, Dict[str, Any]]) -> None:
    """Calculates and prints detailed statistics of the results."""
    logger.info("\n--- Overall Statistics ---")
    total_features = 0
    good_icc_features = set()
    
    for group_name, features in results.items():
        logger.info(f"Stats for group: {group_name}")
        group_total = len(features)
        total_features += group_total
        
        group_good_icc = set()
        for feature_name, ft_results in features.items():
            if not isinstance(ft_results, dict):
                continue
            
            for metric_name, result_dict in ft_results.items():
                if isinstance(result_dict, dict) and 'value' in result_dict:
                    value = result_dict['value']
                    if value is not None and not _is_nan(value):
                        if metric_name.upper().startswith('ICC') and value >= 0.75:
                            group_good_icc.add(feature_name)
                            good_icc_features.add(feature_name)

        logger.info(f"  - Features with good ICC (>= 0.75): {len(group_good_icc)} / {group_total}")

    logger.info("------------------------")
    logger.info(f"Total unique features with good ICC (>= 0.75) across all groups: {len(good_icc_features)}")
    logger.info("--- End of Statistics ---\n")