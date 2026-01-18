"""
Simple ICC Calculator Module

Simplified implementation for calculating reliability metrics.
All functions return native Python types for easy JSON serialization.
"""

import pandas as pd
import numpy as np
import pingouin as pg
from sklearn.metrics import cohen_kappa_score
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_all_icc(data: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate all 6 ICC types from long-format data.
    
    Args:
        data: Long-format DataFrame with columns: 'target', 'reader', 'value'
        
    Returns:
        Dictionary with ICC values as native Python types
        Format: {'icc1': float, 'icc2': float, ...}
        Returns None for metrics that cannot be calculated
    """
    try:
        result = pg.intraclass_corr(
            data=data,
            targets='target',
            raters='reader',
            ratings='value',
            nan_policy='omit'
        )
        
        return {
            'icc1': _safe_float(result.loc[0, 'ICC']),
            'icc2': _safe_float(result.loc[1, 'ICC']),
            'icc3': _safe_float(result.loc[2, 'ICC']),
            'icc1k': _safe_float(result.loc[3, 'ICC']),
            'icc2k': _safe_float(result.loc[4, 'ICC']),
            'icc3k': _safe_float(result.loc[5, 'ICC'])
        }
    except Exception as e:
        logger.error(f"Error calculating ICC: {e}")
        return {key: None for key in ['icc1', 'icc2', 'icc3', 'icc1k', 'icc2k', 'icc3k']}


def calculate_cohen_kappa(data: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate Cohen's Kappa for 2 raters.
    
    Args:
        data: Long-format DataFrame with columns: 'target', 'reader', 'value'
        
    Returns:
        Dictionary with Kappa value or None if data is continuous
    """
    try:
        # Pivot to wide format
        pivot_data = data.pivot(index='target', columns='reader', values='value')
        
        if pivot_data.shape[1] != 2:
            return {'cohen': None, 'error': 'Cohen\'s Kappa requires exactly 2 raters'}
        
        rater1 = pivot_data.iloc[:, 0].values
        rater2 = pivot_data.iloc[:, 1].values
        
        # Remove NaN pairs
        mask = ~(np.isnan(rater1) | np.isnan(rater2))
        r1_clean = rater1[mask]
        r2_clean = rater2[mask]
        
        # Check if data is categorical
        unique_r1 = len(np.unique(r1_clean))
        unique_r2 = len(np.unique(r2_clean))
        
        if unique_r1 > 10 or unique_r2 > 10:
            return {
                'cohen': None,
                'error': f'Data is continuous ({unique_r1} and {unique_r2} unique values), Cohen\'s Kappa requires categorical data'
            }
        
        # Calculate Kappa
        kappa = cohen_kappa_score(r1_clean, r2_clean)
        return {'cohen': _safe_float(kappa)}
        
    except Exception as e:
        logger.error(f"Error calculating Cohen's Kappa: {e}")
        return {'cohen': None, 'error': str(e)}


def calculate_fleiss_kappa(data: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate Fleiss' Kappa for multiple raters.
    
    Args:
        data: Long-format DataFrame with columns: 'target', 'reader', 'value'
        
    Returns:
        Dictionary with Kappa value or None if data is continuous
    """
    try:
        # Pivot to wide format (raters x targets)
        pivot_data = data.pivot(index='target', columns='reader', values='value')
        
        # Check if data is categorical
        unique_values = data['value'].nunique()
        if unique_values > 10:
            return {
                'fleiss': None,
                'error': f'Data is continuous ({unique_values} unique values), Fleiss\' Kappa requires categorical data'
            }
        
        # Use pingouin's fleiss_kappa
        from scipy.stats import contingency
        kappa = pg.fleiss_kappa(data, rating_col='value', subject_col='target', rater_col='reader')
        return {'fleiss': _safe_float(kappa)}
        
    except Exception as e:
        logger.error(f"Error calculating Fleiss' Kappa: {e}")
        return {'fleiss': None, 'error': str(e)}


def calculate_krippendorff_alpha(data: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate Krippendorff's Alpha.
    
    Args:
        data: Long-format DataFrame with columns: 'target', 'reader', 'value'
        
    Returns:
        Dictionary with Alpha value
    """
    try:
        # Pivot to wide format (raters x targets)
        pivot_data = data.pivot(index='reader', columns='target', values='value')
        reliability_data = pivot_data.values
        
        # Try to use krippendorff package
        try:
            import krippendorff
            alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='interval')
            return {'krippendorff': _safe_float(alpha)}
        except ImportError:
            # Fallback to simplified calculation
            alpha = _calculate_alpha_simplified(reliability_data)
            return {'krippendorff': _safe_float(alpha)}
            
    except Exception as e:
        logger.error(f"Error calculating Krippendorff's Alpha: {e}")
        return {'krippendorff': None, 'error': str(e)}


def _calculate_alpha_simplified(data: np.ndarray) -> float:
    """
    Simplified Alpha calculation for interval data.
    
    Args:
        data: Matrix of shape (n_raters, n_units)
        
    Returns:
        Alpha value
    """
    valid = ~np.isnan(data)
    n_units = data.shape[1]
    
    # Calculate observed disagreement
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


def _safe_float(value) -> Optional[float]:
    """
    Safely convert value to float, handling NaN and numpy types.
    
    Args:
        value: Value to convert
        
    Returns:
        Native Python float or None
    """
    if pd.isna(value) or np.isnan(value):
        return None
    return float(value)