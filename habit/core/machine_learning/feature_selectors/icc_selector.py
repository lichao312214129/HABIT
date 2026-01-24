"""
ICC-based Feature Selector

This module provides a feature selector that filters features based on their
Intraclass Correlation Coefficient (ICC) values.
"""
import json
from typing import Dict, List, Optional
from .selector_registry import register_selector

@register_selector('icc')
def icc_selector(
    icc_results: Optional[str] = None,
    icc_results_path: Optional[str] = None,
    keys: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    threshold: float = 0.75,
    metric: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    Filters features by selecting those that meet a minimum ICC threshold
    across a specified set of groups.

    Args:
        icc_results (str): Path to the JSON file containing ICC results (alias for icc_results_path).
        icc_results_path (str): Path to the JSON file containing ICC results.
        keys (List[str]): A list of group names from the ICC results (alias for groups).
        groups (List[str]): A list of group names from the ICC results.
        threshold (float): The minimum ICC value for a feature to be considered stable.
        metric (str): The specific ICC metric to use for thresholding (e.g., 'ICC3', 'ICC2').
        **kwargs: Additional arguments.

    Returns:
        List[str]: A list of feature names that meet the ICC threshold in all specified groups.
    """
    # Handle aliases
    results_path = icc_results or icc_results_path
    target_groups = keys or groups

    if not results_path:
        raise ValueError("ICC results path must be specified (use 'icc_results' or 'icc_results_path')")
    
    if not target_groups:
        raise ValueError("Groups/keys for ICC selection must be specified")

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            icc_results = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"ICC results file not found at: {results_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Could not decode JSON from file: {results_path}")

    selected_features_per_group = []

    for group_name in target_groups:
        if group_name not in icc_results:
            # Try to find a partial match if exact match fails
            matched_group = None
            for existing_group in icc_results.keys():
                if group_name in existing_group:
                    matched_group = existing_group
                    break
            
            if matched_group:
                print(f"Using matched group '{matched_group}' for requested key '{group_name}'")
                group_name = matched_group
            else:
                raise KeyError(f"Group/Key '{group_name}' not found in the ICC results file.")
        
        features_in_group = icc_results[group_name]
        stable_features = set()
        
        for feature, metrics in features_in_group.items():
            # Support both simple {feature: value} and complex {feature: {metric: {value: X}}} formats
            if not isinstance(metrics, dict):
                # Simple format: {feature: icc_value}
                if metrics >= threshold:
                    stable_features.add(feature)
                continue

            # Complex format: {feature: {metric_name: {value: X}}} or {feature: {metric_name: X}}
            icc_value = None
            
            # 1. If metric is specified, try that first
            if metric:
                metric_data = metrics.get(metric)
                if isinstance(metric_data, dict):
                    icc_value = metric_data.get('value')
                elif isinstance(metric_data, (int, float)):
                    icc_value = metric_data
            
            # 2. If no metric specified or not found, try common defaults
            if icc_value is None:
                for m_key in ['ICC3', 'ICC2', 'icc3', 'icc2']:
                    m_data = metrics.get(m_key)
                    if isinstance(m_data, dict):
                        icc_value = m_data.get('value')
                        break
                    elif isinstance(m_data, (int, float)):
                        icc_value = m_data
                        break
            
            # 3. Last resort: try any key containing 'icc'
            if icc_value is None:
                for m_key, m_data in metrics.items():
                    if 'icc' in m_key.lower():
                        if isinstance(m_data, dict):
                            icc_value = m_data.get('value')
                            break
                        elif isinstance(m_data, (int, float)):
                            icc_value = m_data
                            break

            if icc_value is not None and icc_value >= threshold:
                stable_features.add(feature)
        
        selected_features_per_group.append(stable_features)
        print(f"Group '{group_name}': {len(stable_features)} features above threshold {threshold}")

    if not selected_features_per_group:
        return []
        
    common_stable_features = set.intersection(*selected_features_per_group)
    print(f"ICC Selection: {len(common_stable_features)} common features found across {len(target_groups)} groups")
    
    return sorted(list(common_stable_features))
