#!/usr/bin/env python
"""
Utility functions for habitat feature extraction
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

class FeatureUtils:
    """Utility class for feature extraction"""
    
    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary
        
        Args:
            d: Dictionary to flatten
            parent_key: Key of parent dictionary (used in recursion)
            sep: Separator between keys in flattened dictionary
            
        Returns:
            Dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(FeatureUtils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def get_n_habitats_from_csv(habitat_folder: str) -> Optional[int]:
        """
        Read the number of habitats from habitats.csv file
        
        Args:
            habitat_folder: Path to the folder containing habitats.csv
            
        Returns:
            int: Number of habitats if found, None otherwise
        """
        import os
        
        try:
            csv_path = os.path.join(habitat_folder, 'habitats.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'Habitats' in df.columns:
                    # Assume the Habitats column contains habitat labels, count unique values
                    unique_habitats = df['Habitats'].nunique()
                    logging.info(f"Read {unique_habitats} habitats from habitats.csv")
                    return unique_habitats
                else:
                    logging.error("Habitats column not found in habitats.csv")
            else:
                logging.error(f"habitats.csv file not found: {csv_path}")
        except Exception as e:
            logging.error(f"Error reading habitats.csv: {str(e)}")
        
        return None
    
    @staticmethod
    def create_empty_dataframe_like(reference_df: pd.DataFrame, index: List[str] = None) -> pd.DataFrame:
        """
        Create an empty DataFrame with the same structure as a reference DataFrame
        
        Args:
            reference_df: Reference DataFrame to copy structure from
            index: List of index values for the new DataFrame
            
        Returns:
            pd.DataFrame: Empty DataFrame with same structure as reference_df
        """
        if index is None:
            index = [0]
            
        return pd.DataFrame(
            data=np.nan, 
            index=index, 
            columns=reference_df.columns
        ) 