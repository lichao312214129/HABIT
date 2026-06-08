# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
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
        Read the number of habitats from the habitats results table.

        Supports ``habitats.parquet`` (default) and legacy ``habitats.csv``.

        Args:
            habitat_folder: Path to the folder containing the habitats table.

        Returns:
            int: Number of habitats if found, None otherwise.
        """
        from habit.utils.habitats_results_io import (
            find_habitats_results_file,
            load_habitats_results,
        )

        try:
            results_path = find_habitats_results_file(habitat_folder)
            if results_path is None:
                logging.error(
                    "Habitats results file not found in folder: %s",
                    habitat_folder,
                )
                return None

            df = load_habitats_results(results_path)
            if "Habitats" in df.columns:
                unique_habitats = int(df["Habitats"].nunique())
                logging.info(
                    "Read %s habitats from %s",
                    unique_habitats,
                    results_path.name,
                )
                return unique_habitats

            logging.error(
                "Habitats column not found in habitats results file: %s",
                results_path,
            )
        except Exception as exc:
            logging.error("Error reading habitats results table: %s", exc)

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