"""
Base strategy interface for habitat analysis.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis


class BaseHabitatStrategy(ABC):
    """
    Abstract base class for habitat analysis strategies.

    Each strategy should implement run() and return a results DataFrame.
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize the strategy with a HabitatAnalysis instance.

        Args:
            analysis: HabitatAnalysis instance with shared utilities and configuration
        """
        self.analysis = analysis
        self.config = analysis.config
        self.logger = analysis.logger

    @abstractmethod
    def run(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: bool = True
    ) -> pd.DataFrame:
        """
        Execute the strategy and return results.

        Args:
            subjects: List of subjects to process (None means all subjects)
            save_results_csv: Whether to save results to CSV

        Returns:
            Results DataFrame
        """
        raise NotImplementedError
