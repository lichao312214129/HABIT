"""
Feature extraction module for habitat analysis.

This module provides functionality for extracting various features from habitat maps,
including radiomic features, non-radiomic features, and MSI features.
"""

from .habitat_analyzer import HabitatMapAnalyzer

# Backward compatibility alias
HabitatFeatureExtractor = HabitatMapAnalyzer

__all__ = ['HabitatMapAnalyzer', 'HabitatFeatureExtractor']
