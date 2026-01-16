#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to fix two_step_strategy.py
"""

file_path = 'habit/core/habitat_analysis/strategies/two_step_strategy.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the two calls
old1 = 'self.analysis.results_df = self.analysis.clustering_manager.perform_population_clustering('
new1 = 'self.analysis.results_df = self._perform_population_clustering('
content = content.replace(old1, new1)

old2 = '''            # Ensure ResultManager has the latest results_df
            self.analysis.result_manager.results_df = self.analysis.results_df
            self.analysis.result_manager.save_results(
                subjects, failed_subjects, self.analysis.pipeline, optimal_n_clusters
            )'''
new2 = '''            self._save_results(
                subjects, failed_subjects, self.analysis.pipeline, optimal_n_clusters
            )'''
content = content.replace(old2, new2)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('File updated successfully')
print('- Updated perform_population_clustering call')
print('- Updated save_results call')
