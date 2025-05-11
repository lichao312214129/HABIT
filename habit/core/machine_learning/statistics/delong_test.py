#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy.stats
from scipy import stats
import argparse
import sys
import os
import json
# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
   """Computes midranks.
   Args:
      x - a 1D numpy array
   Returns:
      array of midranks
   """
   J = np.argsort(x)
   Z = x[J]
   N = len(x)
   T = np.zeros(N, dtype=np.float32)
   i = 0
   while i < N:
       j = i
       while j < N and Z[j] == Z[i]:
           j += 1
       T[i:j] = 0.5*(i + j - 1)
       i = j
   T2 = np.empty(N, dtype=np.float32)
   # Note(kazeevn) +1 is due to Python using 0-based indexing
   # instead of 1-based in the AUC formula in the paper
   T2[J] = T + 1
   return T2

def compute_midrank_weight(x, sample_weight):
   """Computes midranks.
   Args:
      x - a 1D numpy array
   Returns:
      array of midranks
   """
   J = np.argsort(x)
   Z = x[J]
   cumulative_weight = np.cumsum(sample_weight[J])
   N = len(x)
   T = np.zeros(N, dtype=np.float32)
   i = 0
   while i < N:
       j = i
       while j < N and Z[j] == Z[i]:
           j += 1
       T[i:j] = cumulative_weight[i:j].mean()
       i = j
   T2 = np.empty(N, dtype=np.float32)
   T2[J] = T
   return T2

def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight=None):

   if sample_weight is None:

       return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)

   else:

       return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)

def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):

   """

   The fast version of DeLong's method for computing the covariance of

   unadjusted AUC.

   Args:

      predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]

         sorted such as the examples with label "1" are first

   Returns:

      (AUC value, DeLong covariance)

   Reference:

    @article{sun2014fast,

      title={Fast Implementation of DeLong's Algorithm for

             Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},

      author={Xu Sun and Weichao Xu},

      journal={IEEE Signal Processing Letters},

      volume={21},

      number={11},

      pages={1389--1393},

      year={2014},

      publisher={IEEE}

    }

   """

   # Short variables are named as they are in the paper

   m = label_1_count

   n = predictions_sorted_transposed.shape[1] - m

   positive_examples = predictions_sorted_transposed[:, :m]

   negative_examples = predictions_sorted_transposed[:, m:]

   k = predictions_sorted_transposed.shape[0]



   tx = np.empty([k, m], dtype=np.float32)

   ty = np.empty([k, n], dtype=np.float32)

   tz = np.empty([k, m + n], dtype=np.float32)

   for r in range(k):

       tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])

       ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])

       tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)

   total_positive_weights = sample_weight[:m].sum()

   total_negative_weights = sample_weight[m:].sum()

   pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])

   total_pair_weights = pair_weights.sum()

   aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights

   v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights

   v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights

   sx = np.cov(v01)

   sy = np.cov(v10)

   delongcov = sx / m + sy / n

   return aucs, delongcov

def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):

   """

   The fast version of DeLong's method for computing the covariance of

   unadjusted AUC.

   Args:

      predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]

         sorted such as the examples with label "1" are first

   Returns:

      (AUC value, DeLong covariance)

   Reference:

    @article{sun2014fast,

      title={Fast Implementation of DeLong's Algorithm for

             Comparing the Areas Under Correlated Receiver Oerating

             Characteristic Curves},

      author={Xu Sun and Weichao Xu},

      journal={IEEE Signal Processing Letters},

      volume={21},

      number={11},

      pages={1389--1393},

      year={2014},

      publisher={IEEE}

    }

   """

   # Short variables are named as they are in the paper

   m = label_1_count
   n = predictions_sorted_transposed.shape[1] - m
   positive_examples = predictions_sorted_transposed[:, :m]
   negative_examples = predictions_sorted_transposed[:, m:]
   k = predictions_sorted_transposed.shape[0]



   tx = np.empty([k, m], dtype=np.float32)
   ty = np.empty([k, n], dtype=np.float32)
   tz = np.empty([k, m + n], dtype=np.float32)

   for r in range(k):

       tx[r, :] = compute_midrank(positive_examples[r, :])
       ty[r, :] = compute_midrank(negative_examples[r, :])
       tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

   aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
   v01 = (tz[:, :m] - tx[:, :]) / n
   v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

   sx = np.cov(v01)
   sy = np.cov(v10)
   delongcov = sx / m + sy / n

   return aucs, delongcov

def calc_pvalue(aucs, sigma):
   """Computes log(10) of p-values.
   Args:
      aucs: 1D array of AUCs
      sigma: AUC DeLong covariances
   Returns:
      log10(pvalue)

   """

   l = np.array([[1, -1]])

   z = np.abs(np.diff(aucs)) / (np.sqrt(np.dot(np.dot(l, sigma), l.T)) + 1e-8)
   pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
   #  print(10**(np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)))
   return pvalue

def compute_ground_truth_statistics(ground_truth, sample_weight=None):
   assert np.array_equal(np.unique(ground_truth), [0, 1])
   order = (-ground_truth).argsort()
   label_1_count = int(ground_truth.sum())
   if sample_weight is None:
       ordered_sample_weight = None
   else:
       ordered_sample_weight = sample_weight[order]

   return order, label_1_count, ordered_sample_weight

def delong_roc_variance(ground_truth, predictions):
   """
   Computes ROC AUC variance for a single set of predictions
   Args:
      ground_truth: np.array of 0 and 1
      predictions: np.array of floats of the probability of being class 1
   """
   sample_weight = None
   order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
       ground_truth, sample_weight)
   predictions_sorted_transposed = predictions[np.newaxis, order]
   aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)

   assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
   return aucs[0], delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two):
   """
   Computes log(p-value) for hypothesis that two ROC AUCs are different
   Args:
      ground_truth: np.array of 0 and 1
      predictions_one: predictions of the first model,
         np.array of floats of the probability of being class 1
      predictions_two: predictions of the second model,
         np.array of floats of the probability of being class 1
   """
   sample_weight = None
   order, label_1_count,ordered_sample_weight = compute_ground_truth_statistics(ground_truth)
   predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
   aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count,sample_weight)

   return calc_pvalue(aucs, delongcov)

def delong_roc_ci(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.95) -> tuple:
    """
    Computes ROC AUC and its confidence interval using DeLong's method.
    Args:
        y_true: true labels
        y_pred: predicted probabilities
        alpha: confidence level
    Returns:
        (AUC, (lower_ci, upper_ci))
    """
    aucs, auc_cov = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(
        lower_upper_q,
        loc=aucs,
        scale=auc_std)
    ci[ci > 1] = 1
    return aucs, ci

def perform_delong_test(input_file: str, true_label_col: str, model_cols: list) -> list:
    """
    Performs DeLong test on multiple models' predictions from a CSV file.
    Args:
        input_file: path to input CSV file
        true_label_col: column name for true labels
        model_cols: list of column names for model predictions
    Returns:
        list of comparison results
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get true labels
    y_true = df[true_label_col].values
    
    # Perform pairwise comparisons
    results = []
    for i in range(len(model_cols)):
        for j in range(i + 1, len(model_cols)):
            model1 = model_cols[i]
            model2 = model_cols[j]
            y_pred1 = df[model1].values
            y_pred2 = df[model2].values
            
            # Calculate AUCs and CIs
            auc1, ci1 = delong_roc_ci(y_true, y_pred1)
            auc2, ci2 = delong_roc_ci(y_true, y_pred2)
            
            # Calculate p-value
            p_value = delong_roc_test(y_true, y_pred1, y_pred2)
            
            # Create comparison result
            comparison_result = {
                'comparison': f"{model1} vs {model2}",
                f'{model1}_auc': float(auc1),
                f'{model1}_ci_lower': float(ci1[0]),
                f'{model1}_ci_upper': float(ci1[1]),
                f'{model2}_auc': float(auc2),
                f'{model2}_ci_lower': float(ci2[0]),
                f'{model2}_ci_upper': float(ci2[1]),
                'p_value': float(p_value),
                'significant_difference': bool(p_value < 0.05),
                'conclusion': f"{model1} and {model2} have significantly different AUCs (p<0.05)" if p_value < 0.05 else f"{model1} and {model2} do not have significantly different AUCs (p≥0.05)"
            }
            results.append(comparison_result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Perform DeLong test on multiple models')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--true_label', type=str, required=True, help='Column name for true labels')
    parser.add_argument('--model_cols', type=str, required=True, help='Comma-separated list of column names for model predictions')
    parser.add_argument('--output', type=str, help='Output JSON file path (optional)')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} does not exist")
        sys.exit(1)
    
    # Parse model columns
    model_cols = [col.strip() for col in args.model_cols.split(',')]
    
    # Perform DeLong test
    results = perform_delong_test(args.input, args.true_label, model_cols)
    
    # Print results
    print("\nDeLong Test Results:")
    print("=" * 50)
    for result in results:
        print(f"\n{result['comparison']}")
        print(f"P-value: {result['p_value']:.4f}")
        print(f"Conclusion: {result['conclusion']}")
        print(f"AUCs with 95% CI:")
        model1, model2 = result['comparison'].split(" vs ")
        print(f"{model1}: {result[f'{model1}_auc']:.3f} ({result[f'{model1}_ci_lower']:.3f}-{result[f'{model1}_ci_upper']:.3f})")
        print(f"{model2}: {result[f'{model2}_auc']:.3f} ({result[f'{model2}_ci_lower']:.3f}-{result[f'{model2}_ci_upper']:.3f})")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    # For debugging
    if len(sys.argv) == 1:
        # Default test case
        sys.argv.extend([
            '--input', '../demo_data/results/all_prediction_results.csv',
            '--true_label', 'true_label',
            '--model_cols', 'LogisticRegression_prob,SVC_prob,XGBoost_prob',
            '--output', '../demo_data/results/delong_test_results.json'
        ])
    
    main() 