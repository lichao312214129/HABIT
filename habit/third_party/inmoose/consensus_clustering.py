# -----------------------------------------------------------------------------
# Copyright (C) 2019-2021 Ž. Sajovic
# Copyright (C) 2023 S. Weill and G. Appé
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
#
# Vendored from InMoose (GPL-3.0-or-later):
#   https://github.com/epigenelabs/inmoose/blob/v0.9.1/inmoose/consensus_clustering/consensus_clustering.py
# which states that it is based on:
#   https://github.com/ZigaSajovic/Consensus_Clustering
#   file consensusClustering.py as of 30 July 2021.
#
# HABIT modifications, still under GPL-3.0-or-later:
#   * ``logging.getLogger`` replaces ``from ..utils import LOGGER``.
#   * Methods that import seaborn or pandas and write image files
#     (``plot_clustermap``, ``plot_clusters_consensus``,
#     ``line_plots_cluster_consensus``, ``plot_deltak``,
#     ``build_clusters_consensus_df``) are omitted. HABIT draws figures
#     from the numeric attributes in ``habit.viz``.
#   * The numerical methods below are unchanged from that InMoose file.
#
"""Monti consensus clustering as implemented by Sajovic and InMoose.

The class builds one ``n_items x n_items`` consensus matrix per candidate
cluster count. Each matrix entry is the fraction of resamples in which two
items were both drawn and placed in the same cluster. ``bestK`` is the
argmax of the relative change in area under the consensus-index CDF, using
the Sajovic / InMoose indexing (the largest candidate ``k`` has no delta
entry and is never selected by that rule).

This is item resampling only. Feature resampling from Monti et al. (2003)
and the PAC / ``pFeature`` options of ConsensusClusterPlus are not part of
this implementation.
"""

from __future__ import annotations

import bisect
import logging
from itertools import combinations

import numpy as np

LOGGER = logging.getLogger(__name__)


class consensusClustering:
    """
    Implementation of Consensus clustering, following the paper https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf

    Arguments
    ---------
    cluster: sklearn clustering class
        Clustering algorithm to use for consensus clustering
        NOTE: the class is to be instantiated with parameter `n_clusters`, and possess a `fit_predict` method, which is invoked on data.
    mink: int
        smallest number of clusters to try, default = 2
    maxk: int
        biggest number of clusters to try, default = 10
    nb_resampling_iteration: int
        number of resamplings for each cluster number, default = 50
    resample_proportion: float
        percentage to sample. Number between 0 and 1, default = 0.5
    n_bins: int
        Number of bins used to compute histogram in compute_area_under_curve, default = 10
    consensus_matrices: ndarray[float]
        consensus matrices for each k
        NOTE: every consensus matrix is retained, like specified in the paper
    Ak: array[float]
        area under CDF for each number of clusters (see paper: section 3.3.1. Consensus distribution.)
    deltaK: array[float]
        changes in areas under CDF (see paper: section 3.3.1. Consensus distribution.)
    bestK: int
        number of clusters that was found to be best
    """

    def __init__(
        self,
        cluster,
        mink=2,
        maxk=10,
        nb_resampling_iteration=50,
        resample_proportion=0.5,
        n_bins=10,
    ):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.cluster_ = cluster
        self.resample_proportion = resample_proportion
        self.min_k = mink
        self.max_k = maxk
        self.nb_iteration = nb_resampling_iteration
        self.nbins = n_bins
        self.consensus_matrices = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    def _internal_resample(self, data, proportion, rand_state):
        """
        Sampling data based on the proportion of samples

        Arguments
        ---------
        data: ndarray
            numpy matrix to run the cluster on samples*features
        proportion: float
            percentage to sample
        rand_state: numpy.random.Generator
            Numpy random generator instance

        Returns
        -------
        resampled_indices
            sample's indices to use for the iteration
        """

        resampled_indices = rand_state.choice(
            range(data.shape[0]), size=int(data.shape[0] * proportion), replace=False
        )
        return resampled_indices

    def compute_consensus_clustering(self, data, random_state, verbose=False):
        """
        Fits a consensus matrix for each number of clusters

        Arguments
        ---------
        data: ndarray
            numpy matrix to run the cluster on samples*features
        random_state: int
            seed to use to generate a numpy generator instance. Default is None
        verbose: bool (default=False)
            If True, print the number of clusters for which the consensus matrix is computed
        """
        rand_state = np.random.default_rng(random_state)
        consensus_mats = np.zeros(
            (self.max_k - self.min_k + 1, data.shape[0], data.shape[0])
        )

        for k in range(self.min_k, self.max_k + 1):  # for each number of clusters
            if verbose:
                LOGGER.info(f"Computing consensus matrix for {k} clusters")
            indicator_matrix = np.zeros((data.shape[0],) * 2)
            connectivity_matrix = np.zeros((data.shape[0],) * 2)

            for _ in range(self.nb_iteration):  # resample H times
                resampled_indices = self._internal_resample(
                    data, self.resample_proportion, rand_state
                )

                Mh = self.cluster_(n_clusters=k).fit_predict(data[resampled_indices, :])

                connectivity_matrix += self.compute_iteration_connectivity_matrix(
                    resampled_indices, Mh, k, data.shape[0]
                )

                indicator_matrix += self.compute_iteration_indicator_mat(
                    resampled_indices, data.shape[0]
                )

            consensus_mats[k - self.min_k] = self.compute_consensus_mat(
                connectivity_matrix, indicator_matrix
            )

        self.consensus_matrices = consensus_mats

        self.bestK = self.compute_bestk()

    def compute_iteration_connectivity_matrix(
        self, resampled_indices, clust_res, k, nb_samples
    ):
        """
        Compute connectivity matrix

        The connectivity matrix allows to track how many times 2 elements of the matrix where sampled and clustered together

        Arguments
        ---------
        resampled_indices: array
            indices of the elements selected for the current iteration
        clust_res: array
            Clustering results
        k: int
            number of clusters
        nb_samples: int
            total number of elements of the input matrix

        Returns
        -------
        conn_mat
            Connectivity matrix for the iteration
        """
        # find indices of elements from same clusters with bisection
        # on sorted array => this is more efficient than brute force search
        conn_mat = np.zeros((nb_samples,) * 2)
        index_mapping = np.array((clust_res, resampled_indices)).T
        index_mapping = index_mapping[index_mapping[:, 0].argsort()]
        sorted_ = index_mapping[:, 0]
        id_clusts = index_mapping[:, 1]
        for i in range(k):  # for each cluster
            ia = bisect.bisect_left(sorted_, i)
            ib = bisect.bisect_right(sorted_, i)
            is_ = np.sort(id_clusts[ia:ib])
            ids_ = np.array(list(combinations(is_, 2))).T
            # sometimes only one element is in a cluster (no combinations)
            if ids_.size != 0:
                conn_mat[ids_[0], ids_[1]] += 1
        return conn_mat

    def compute_iteration_indicator_mat(self, resampled_indices, nb_samples):
        """
        Compute indicator matrix for one iteration

        The indicator matrix allows to track how many time 2 elements of the matrix where sampled together

        Arguments
        ---------
        resampled_indices: array
            sample's indices to use for the iteration
        nb_samples: int
            Number of samples in the initial matrix

        Returns
        -------
        indic_mat
            indicator matrix for the current iteration
        """
        indic_mat = np.zeros((nb_samples,) * 2)
        ids_2 = np.array(list(combinations(np.sort(resampled_indices), 2))).T
        indic_mat[ids_2[0], ids_2[1]] += 1
        return indic_mat

    def compute_consensus_mat(self, connectivity_mat, indicator_mat):
        """
        Compute consensus matrix defined as the normalized sum of the connectivity matrices of all the resampled datasets

        Arguments
        ---------
        connectivity_mat: ndarray
            connectivity matrix for k clusters. Sum of the iteration connectivity matrix
        indicator_mat: ndarray
            indicator matrix for k clusters. Sum of the iteration indicator matrix

        Returns
        -------
        consensus_mat
            Consensus matrix for k clusters
        """
        consensus_mat = connectivity_mat / (indicator_mat + 1e-8)
        # consensus_mat is upper triangular (with zeros on diagonal), we now make it symmetric
        consensus_mat += consensus_mat.T
        consensus_mat[range(consensus_mat.shape[0]), range(consensus_mat.shape[0])] = (
            1  # always with self
        )
        return consensus_mat

    def compute_bestk(self):
        """
        Get best number of clusters

        Returns
        -------
        best number of clusters
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        self.Ak = self.compute_area_under_curve()
        self.deltaK = self.compute_area_delta()
        return (
            np.argmax(self.deltaK) + self.min_k if self.deltaK.size > 0 else self.min_k
        )

    def compute_area_delta(self):
        """
        Compute the differences between areas under CDFs

        Returns
        -------
        Array containing the difference between the area under the CDFs for each number of clusters
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        return np.array(
            [
                (Ab - Aa) / Aa if i > 2 else Aa / self.nbins
                for Ab, Aa, i in zip(
                    self.Ak[1:], self.Ak[:-1], range(self.min_k, self.max_k + 1)
                )
            ]
        )

    def compute_area_under_curve(self):
        """
        Compute area under the CDFs curve

        Returns
        -------
        area_under_curve
            array of the area under the CDF for each cluster number
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        area_under_curve = np.zeros(self.max_k - self.min_k + 1)
        for i, m in enumerate(self.consensus_matrices):
            hist, bins = np.histogram(m.ravel(), density=True, bins=self.nbins)
            area_under_curve[i] = sum(
                h * (b - a) for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist))
            )
        return area_under_curve

    def compute_summary_statistics(self, k):
        """
        For one prediction, compute a summary statistics, cluster consensus and item consensus, showing cluster stability and most representative cluster items.

        Cluster consensus is defined as the average consensus index between all pairs of items belonging to the cluster.
        Item consensus is defined as the average consensus index between item ei and all the (other) items in a cluster.

        Arguments
        ---------
        k: int
            Number of clusters

        Returns
        -------
        predictions
            Array of the predicted cluster
        clusters_consensus
            Array of cluster consensus
        items_consensus
            Array of the item consensus (nb_items * k)
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        predictions = self.predict(k)
        clusters_consensus = self.compute_clusters_consensus(predictions, k)
        items_consensus = self.compute_items_consensus(predictions, k)
        return predictions, clusters_consensus, items_consensus

    def compute_clusters_consensus(self, prediction, k):
        """
        For one prediction, compute clusters consensus, showing cluster stability.

        Cluster consensus is defined as the average consensus index between all pairs of items belonging to the cluster.

        Arguments
        ---------
        prediction: ndarray
            Array of the predicted cluster
        k: int
            Number of clusters

        Returns
        -------
        clusters_consensus
            Array of cluster consensus
        """
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        clusters_consensus = np.zeros(k)
        for clust in range(k):
            ids = np.where(prediction == clust)[0]
            clust_size = len(ids)
            ids_ = np.array(list(combinations(np.sort(ids), 2))).T
            if ids_.size == 0:
                clusters_consensus[clust] = np.nan
                LOGGER.warning(
                    f"Single sample cluster for cluster {str(clust)} of k={k}. Setting cluster consensus to NaN."
                )
                continue
            clusters_consensus[clust] = self.consensus_matrices[
                k - self.min_k, ids_[0], ids_[1]
            ].sum() / (clust_size * (clust_size - 1) / 2)
        return clusters_consensus

    def compute_items_consensus(self, prediction, k):
        """
        For one prediction, compute item consensus, showing most representative cluster items.

        Item consensus is defined as the average consensus index between item ei and all the (other) items in a cluster.

        Arguments
        ---------
        prediction: ndarray
            Array of the predicted cluster
        k: int
            Number of clusters

        Returns
        -------
        items_consensus
            Array of the item consensus (nb_items * k)
        """
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        items_consensus = np.zeros(
            (self.consensus_matrices[k - self.min_k].shape[0], k)
        )

        clusters, sizes = np.unique(prediction, return_counts=True)

        for id in range(items_consensus.shape[0]):
            for clust, size in zip(clusters, sizes):
                clust_elem = np.where(prediction == clust)[0]
                if id in clust_elem:
                    cols = clust_elem[clust_elem != id]
                    items_consensus[id, clust] = np.sum(
                        self.consensus_matrices[k - self.min_k, id, cols]
                    ) / (size - 1)
                else:
                    items_consensus[id, clust] = (
                        np.sum(self.consensus_matrices[k - self.min_k, id, clust_elem])
                        / size
                    )
        return items_consensus

    def predict(self, k):
        """
        Predicts clusters on the consensus matrix, for k clusters using the consensus matrix

        Arguments
        ---------
        k: int
            Number of clusters

        Returns
        -------
        predicted cluster for k clusters
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        assert self.min_k <= k <= self.max_k, (
            "Number of clusters must be between min_k and max_k"
        )
        return self.cluster_(n_clusters=k).fit_predict(
            1 - self.consensus_matrices[k - self.min_k]
        )

    def predict_data(self, data):
        """
        Predicts clusters on the data, for best found cluster number

        Arguments
        ---------
        data: ndarray
            input matrix (samples * attributes)

        Returns
        -------
        predicted cluster for best number of clusters
        """
        assert self.consensus_matrices is not None, "First compute consensus clustering"
        return self.cluster_(n_clusters=self.bestK).fit_predict(data)
