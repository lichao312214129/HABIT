"""
SLIC clustering implementation for habitat analysis.

This module provides a SLIC-style clustering algorithm for voxel-level habitat
segmentation. It uses ``skimage.segmentation.slic`` for final clustering and
supports one-step cluster-number selection with spatially regularized features.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .base_clustering import BaseClustering, register_clustering


@register_clustering("slic")
class SLICClustering(BaseClustering):
    """
    SLIC-based clustering for voxel-level habitat analysis.

    Notes:
        - ``predict`` is only valid for the same subject after ``fit`` because
          SLIC is not a parametric model for unseen samples.
        - Output labels are returned as 0-indexed values to match the current
          HABIT convention (manager adds +1 later).
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        random_state: int = 0,
        compactness: float = 0.1,
        max_iter: int = 10,
        sigma: float = 0.0,
        enforce_connectivity: bool = True,
        n_init: int = 10,
        spacing: Optional[Tuple[float, float, float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize SLIC clustering.

        Args:
            n_clusters: Target number of supervoxels/habitats.
            random_state: Random seed for reproducibility.
            compactness: Trade-off between appearance and spatial proximity.
            max_iter: Maximum number of iterations for SLIC.
            sigma: Gaussian smoothing width before segmentation.
            enforce_connectivity: Whether to enforce connected supervoxels.
            n_init: Number of initializations for KMeans in optimal-k search.
            spacing: Optional voxel spacing (z, y, x) for anisotropic data.
            **kwargs: Reserved for compatibility.
        """
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.compactness: float = compactness
        self.max_iter: int = max_iter
        self.sigma: float = sigma
        self.enforce_connectivity: bool = enforce_connectivity
        self.n_init: int = n_init
        self.spacing: Optional[Tuple[float, float, float]] = spacing
        self.kwargs: Dict[str, Any] = kwargs
        self._fitted_sample_count: Optional[int] = None

    @staticmethod
    def _coords_from_mask(mask_array: np.ndarray) -> np.ndarray:
        """
        Extract foreground voxel coordinates from a binary/non-zero mask.

        Args:
            mask_array: 3D mask array where non-zero values indicate ROI voxels.

        Returns:
            np.ndarray: Coordinates with shape (n_voxels, 3) in (z, y, x) order.
        """
        return np.column_stack(np.where(mask_array > 0))

    def _build_regularized_features(
        self,
        X: np.ndarray,
        spatial_coords: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build SLIC-style regularized features for cluster-number selection.

        The feature embedding appends scaled spatial coordinates so that cluster
        selection can incorporate spatial compactness similarly to SLIC.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            spatial_coords: Spatial coordinates with shape (n_samples, 3), optional.

        Returns:
            np.ndarray: Regularized feature matrix.
        """
        if spatial_coords is None or spatial_coords.shape[0] != X.shape[0]:
            return X

        coords = spatial_coords.astype(np.float32)
        coords -= coords.mean(axis=0, keepdims=True)
        coords_std = coords.std(axis=0, keepdims=True)
        coords_std[coords_std == 0.0] = 1.0
        coords /= coords_std

        X_float = X.astype(np.float32, copy=False)
        feature_std = float(np.mean(np.std(X_float, axis=0)))
        if feature_std <= 0.0:
            feature_std = 1.0

        spatial_scale = self.compactness * feature_std
        return np.concatenate([X_float, coords * spatial_scale], axis=1)

    @staticmethod
    def _sanitize_feature_matrix(X: np.ndarray) -> np.ndarray:
        """
        Replace non-finite values (NaN/Inf) with finite values per feature column.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            np.ndarray: Sanitized feature matrix as float32.
        """
        X_clean: np.ndarray = X.astype(np.float32, copy=True)
        for col_idx in range(X_clean.shape[1]):
            col = X_clean[:, col_idx]
            finite_mask = np.isfinite(col)
            if finite_mask.all():
                continue
            if np.any(finite_mask):
                fill_value = float(np.median(col[finite_mask]))
            else:
                fill_value = 0.0
            col[~finite_mask] = fill_value
            X_clean[:, col_idx] = col
        return X_clean

    def fit(
        self,
        X: np.ndarray,
        spatial_coords: Optional[np.ndarray] = None,
        mask_array: Optional[np.ndarray] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> "SLICClustering":
        """
        Fit SLIC on one subject.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            spatial_coords: Optional voxel coordinates (z, y, x), shape (n_samples, 3).
            mask_array: Optional full-size ROI mask array for coordinate recovery.
            spacing: Optional voxel spacing (z, y, x). Overrides constructor value.

        Returns:
            SLICClustering: Fitted clustering instance.
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters must be specified for SLIC clustering.")
        if spacing is not None:
            self.spacing = spacing

        if spatial_coords is None:
            if mask_array is None:
                raise ValueError(
                    "SLIC requires spatial information. "
                    "Provide spatial_coords or mask_array."
                )
            spatial_coords = self._coords_from_mask(mask_array)

        if spatial_coords.shape[0] != X.shape[0]:
            min_len = min(spatial_coords.shape[0], X.shape[0])
            warnings.warn(
                "Voxel-feature length does not match ROI voxel count. "
                "Truncating to the minimum length for SLIC.",
                RuntimeWarning,
            )
            spatial_coords = spatial_coords[:min_len]
            X = X[:min_len]

        # skimage.slic does not allow unmasked non-finite values in image tensor.
        X = self._sanitize_feature_matrix(X)

        # Build a tight ROI bounding box to avoid allocating full-volume tensors.
        # This is critical for memory usage when whole-image size is much larger
        # than the tumor ROI.
        if mask_array is not None:
            z_idx, y_idx, x_idx = np.where(mask_array > 0)
            if z_idx.size > 0:
                z_min, z_max = int(z_idx.min()), int(z_idx.max())
                y_min, y_max = int(y_idx.min()), int(y_idx.max())
                x_min, x_max = int(x_idx.min()), int(x_idx.max())
            else:
                # Fallback to coordinate-derived bounds if mask is unexpectedly empty.
                z_min, y_min, x_min = np.min(spatial_coords, axis=0).astype(int).tolist()
                z_max, y_max, x_max = np.max(spatial_coords, axis=0).astype(int).tolist()
        else:
            z_min, y_min, x_min = np.min(spatial_coords, axis=0).astype(int).tolist()
            z_max, y_max, x_max = np.max(spatial_coords, axis=0).astype(int).tolist()

        local_shape = (
            int(z_max - z_min + 1),
            int(y_max - y_min + 1),
            int(x_max - x_min + 1),
        )
        local_coords = spatial_coords.copy()
        local_coords[:, 0] -= z_min
        local_coords[:, 1] -= y_min
        local_coords[:, 2] -= x_min

        # Allocate only ROI-bbox-sized dense arrays.
        feature_volume = np.zeros(local_shape + (X.shape[1],), dtype=np.float32)
        valid_mask = np.zeros(local_shape, dtype=bool)
        valid_mask[
            local_coords[:, 0],
            local_coords[:, 1],
            local_coords[:, 2],
        ] = True
        feature_volume[
            local_coords[:, 0],
            local_coords[:, 1],
            local_coords[:, 2],
        ] = X.astype(np.float32, copy=False)
        
        # Final safeguard before SLIC call.
        np.nan_to_num(feature_volume, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        segments = slic(
            image=feature_volume,
            n_segments=int(self.n_clusters),
            compactness=float(self.compactness),
            max_num_iter=int(self.max_iter),
            sigma=float(self.sigma),
            enforce_connectivity=bool(self.enforce_connectivity),
            mask=valid_mask,
            start_label=1,
            channel_axis=-1,
            spacing=self.spacing,
        )

        voxel_labels = segments[
            local_coords[:, 0],
            local_coords[:, 1],
            local_coords[:, 2],
        ].astype(np.int32)

        # Keep 0-indexed labels for compatibility with existing manager logic.
        self.labels_ = voxel_labels - 1
        self._fitted_sample_count = int(self.labels_.shape[0])
        self.model = "slic"
        return self

    def predict(
        self,
        X: np.ndarray,
        spatial_coords: Optional[np.ndarray] = None,
        mask_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return labels for the fitted subject.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            spatial_coords: Unused placeholder for interface compatibility.
            mask_array: Unused placeholder for interface compatibility.

        Returns:
            np.ndarray: 0-indexed labels with shape (n_samples,).
        """
        if self.labels_ is None or self._fitted_sample_count is None:
            raise ValueError("fit must be called before predict.")

        if X.shape[0] != self._fitted_sample_count:
            raise ValueError(
                "SLIC clustering is subject-specific and cannot predict labels "
                "for a different sample count. Re-run fit for each subject."
            )
        return self.labels_

    def find_optimal_clusters(
        self,
        X: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 10,
        methods: Optional[Union[List[str], str]] = None,
        show_progress: bool = True,
        spatial_coords: Optional[np.ndarray] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[int, Dict[str, List[float]]]:
        """
        Find optimal cluster number using spatially regularized features.

        This method evaluates K values on a SLIC-style feature embedding
        (appearance + compactness-weighted coordinates). It is used only for
        model selection in one-step mode.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            min_clusters: Minimum cluster number to evaluate.
            max_clusters: Maximum cluster number to evaluate.
            methods: Validation methods; supports silhouette, calinski_harabasz,
                davies_bouldin, inertia, kneedle.
            show_progress: Keep argument for interface compatibility.
            spatial_coords: Optional voxel coordinates for spatial regularization.
            spacing: Optional voxel spacing (z, y, x). Reserved for API compatibility.

        Returns:
            Tuple[int, Dict[str, List[float]]]: Best cluster number and score dict.
        """
        if min_clusters <= 0:
            raise ValueError("min_clusters must be positive.")
        if max_clusters <= min_clusters:
            raise ValueError("max_clusters must be greater than min_clusters.")
        if X.shape[0] < max_clusters:
            raise ValueError(
                f"Number of samples ({X.shape[0]}) must be greater than "
                f"max_clusters ({max_clusters})."
            )

        if methods is None:
            methods = ["silhouette"]
        if spacing is not None:
            self.spacing = spacing
        if isinstance(methods, str):
            methods = [methods]

        supported = {
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "inertia",
            "kneedle",
        }
        methods = [m for m in methods if m in supported]
        if not methods:
            raise ValueError("No valid validation method for SLIC.")

        embedded = self._build_regularized_features(X, spatial_coords)
        self.cluster_range = list(range(min_clusters, max_clusters + 1))
        self.scores = {m: [] for m in methods}

        for n_clusters in self.cluster_range:
            km = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = km.fit_predict(embedded)

            if "silhouette" in self.scores:
                if len(np.unique(labels)) > 1:
                    self.scores["silhouette"].append(silhouette_score(embedded, labels))
                else:
                    self.scores["silhouette"].append(0.0)

            if "calinski_harabasz" in self.scores:
                if len(np.unique(labels)) > 1:
                    self.scores["calinski_harabasz"].append(
                        calinski_harabasz_score(embedded, labels)
                    )
                else:
                    self.scores["calinski_harabasz"].append(0.0)

            if "davies_bouldin" in self.scores:
                if len(np.unique(labels)) > 1:
                    self.scores["davies_bouldin"].append(
                        davies_bouldin_score(embedded, labels)
                    )
                else:
                    self.scores["davies_bouldin"].append(0.0)

            if "inertia" in self.scores or "kneedle" in self.scores:
                inertia_val = float(km.inertia_)
                if "inertia" in self.scores:
                    self.scores["inertia"].append(inertia_val)
                if "kneedle" in self.scores:
                    self.scores["kneedle"].append(inertia_val)

        best_method = methods[0] if len(methods) == 1 else "_".join(methods)
        best_n_clusters = self.auto_select_best_n_clusters(self.scores, best_method)
        self.n_clusters = best_n_clusters
        return best_n_clusters, self.scores
