# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Habitat stability under perturbation: matched Dice scores and label align.

When habitats are computed independently on the original and on a perturbed
image, the cluster labels are not comparable -- cluster 1 of the second fit
may correspond to cluster 3 of the first. The reference implementation
(Prior et al., Radiol Artif Intell 2024;6(2):e230118) matches the clusters
by maximal overlap (Hungarian assignment, their ``munkres`` step) and then
reports the Dice similarity of every matched pair. This module implements
exactly that on :class:`~habit.contracts.habitat.HabitatMap` objects; WHO
produced the maps (per-subject GMM, cohort model, any recipe) is not this
layer's concern.

:func:`align_habitat_map` is the in-memory remapper the test-retest recipe
deferred: habitat maps in, remapped map out. Default matching is by
feature-space centroids (the test-retest idea); overlap matching is the
same assignment ``habitat_stability`` uses for Dice.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from habit.contracts.habitat import HabitatMap
from habit.exceptions import HABITAPIError
from habit.kernels.habitat_label_match import (
    align_label_array,
    match_label_ids,
    present_habitat_ids,
)

__all__ = ["align_habitat_map", "habitat_stability"]

AlignMethod = Literal["centroid", "overlap"]


def _as_image_array(image: Any, argument_name: str) -> np.ndarray:
    """Extract a numeric volume from an array or an ImageVolume-like object."""
    if image is None:
        raise HABITAPIError(f"align_habitat_map: {argument_name} is required.")
    data = getattr(image, "data", image)
    return np.asarray(data)


def _same_model_id(reference: HabitatMap, moving: HabitatMap) -> bool:
    """True when both maps already share a non-empty habitat-model id."""
    ref_id = str(reference.model_id or "")
    mov_id = str(moving.model_id or "")
    return bool(ref_id) and ref_id == mov_id


def align_habitat_map(
    reference: HabitatMap,
    moving: HabitatMap,
    *,
    image: Optional[Any] = None,
    moving_image: Optional[Any] = None,
    method: AlignMethod = "centroid",
    reference_centroids: Optional[np.ndarray] = None,
    moving_centroids: Optional[np.ndarray] = None,
    force: bool = False,
) -> HabitatMap:
    """
    Remap ``moving`` habitat ids onto the ``reference`` id space.

    Independently clustered maps (two ``fit_predict`` runs) permute integer
    ids. This operator recovers the correspondence and returns a new
    :class:`~habit.contracts.habitat.HabitatMap` whose labels are comparable
    to ``reference``. Maps that already share a ``model_id`` (the
    apply-saved-model path) are returned unchanged: those ids are already
    the same definition.

    Default matching is **centroid** -- Hungarian assignment on Euclidean
    distance between per-habitat **mean** feature vectors (the same
    quantity k-means stores as a cluster centre). Pass the two fits'
    :attr:`~habit.contracts.habitat.HabitatModel.centroids` when available;
    otherwise mean image intensity, then spatial means.

    ``method="overlap"`` uses maximal voxel overlap. Use the same
    ``method`` in :func:`habitat_stability` so Dice is scored on the
    same pairing. Do not feed an already-aligned map back into
    ``habitat_stability``: Dice should be scored on the original pair
    (stability matches internally).

    Args:
        reference: Habitat map whose ids are the target space.
        moving: Independently labelled map to remap.
        image: Optional intensity volume for the reference map (and for
            the moving map when ``moving_image`` is omitted). Accepts a
            NumPy array or an object with ``.data`` (``ImageVolume``).
        moving_image: Optional intensity volume for the moving map.
        method: ``"centroid"`` (default) or ``"overlap"``.
        reference_centroids: Optional cluster centres of the reference fit,
            shape ``(n_habitats, n_features)``, rows in
            ``reference.habitat_ids`` order (row ``i`` is habitat
            ``habitat_ids[i]``).
        moving_centroids: Optional cluster centres of the moving fit.
        force: If True, align even when ``model_id`` already matches.
            Independent ``one_step`` / ``fit_predict`` runs on the same
            subject share a model_id (spec + subject-id digest, not image
            content) and need ``force=True``.

    Returns:
        A new map with remapped labels, ``model_id`` / ``habitat_ids``
        taken from ``reference`` so a later compare treats the pair as
        sharing a definition. The input ``moving`` map is returned as-is
        when the model ids already match and ``force`` is False.

    Raises:
        HABITAPIError: If the grids differ or centroid inputs are incomplete.
    """
    if moving.label_array.shape != reference.label_array.shape:
        raise HABITAPIError(
            "align_habitat_map: moving map has shape "
            f"{moving.label_array.shape}, expected {reference.label_array.shape}."
        )
    if not force and _same_model_id(reference, moving):
        return moving
    resolved = str(method).strip().lower()
    if resolved not in ("centroid", "overlap"):
        raise HABITAPIError(
            f"align_habitat_map: method must be 'centroid' or 'overlap'; "
            f"got {method!r}."
        )
    ref_image: Optional[np.ndarray] = None
    mov_image: Optional[np.ndarray] = None
    if image is not None:
        ref_image = _as_image_array(image, "image")
        mov_image = (
            _as_image_array(moving_image, "moving_image")
            if moving_image is not None
            else ref_image
        )
    elif moving_image is not None:
        raise HABITAPIError(
            "align_habitat_map: moving_image requires image for the reference."
        )
    try:
        remapped = align_label_array(
            np.asarray(reference.label_array),
            np.asarray(moving.label_array),
            image=ref_image,
            moving_image=mov_image,
            method=resolved,
            reference_centroids=reference_centroids,
            moving_centroids=moving_centroids,
            reference_ids=(
                np.asarray(reference.habitat_ids, dtype=np.int64)
                if reference_centroids is not None
                else None
            ),
            moving_ids=(
                np.asarray(moving.habitat_ids, dtype=np.int64)
                if moving_centroids is not None
                else None
            ),
        )
    except ValueError as exc:
        raise HABITAPIError(f"align_habitat_map: {exc}") from exc
    return HabitatMap(
        subject_id=moving.subject_id,
        label_array=remapped,
        geometry=moving.geometry,
        model_id=reference.model_id,
        habitat_ids=reference.habitat_ids,
        provenance=moving.provenance.derive(
            produced_by="align_habitat_map",
            spec_fingerprint=f"align_habitat_map:{resolved}",
        ),
    )


def habitat_stability(
    reference: HabitatMap,
    perturbed: Sequence[HabitatMap],
    *,
    method: AlignMethod = "overlap",
    image: Optional[Any] = None,
    moving_images: Optional[Sequence[Any]] = None,
    reference_centroids: Optional[np.ndarray] = None,
    moving_centroids: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> pd.DataFrame:
    """
    Score habitat stability between a reference map and perturbed maps.

    Each perturbed map is paired to the reference, then ordinary Dice is
    computed on that pair: ``2 * intersection / (n_reference + n_matched)``,
    where the two counts are voxel sizes of one reference habitat and its
    matched perturbed habitat. Unmatched reference habitats (fewer
    clusters on the perturbed map) score Dice 0.

    Default ``method="overlap"`` is the Prior 2024 Hungarian / overlap
    pairing. ``method="centroid"`` pairs by Hungarian assignment on
    per-habitat **mean** feature distance (explicit centroids, else mean
    intensity of ``image`` / ``moving_images``, else spatial means).
    Use the same ``method`` as :func:`align_habitat_map` so the compare
    figure and the Dice table share one correspondence.

    This function does **not** rewrite the input maps. Pass the original
    independently clustered pair, not a map that was already remapped.

    Args:
        reference: Habitat map of the original subject.
        perturbed: Habitat maps computed independently on perturbed copies,
            each on the same voxel grid as ``reference``.
        method: ``"overlap"`` (default) or ``"centroid"``.
        image: Optional intensity / feature volume for the reference map.
            Required for intensity-mean centroid matching when explicit
            centroids are omitted.
        moving_images: Optional per-perturbed intensity volumes. When
            omitted, ``image`` is reused for every perturbed map.
        reference_centroids: Optional explicit reference cluster centres.
        moving_centroids: Optional explicit centres, one array per
            perturbed map (``None`` entries fall back to image / spatial
            means).

    Returns:
        Long-format DataFrame with one row per perturbation per reference
        habitat: ``perturbation`` (positional index), ``habitat_id``,
        ``matched_id`` (NA when unmatched), ``dice``, ``n_reference`` and
        ``n_matched`` voxel counts.

    Raises:
        HABITAPIError: If no perturbed map is given, the grids differ,
            ``method`` is unknown, or centroid inputs are incomplete.
    """
    if not perturbed:
        raise HABITAPIError("habitat_stability: at least one perturbed map is required.")
    resolved = str(method).strip().lower()
    if resolved not in ("centroid", "overlap"):
        raise HABITAPIError(
            f"habitat_stability: method must be 'centroid' or 'overlap'; "
            f"got {method!r}."
        )
    if moving_images is not None and len(moving_images) != len(perturbed):
        raise HABITAPIError(
            "habitat_stability: moving_images must have one entry per "
            f"perturbed map; got {len(moving_images)} vs {len(perturbed)}."
        )
    if moving_centroids is not None and len(moving_centroids) != len(perturbed):
        raise HABITAPIError(
            "habitat_stability: moving_centroids must have one entry per "
            f"perturbed map; got {len(moving_centroids)} vs {len(perturbed)}."
        )
    reference_labels = np.asarray(reference.label_array)
    reference_ids = present_habitat_ids(reference_labels)
    ref_image: Optional[np.ndarray] = (
        _as_image_array(image, "image") if image is not None else None
    )
    records = []
    for index, moved in enumerate(perturbed):
        moved_labels = np.asarray(moved.label_array)
        if moved_labels.shape != reference_labels.shape:
            raise HABITAPIError(
                f"habitat_stability: perturbed map {index} has shape "
                f"{moved_labels.shape}, expected {reference_labels.shape}."
            )
        mov_image: Optional[np.ndarray] = None
        if moving_images is not None and moving_images[index] is not None:
            mov_image = _as_image_array(
                moving_images[index], f"moving_images[{index}]"
            )
        elif ref_image is not None:
            mov_image = ref_image
        mov_cent = None
        if moving_centroids is not None:
            mov_cent = moving_centroids[index]
        try:
            mapping = match_label_ids(
                reference_labels,
                moved_labels,
                image=ref_image,
                moving_image=mov_image,
                method=resolved,
                reference_centroids=reference_centroids,
                moving_centroids=mov_cent,
                reference_ids=(
                    np.asarray(reference.habitat_ids, dtype=np.int64)
                    if reference_centroids is not None
                    else None
                ),
                moving_ids=(
                    np.asarray(moved.habitat_ids, dtype=np.int64)
                    if mov_cent is not None
                    else None
                ),
            )
        except ValueError as exc:
            raise HABITAPIError(f"habitat_stability: {exc}") from exc
        # Invert {moving_id: reference_id} so each reference habitat looks up
        # its matched moving id (unmatched reference habitats score Dice 0).
        matched_moving = {ref_id: mov_id for mov_id, ref_id in mapping.items()}
        for habitat_id in reference_ids:
            habitat_id = int(habitat_id)
            n_reference = int(np.count_nonzero(reference_labels == habitat_id))
            if habitat_id in matched_moving:
                moved_id = int(matched_moving[habitat_id])
                n_moved = int(np.count_nonzero(moved_labels == moved_id))
                intersection = int(
                    np.count_nonzero(
                        (reference_labels == habitat_id) & (moved_labels == moved_id)
                    )
                )
                dice = (
                    2.0 * intersection / (n_reference + n_moved)
                    if n_reference + n_moved > 0
                    else 0.0
                )
                records.append(
                    (index, habitat_id, moved_id, dice, n_reference, n_moved)
                )
            else:
                records.append(
                    (index, habitat_id, None, 0.0, n_reference, 0)
                )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "perturbation",
            "habitat_id",
            "matched_id",
            "dice",
            "n_reference",
            "n_matched",
        ],
    )
