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

Label matching has exactly two operators here, chosen by what the maps
share:

* :func:`align_habitat_map` / :func:`habitat_stability` -- the two maps
  label the **same voxels** (retest, perturbation, another preprocessing
  chain). Ids are paired by maximal voxel overlap.
* :func:`align_habitat_maps_to_prototypes` -- the maps label **different
  subjects**. Every subject's habitats are matched one-to-one to shared
  prototypes (K = largest habitat count), so no reference subject is
  chosen and no subject's habitat count changes. Pass ``prototypes=`` to
  name new subjects with a trained definition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.contracts.habitat import HabitatMap
from habit.exceptions import HABITAPIError
from habit.kernels.habitat_label_match import (
    fit_feature_match_scale,
    habitat_dice_from_mapping,
    habitat_intensity_centroids,
    match_labels_by_overlap,
    match_rows_to_prototypes,
    present_habitat_ids,
    remap_label_array,
)

__all__ = [
    "HabitatPrototypeAlignment",
    "align_habitat_map",
    "align_habitat_maps_to_prototypes",
    "habitat_stability",
]

_PROTOTYPE_CALLER = "align_habitat_maps_to_prototypes"


def _as_image_array(image: Any, argument_name: str) -> np.ndarray:
    """Extract a numeric volume from an array or an ImageVolume-like object."""
    if image is None:
        raise HABITAPIError(f"{_PROTOTYPE_CALLER}: {argument_name} is required.")
    data = getattr(image, "data", image)
    return np.asarray(data)


def _centroids_from_model(model: Any, argument_name: str) -> np.ndarray:
    """Read a ``(n_habitats, n_features)`` centroid matrix from a fitted model."""
    centroids = getattr(model, "centroids", None)
    if centroids is None:
        raise HABITAPIError(
            f"{_PROTOTYPE_CALLER}: {argument_name} must provide .centroids "
            "(a HabitatModel)."
        )
    array = np.asarray(centroids, dtype=np.float64)
    if array.ndim != 2:
        raise HABITAPIError(
            f"{_PROTOTYPE_CALLER}: {argument_name}.centroids must be 2-D; "
            f"got {array.ndim}D."
        )
    return array


def _same_model_id(reference: HabitatMap, moving: HabitatMap) -> bool:
    """True when both maps already share a non-empty habitat-model id."""
    ref_id = str(reference.model_id or "")
    mov_id = str(moving.model_id or "")
    return bool(ref_id) and ref_id == mov_id


def align_habitat_map(
    reference: HabitatMap,
    moving: HabitatMap,
    *,
    force: bool = False,
) -> HabitatMap:
    """
    Remap ``moving`` habitat ids onto the ``reference`` id space.

    For two maps of the **same voxels** (retest, perturbation, another
    preprocessing chain, another reader): independently clustered maps
    permute integer ids, so ids are paired by maximal voxel overlap
    (Hungarian on the overlap counts, the Prior 2024 ``munkres`` step)
    and ``moving`` is rewritten into the reference ids. Maps that already
    share a ``model_id`` (the apply-saved-model path) are returned
    unchanged: those ids are already the same definition.

    To name habitats of **different** subjects use
    :func:`align_habitat_maps_to_prototypes`; overlap between two
    patients' voxels has no meaning.

    :func:`habitat_stability` uses the same pairing, so the compare figure
    and the Dice table share one correspondence. Do not feed an
    already-aligned map back into ``habitat_stability``; it pairs the
    original maps itself.

    Args:
        reference: Habitat map whose ids are the target space.
        moving: Independently labelled map of the same voxel grid.
        force: If True, align even when ``model_id`` already matches.
            Independent ``one_step`` / ``fit_predict`` runs on the same
            subject share a model_id (spec + subject-id digest, not image
            content) and need ``force=True``.

    Returns:
        A new map with remapped labels. ``model_id`` is taken from
        ``reference``. ``habitat_ids`` starts with the reference ids and
        appends leftover moving habitats (rewritten to unused ids after
        ``max(reference ids)``) when the moving map had more clusters.
        The input ``moving`` map is returned as-is when the model ids
        already match and ``force`` is False.

    Raises:
        HABITAPIError: If the grids differ.
    """
    if moving.label_array.shape != reference.label_array.shape:
        raise HABITAPIError(
            "align_habitat_map: moving map has shape "
            f"{moving.label_array.shape}, expected {reference.label_array.shape}."
        )
    if not force and _same_model_id(reference, moving):
        return moving
    reference_ids = tuple(int(habitat_id) for habitat_id in reference.habitat_ids)
    mapping = match_labels_by_overlap(
        np.asarray(reference.label_array), np.asarray(moving.label_array)
    )
    remapped = remap_label_array(
        np.asarray(moving.label_array), mapping, reserved_ids=reference_ids
    )
    reserved = set(reference_ids)
    leftover_ids = tuple(
        int(habitat_id)
        for habitat_id in present_habitat_ids(remapped).tolist()
        if int(habitat_id) not in reserved
    )
    return HabitatMap(
        subject_id=moving.subject_id,
        label_array=remapped,
        geometry=moving.geometry,
        model_id=reference.model_id,
        habitat_ids=reference_ids + leftover_ids,
        provenance=moving.provenance.derive(
            produced_by="align_habitat_map",
            spec_fingerprint="align_habitat_map:overlap",
        ),
    )


@dataclass(frozen=True)
class HabitatPrototypeAlignment:
    """
    Cohort habitat maps renamed onto shared prototypes.

    Returned by :func:`align_habitat_maps_to_prototypes`.

    Pass it back as ``prototypes=`` to name new subjects with the same
    definition (same prototypes, same scaler, same metric, same
    ``model_id``).

    Attributes:
        habitat_maps: Aligned maps, input order. Prototype ``k`` is habitat
            id ``k + 1`` in every map, so the same id means the same
            habitat across subjects. No subject loses or merges a habitat.
            A habitat stays unnamed only when ``max_distance`` is set (or
            when a subject has more habitats than frozen prototypes); it
            then keeps a subject-local id above ``K`` that is **not**
            comparable across subjects.
        prototypes: Shape ``(K, n_features)`` in the units of the input
            summaries (row ``k`` is habitat id ``k + 1``): the mean (median
            for ``metric="manhattan"``) of every subject summary assigned to
            that prototype. Descriptive; matching uses
            ``match_prototypes``. For frozen runs these are the reference's.
        feature_names: Column names of ``prototypes``: the models' or
            feature fields' names when available, else ``f0, f1, ...``.
        source: Where the habitat summaries came from: ``"models"``
            (fitted clustering centroids), ``"features"`` (per-habitat means
            of a supplied voxel feature volume / field), or ``"centroids"``
            (caller-supplied matrices).
        assignments: One row per input habitat with columns ``subject_id``,
            ``habitat_id`` (original id), ``prototype_id`` (new id, NA when
            left unnamed), and ``distance`` to the prototype in the
            matching space (see ``metric``).
        objective: Sum of matched metric costs plus the unmatched penalty,
            in the matching space.
        n_iter: Assign / update rounds of the winning start (1 when frozen).
        converged: False when ``max_iter`` stopped the search first.
        seed_subject_id: Subject whose summaries seeded the winning start,
            empty for frozen runs.
        metric: Matching distance (see
            :data:`~habit.kernels.habitat_label_match.PROTOTYPE_METRICS`).
        standardize: ``"none"`` or ``"zscore"``.
        reduction: Per-habitat reduction used for ``features=`` sources.
        location: Column means of the z-score (``None`` without z-score).
        scale: Column standard deviations of the z-score, or ``None``.
        match_prototypes: Prototypes in the matching space (after z-score;
            unit rows for cosine / correlation). Reused by frozen runs.
        model_id: Shared ``model_id`` written into every aligned map.
    """

    habitat_maps: Tuple[HabitatMap, ...]
    prototypes: np.ndarray
    feature_names: Tuple[str, ...]
    source: str
    assignments: pd.DataFrame
    objective: float
    n_iter: int
    converged: bool
    seed_subject_id: str
    metric: str
    standardize: str
    reduction: str
    location: Optional[np.ndarray]
    scale: Optional[np.ndarray]
    match_prototypes: np.ndarray
    model_id: str


def _field_habitat_means(
    habitat_map: HabitatMap,
    field: Any,
    reduction: str,
    argument_name: str,
) -> Tuple[List[int], np.ndarray]:
    """Per-habitat summary of a VoxelFeatureField-like object (rows + indices)."""
    caller = "align_habitat_maps_to_prototypes"
    values = np.asarray(field.values, dtype=np.float64)
    index = np.asarray(field.voxel_index, dtype=np.int64)
    labels = np.asarray(habitat_map.label_array)
    if index.ndim != 2 or index.shape[1] != labels.ndim:
        raise HABITAPIError(
            f"{caller}: {argument_name}.voxel_index must have shape "
            f"(n_voxels, {labels.ndim}); got {index.shape}."
        )
    if np.any(index < 0) or np.any(index >= np.asarray(labels.shape)):
        raise HABITAPIError(
            f"{caller}: {argument_name} indexes voxels outside the habitat "
            f"map grid {labels.shape} of subject {habitat_map.subject_id!r}."
        )
    # Habitat label of every feature row, read at the row's (z, y, x).
    row_labels = labels[tuple(index.T)]
    reducer = np.mean if reduction == "mean" else np.median
    ids: List[int] = []
    rows: List[np.ndarray] = []
    for habitat_id in habitat_map.habitat_ids:
        selector = row_labels == int(habitat_id)
        if not np.any(selector):
            if np.any(labels == int(habitat_id)):
                raise HABITAPIError(
                    f"{caller}: habitat {int(habitat_id)} of subject "
                    f"{habitat_map.subject_id!r} has voxels but none of them "
                    f"are rows of {argument_name}; the feature field must "
                    "cover the habitat ROI."
                )
            # Id listed but empty in this map: nothing to name.
            continue
        ids.append(int(habitat_id))
        rows.append(reducer(values[selector], axis=0))
    return ids, np.vstack(rows) if rows else np.empty((0, values.shape[1]))


def _volume_habitat_means(
    habitat_map: HabitatMap,
    volume: Any,
    reduction: str,
    argument_name: str,
) -> Tuple[List[int], np.ndarray]:
    """Per-habitat summary of a volume on the map grid (optional channel axis)."""
    caller = "align_habitat_maps_to_prototypes"
    labels = np.asarray(habitat_map.label_array)
    values = np.asarray(_as_image_array(volume, argument_name), dtype=np.float64)
    if values.shape[: labels.ndim] != labels.shape or values.ndim not in (
        labels.ndim,
        labels.ndim + 1,
    ):
        raise HABITAPIError(
            f"{caller}: {argument_name} has shape {values.shape}; expected the "
            f"habitat map grid {labels.shape}, optionally with one trailing "
            "feature axis."
        )
    try:
        present, block = habitat_intensity_centroids(
            values, labels, reduction=reduction  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise HABITAPIError(f"{caller}: {exc}") from exc
    row_of = {int(habitat_id): row for row, habitat_id in enumerate(present.tolist())}
    # Keep habitat_ids order; skip ids that are listed but have no voxels.
    ids = [int(h) for h in habitat_map.habitat_ids if int(h) in row_of]
    rows = [block[row_of[h]] for h in ids]
    return ids, np.vstack(rows) if rows else np.empty((0, block.shape[1]))


def _prototype_blocks(
    habitat_maps: Sequence[HabitatMap],
    models: Optional[Sequence[Any]],
    features: Optional[Sequence[Any]],
    centroids: Optional[Sequence[np.ndarray]],
    reduction: str,
) -> Tuple[str, List[List[int]], List[np.ndarray], Tuple[str, ...]]:
    """
    Build one habitat-summary block per map from exactly one source.

    Returns:
        ``(source, ids, blocks, feature_names)`` where ``blocks[s][i]`` is
        the summary of habitat ``ids[s][i]`` of map ``s``.
    """
    caller = "align_habitat_maps_to_prototypes"
    given = {
        name: value
        for name, value in (
            ("models", models),
            ("features", features),
            ("centroids", centroids),
        )
        if value is not None
    }
    if len(given) != 1:
        raise HABITAPIError(
            f"{caller}: pass exactly one of models= (fitted clustering "
            "centroids), features= (voxel features to average per habitat), "
            f"or centroids= (your own matrices); got {sorted(given) or 'none'}."
        )
    source, sources = next(iter(given.items()))
    if len(sources) != len(habitat_maps):
        raise HABITAPIError(
            f"{caller}: need one {source} entry per habitat map; got "
            f"{len(sources)} vs {len(habitat_maps)}."
        )
    all_ids: List[List[int]] = []
    blocks: List[np.ndarray] = []
    feature_names: Optional[Tuple[str, ...]] = None
    for index, (habitat_map, entry) in enumerate(zip(habitat_maps, sources)):
        argument_name = f"{source}[{index}]"
        names: Optional[Tuple[str, ...]] = None
        if source == "features":
            if hasattr(entry, "values") and hasattr(entry, "voxel_index"):
                ids, block = _field_habitat_means(
                    habitat_map, entry, reduction, argument_name
                )
            else:
                ids, block = _volume_habitat_means(
                    habitat_map, entry, reduction, argument_name
                )
            raw_names = getattr(entry, "feature_names", None)
        else:
            if source == "models":
                block = _centroids_from_model(entry, argument_name)
                raw_names = getattr(entry, "feature_names", None)
            else:
                block = np.asarray(entry, dtype=np.float64)
                raw_names = None
                if block.ndim != 2:
                    raise HABITAPIError(
                        f"{caller}: {argument_name} must be 2-D; got {block.ndim}D."
                    )
            ids = [int(habitat_id) for habitat_id in habitat_map.habitat_ids]
            if block.shape[0] != len(ids):
                raise HABITAPIError(
                    f"{caller}: subject {habitat_map.subject_id!r} has "
                    f"{len(ids)} habitat ids but {argument_name} has "
                    f"{block.shape[0]} rows; row i must be habitat_ids[i]."
                )
        if raw_names is not None:
            names = tuple(str(name) for name in raw_names)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                # Different features = different habitat definitions;
                # naming them against each other would be meaningless.
                raise HABITAPIError(
                    f"{caller}: {argument_name} uses features {list(names)}, "
                    f"but the first entry uses {list(feature_names)}. "
                    "Prototype matching needs one shared feature definition."
                )
        all_ids.append(ids)
        blocks.append(block)
    width = next((int(block.shape[1]) for block in blocks if block.size), 0)
    names_out = feature_names or tuple(f"f{column}" for column in range(width))
    return source, all_ids, blocks, names_out


def align_habitat_maps_to_prototypes(
    habitat_maps: Sequence[HabitatMap],
    *,
    models: Optional[Sequence[Any]] = None,
    features: Optional[Sequence[Any]] = None,
    centroids: Optional[Sequence[np.ndarray]] = None,
    metric: Literal["sqeuclidean", "manhattan", "cosine", "correlation"] = "sqeuclidean",
    reduction: Literal["mean", "median"] = "mean",
    max_distance: Optional[float] = None,
    standardize: Literal["none", "zscore"] = "none",
    max_iter: int = 100,
    prototypes: Optional[HabitatPrototypeAlignment] = None,
) -> HabitatPrototypeAlignment:
    """
    Give every subject's habitats one shared set of names (cohort matching).

    For per-subject (``one_step``) habitats each subject numbers its
    habitats in arbitrary order. Pairwise matching to one reference subject
    makes the names depend on who the reference is, and chains of pairwise
    matches need not agree (A→B→C versus A→C). This operator names **all**
    subjects at once against ``K`` shared prototypes:

    1. start from one subject's habitat summaries as prototypes;
    2. match every subject one-to-one onto the prototypes (Hungarian on
       the ``metric`` cost; two habitats of one tumour never share a
       name);
    3. move each prototype to the centre of its matched summaries (mean,
       median for ``"manhattan"``, mean direction for cosine /
       correlation);
    4. repeat until nothing changes. Every subject with ``K`` habitats is
       tried as the start and the tightest result is kept.

    This is the relabelling algorithm of Stephens (2000) for label
    switching, equivalently k-means with a cannot-link constraint inside
    each subject (Wagstaff et al. 2001). Prototype ids are sorted by the
    first feature (then the second, ...), so they do not depend on input
    order. With two subjects and the default metric the grouping equals
    pairwise Hungarian on squared Euclidean distance.

    ``K`` is the largest subject habitat count. Every habitat of every
    subject gets its own prototype id: no subject loses, gains, or merges
    a habitat. A subject with fewer habitats simply lacks some ids (zero
    volume in cohort tables). ``max_distance`` is off by default; when set,
    a habitat farther than that from every free prototype is left unnamed
    (``prototype_id`` NA, subject-local id above ``K``).

    What describes a habitat (pass exactly one):

    * ``models`` (default use) -- the fitted per-subject models;
      ``HabitatModel.centroids`` are the clustering centroids, row ``i``
      is ``habitat_ids[i]``. Names therefore follow the same features that
      defined the habitats.
    * ``features`` -- one voxel feature source per map: a
      :class:`~habit.contracts.VoxelFeatureField` (rows at
      ``voxel_index``), or a volume / ``ImageVolume`` on the map grid with
      an optional trailing feature axis. Each habitat is summarised by the
      ``reduction`` of its voxels.
    * ``centroids`` -- caller-built ``(n_habitats, n_features)`` matrices,
      rows in ``habitat_ids`` order.

    Values are compared as given, so they must be comparable across
    subjects: same features, computed so that values mean the same thing
    in every patient (for example relative enhancement, or a validated
    intensity normalisation). ``one_step`` models cluster raw voxel values
    by default, and raw MRI signal is **not** comparable across scanners
    or patients. ``standardize="zscore"`` rescales every column once on the
    pooled cohort summaries (unit balance when features have different
    units; it does not make incomparable signal comparable).

    Metric choice. ``"sqeuclidean"`` (default) and ``"manhattan"``
    (robust to one outlying habitat) compare feature **values**.
    ``"cosine"`` compares only the direction of the feature vector, so a
    weakly and a strongly enhancing habitat with the same ratio look
    identical; with one feature of constant sign every habitat looks the
    same. ``"correlation"`` compares only the shape of the feature profile;
    with two features every centred profile is one of two directions, and
    with one feature it is undefined (an error). No error is raised for
    few features otherwise: choose cosine / correlation only when many
    features describe a habitat and their shape, not their level, is the
    habitat definition. ``max_distance`` and ``distance`` are in the
    metric's units (Euclidean, L1, ``1 - cos``, ``1 - r``).

    Frozen prototypes. Pass a previous result as ``prototypes=`` to name
    a validation cohort or a new patient with a trained definition: the
    habitats are assigned once to the stored prototypes, nothing is
    refitted, and the aligned maps get the stored ``model_id``. The
    source feature names, ``metric``, ``standardize`` and ``reduction``
    must equal the stored ones, and a z-score reuses the stored
    ``location`` / ``scale``. A subject with more habitats than stored
    prototypes keeps the extra ones unnamed.

    Maps that already share one cohort model (two-step, direct pooling,
    apply-saved-model) are already named consistently and do not need
    this step.

    Args:
        habitat_maps: One map per subject (any grids; no voxel is compared
            across subjects).
        models: Fitted per-subject models, one per map. Their
            ``feature_names`` must agree.
        features: Voxel feature fields or volumes, one per map. Feature
            names (when present) must agree.
        centroids: Caller-supplied summary matrices, one per map.
        metric: ``"sqeuclidean"`` (default), ``"manhattan"``,
            ``"cosine"``, or ``"correlation"``.
        reduction: ``"mean"`` (default, the quantity a k-means centroid
            stores) or ``"median"``; used only with ``features``.
        max_distance: Optional distance above which a habitat is left
            unnamed instead of being forced onto a prototype. ``None``
            (default) names every habitat.
        standardize: ``"none"`` (default) or ``"zscore"`` on pooled rows.
        max_iter: Upper bound on assign / update rounds per start.
        prototypes: Optional earlier result whose prototypes are reused
            without refitting.

    Returns:
        A :class:`HabitatPrototypeAlignment`. Aligned maps share one
        ``model_id`` derived from the prototypes and the parameters (the
        stored one for frozen runs).

    Raises:
        HABITAPIError: If no source or more than one is given, sources do
            not line up with the maps, feature definitions differ, frozen
            settings disagree, or a cosine / correlation summary has zero
            length.
    """
    caller = _PROTOTYPE_CALLER
    if not habitat_maps:
        raise HABITAPIError(f"{caller}: at least one habitat map is required.")
    resolved_metric = str(metric).strip().lower()
    if resolved_metric not in ("sqeuclidean", "manhattan", "cosine", "correlation"):
        raise HABITAPIError(
            f"{caller}: metric must be 'sqeuclidean', 'manhattan', 'cosine', "
            f"or 'correlation'; got {metric!r}."
        )
    resolved_scale = str(standardize).strip().lower()
    if resolved_scale not in ("none", "zscore"):
        raise HABITAPIError(
            f"{caller}: standardize must be 'none' or 'zscore'; got {standardize!r}."
        )
    resolved_reduction = str(reduction).strip().lower()
    if resolved_reduction not in ("mean", "median"):
        raise HABITAPIError(
            f"{caller}: reduction must be 'mean' or 'median'; got {reduction!r}."
        )
    source, habitat_ids, raw_blocks, feature_names = _prototype_blocks(
        habitat_maps, models, features, centroids, resolved_reduction
    )

    if prototypes is not None:
        # A frozen definition is only the same definition when every
        # setting that shapes the matching space is the same.
        stored = {
            "feature_names": prototypes.feature_names,
            "metric": prototypes.metric,
            "standardize": prototypes.standardize,
            "reduction": prototypes.reduction,
        }
        given = {
            "feature_names": feature_names,
            "metric": resolved_metric,
            "standardize": resolved_scale,
            "reduction": resolved_reduction,
        }
        for key, stored_value in stored.items():
            if given[key] != stored_value:
                raise HABITAPIError(
                    f"{caller}: frozen prototypes were fitted with "
                    f"{key}={stored_value!r}, but this call uses "
                    f"{key}={given[key]!r}. Use the same setting, or refit."
                )
        location, scale = prototypes.location, prototypes.scale
    elif resolved_scale == "zscore":
        try:
            location, scale = fit_feature_match_scale(raw_blocks, method="zscore")
        except ValueError as exc:
            raise HABITAPIError(f"{caller}: {exc}") from exc
    else:
        location, scale = None, None
    if location is not None and scale is not None:
        match_blocks = [
            (block - location) / scale if block.size else block
            for block in raw_blocks
        ]
    else:
        match_blocks = raw_blocks
    try:
        match = match_rows_to_prototypes(
            match_blocks,
            metric=resolved_metric,  # type: ignore[arg-type]
            max_distance=max_distance,
            max_iter=max_iter,
            prototypes=None if prototypes is None else prototypes.match_prototypes,
        )
    except ValueError as exc:
        raise HABITAPIError(f"{caller}: {exc}") from exc

    n_proto = int(match.prototypes.shape[0])
    settings = (
        f"source={source}:metric={resolved_metric}:reduction={resolved_reduction}:"
        f"K={n_proto}:max_distance={max_distance}:standardize={resolved_scale}"
    )
    if prototypes is not None:
        report_prototypes = prototypes.prototypes
        model_id = prototypes.model_id
        fingerprint = f"{caller}:frozen={model_id}:{settings}"
        seed_subject_id = ""
    else:
        # Report prototypes in input units: centre of raw rows per
        # prototype (median for manhattan, mean otherwise). Canonical
        # order was fixed in the matching space and is kept here.
        width = int(match.prototypes.shape[1])
        report_prototypes = np.full((n_proto, width), np.nan, dtype=np.float64)
        members: List[List[np.ndarray]] = [[] for _ in range(n_proto)]
        for block, assignment in zip(raw_blocks, match.assignments):
            for row, prototype in zip(block, assignment.tolist()):
                if prototype >= 0:
                    members[prototype].append(row)
        center = np.median if resolved_metric == "manhattan" else np.mean
        for prototype, rows in enumerate(members):
            if rows:
                report_prototypes[prototype] = center(np.vstack(rows), axis=0)
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(match.prototypes).tobytes())
        digest.update(repr((feature_names, settings)).encode("utf-8"))
        model_id = f"prototype-{digest.hexdigest()[:16]}"
        fingerprint = f"{caller}:{settings}"
        seed_subject_id = str(habitat_maps[match.init_block].subject_id)
    prototype_ids = tuple(range(1, n_proto + 1))

    aligned: List[HabitatMap] = []
    records = []
    for habitat_map, ids, assignment, distance in zip(
        habitat_maps, habitat_ids, match.assignments, match.distances
    ):
        mapping = {
            habitat_id: int(prototype) + 1
            for habitat_id, prototype in zip(ids, assignment.tolist())
            if prototype >= 0
        }
        # Unnamed ids (max_distance, or more habitats than frozen
        # prototypes) move above K: 1..K are reserved for shared names.
        remapped = remap_label_array(
            np.asarray(habitat_map.label_array),
            mapping,
            reserved_ids=prototype_ids,
        )
        leftover_ids = tuple(
            int(habitat_id)
            for habitat_id in present_habitat_ids(remapped).tolist()
            if int(habitat_id) > n_proto
        )
        aligned.append(
            HabitatMap(
                subject_id=habitat_map.subject_id,
                label_array=remapped,
                geometry=habitat_map.geometry,
                model_id=model_id,
                habitat_ids=prototype_ids + leftover_ids,
                provenance=habitat_map.provenance.derive(
                    produced_by=caller,
                    spec_fingerprint=fingerprint,
                ),
            )
        )
        for habitat_id, prototype, dist in zip(
            ids, assignment.tolist(), distance.tolist()
        ):
            records.append(
                (
                    habitat_map.subject_id,
                    habitat_id,
                    int(prototype) + 1 if prototype >= 0 else pd.NA,
                    float(dist) if prototype >= 0 else np.nan,
                )
            )
    assignments = pd.DataFrame.from_records(
        records, columns=["subject_id", "habitat_id", "prototype_id", "distance"]
    )
    assignments["prototype_id"] = assignments["prototype_id"].astype("Int64")
    return HabitatPrototypeAlignment(
        habitat_maps=tuple(aligned),
        prototypes=report_prototypes,
        feature_names=feature_names,
        source=source,
        assignments=assignments,
        objective=float(match.objective),
        n_iter=int(match.n_iter),
        converged=bool(match.converged),
        seed_subject_id=seed_subject_id,
        metric=resolved_metric,
        standardize=resolved_scale,
        reduction=resolved_reduction,
        location=location,
        scale=scale,
        match_prototypes=match.prototypes,
        model_id=model_id,
    )


def habitat_stability(
    reference: HabitatMap,
    perturbed: Sequence[HabitatMap],
) -> pd.DataFrame:
    """
    Score habitat stability between a reference map and perturbed maps.

    Each perturbed map is paired to the reference by maximal voxel overlap
    (the Prior 2024 Hungarian / ``munkres`` step, the same pairing as
    :func:`align_habitat_map`), then ordinary Dice is computed on that
    pair: ``2 * intersection / (n_reference + n_matched)``, where the two
    counts are voxel sizes of one reference habitat and its matched
    perturbed habitat. Unmatched reference habitats (fewer clusters on
    the perturbed map) score Dice 0.

    This function does **not** rewrite the input maps. Pass the original
    independently clustered pair, not a map that was already remapped.

    Args:
        reference: Habitat map of the original subject.
        perturbed: Habitat maps computed independently on perturbed copies,
            each on the same voxel grid as ``reference``.

    Returns:
        Long-format DataFrame with one row per perturbation per reference
        habitat: ``perturbation`` (positional index), ``habitat_id``,
        ``matched_id`` (NA when unmatched), ``dice``, ``n_reference`` and
        ``n_matched`` voxel counts.

    Raises:
        HABITAPIError: If no perturbed map is given or the grids differ.
    """
    if not perturbed:
        raise HABITAPIError("habitat_stability: at least one perturbed map is required.")
    reference_labels = np.asarray(reference.label_array)
    records = []
    for index, moved in enumerate(perturbed):
        moved_labels = np.asarray(moved.label_array)
        if moved_labels.shape != reference_labels.shape:
            raise HABITAPIError(
                f"habitat_stability: perturbed map {index} has shape "
                f"{moved_labels.shape}, expected {reference_labels.shape}."
            )
        mapping = match_labels_by_overlap(reference_labels, moved_labels)
        for habitat_id, moved_id, dice, n_reference, n_moved in habitat_dice_from_mapping(
            reference_labels, moved_labels, mapping
        ):
            records.append(
                (index, habitat_id, moved_id, dice, n_reference, n_moved)
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
