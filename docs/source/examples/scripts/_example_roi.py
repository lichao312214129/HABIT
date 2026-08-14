#!/usr/bin/env python
"""
Tiny I/O helpers for Examples gallery scripts (not shown in Sphinx pages).

Crops clinical demo volumes to a padded ROI / habitat bbox so sklearn-short
demos stay interactive. Synthetic cohorts are returned unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from habit import cohort_from_directory, make_synthetic_cohort
from habit.contracts import ArrayImageRef, Geometry, Subject
from habit.contracts.subject import Cohort

REPO_ROOT = Path(__file__).resolve().parents[4]
DEMO_PREPROCESSED = REPO_ROOT / "demo_data" / "preprocessed"
EXAMPLES_IMG_DIR = (
    Path(__file__).resolve().parents[2] / "_static" / "images" / "examples"
)


def crop_pair(
    volume: np.ndarray,
    mask_or_labels: np.ndarray,
    *,
    pad: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop ``volume`` and ``mask_or_labels`` to a padded foreground bbox.

    Args:
        volume: Anatomy or matching companion array ``(z, y, x)``.
        mask_or_labels: ROI mask or habitat labels (foreground ``> 0``).
        pad: Voxel padding on each side (clipped to bounds).

    Returns:
        Cropped ``(volume, mask_or_labels)`` sharing one shape.
    """
    foreground = mask_or_labels > 0
    if not np.any(foreground):
        raise RuntimeError("No foreground voxels to crop.")
    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    slices = []
    for axis, (lo, hi) in enumerate(zip(mins, maxs)):
        start = max(0, int(lo) - pad)
        stop = min(int(mask_or_labels.shape[axis]), int(hi) + pad + 1)
        slices.append(slice(start, stop))
    indexer = tuple(slices)
    return volume[indexer].copy(), mask_or_labels[indexer].copy()


def one_subject_cohort(
    *,
    demo_modalities: Sequence[str] = ("LAP",),
    demo_roi: str = "LAP",
    synthetic_modalities: Sequence[str] = ("T1",),
    synthetic_shape: Tuple[int, int, int] = (40, 40, 40),
    rng: int = 0,
) -> Tuple[Cohort, Tuple[str, ...], bool]:
    """
    Load one subject from ``demo_data/preprocessed`` or a synthetic fallback.

    Args:
        demo_modalities: Modality keys when demo_data is present.
        demo_roi: Mask key for the demo layout.
        synthetic_modalities: Modality keys for the synthetic fallback.
        synthetic_shape: Volume shape for the synthetic subject.
        rng: Synthetic RNG seed.

    Returns:
        ``(cohort, modalities, from_demo)`` where ``cohort`` has length 1.
    """
    if DEMO_PREPROCESSED.is_dir():
        modalities = tuple(demo_modalities)
        cohort = cohort_from_directory(
            DEMO_PREPROCESSED,
            modalities=modalities,
            roi=demo_roi,
        )[:1]
        return cohort, modalities, True
    modalities = tuple(synthetic_modalities)
    cohort = make_synthetic_cohort(
        n_subjects=1,
        modalities=modalities,
        shape=synthetic_shape,
        rng=rng,
    )
    return cohort, modalities, False


def cropped_subject_from(
    subject: Subject,
    modality: str,
    *,
    pad: int = 5,
) -> Tuple[Subject, np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Rebuild a single-modality ``Subject`` cropped to the ROI bbox.

    Args:
        subject: Source subject (demo or synthetic).
        modality: Image / mask key to keep.
        pad: Crop padding in voxels.

    Returns:
        ``(cropped_subject, image, mask, spacing_xyz)``.
    """
    volume = subject.image(modality)
    image = np.asarray(volume.data, dtype=np.float32)
    mask = np.asarray(subject.mask(modality).data, dtype=np.uint8)
    spacing_xyz = tuple(float(v) for v in volume.spacing)
    image_c, mask_c = crop_pair(image, mask, pad=pad)
    geometry = Geometry.from_array(image_c.shape, spacing=spacing_xyz)
    cropped = Subject(
        subject_id=subject.subject_id,
        images={modality: ArrayImageRef(array=image_c, geometry=geometry)},
        masks={
            modality: ArrayImageRef(
                array=(mask_c > 0).astype(np.int32),
                geometry=geometry,
            )
        },
    )
    return cropped, image_c, mask_c, spacing_xyz


def examples_image_dir() -> Path:
    """Return ``docs/source/_static/images/examples`` (created if needed)."""
    EXAMPLES_IMG_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_IMG_DIR


def copy_out_figures_to_gallery(
    filenames: Sequence[str],
    *,
    out_dir: Path = Path("out"),
) -> Dict[str, Path]:
    """
    Copy PNGs already written under ``out/`` into the Examples gallery.

    This does **not** re-plot. The site image must be the same composition
    as the Sphinx-visible ``# BEGIN example`` … ``# END example`` block.

    Args:
        filenames: Basenames such as ``one_step_overlay.png``.
        out_dir: Directory the visible example wrote (default ``out``).

    Returns:
        Mapping of basename → gallery path for files that existed.

    Raises:
        FileNotFoundError: When none of the requested files exist.
    """
    import shutil

    gallery = examples_image_dir()
    written: Dict[str, Path] = {}
    missing: list[str] = []
    for name in filenames:
        src = Path(out_dir) / name
        if not src.is_file():
            missing.append(name)
            continue
        dest = gallery / name
        shutil.copy2(src, dest)
        written[name] = dest
    if not written:
        raise FileNotFoundError(
            "copy_out_figures_to_gallery: no files found under "
            f"{Path(out_dir).resolve()}: {', '.join(filenames)}"
        )
    if missing:
        print("Gallery copy skipped (missing in out/): " + ", ".join(missing))
    print("Gallery copy: " + ", ".join(written))
    return written


def save_example_figure(fig: object, filename: str, *, dpi: int = 300) -> Path:
    """
    Save a matplotlib figure under the Examples gallery image directory.

    Args:
        fig: Matplotlib ``Figure`` (must expose ``savefig``).
        filename: Basename only (e.g. ``graph_habitat_network_2d.png``).
        dpi: Output resolution.

    Returns:
        Absolute path written.
    """
    import matplotlib.pyplot as plt

    out = examples_image_dir() / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _bbox_indexer(
    mask: np.ndarray,
    *,
    pad: int = 5,
) -> Tuple[slice, ...]:
    """
    Build a padded bbox indexer for ``mask > 0``.

    Args:
        mask: Foreground indicator / label volume.
        pad: Voxel padding on each side.

    Returns:
        Tuple of slices suitable for NumPy advanced indexing.
    """
    foreground = mask > 0
    if not np.any(foreground):
        raise RuntimeError("_bbox_indexer: no foreground voxels.")
    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    slices = []
    for axis, (lo, hi) in enumerate(zip(mins, maxs)):
        start = max(0, int(lo) - pad)
        stop = min(int(mask.shape[axis]), int(hi) + pad + 1)
        slices.append(slice(start, stop))
    return tuple(slices)


def save_habitat_study_figures(
    cohort: Cohort,
    result: object,
    *,
    prefix: str,
    modality: Optional[str] = None,
    map_index: int = 0,
    compare_labels: Optional[np.ndarray] = None,
    compare_titles: Tuple[str, str] = ("Reference", "Predict"),
) -> Dict[str, Path]:
    """
    Write the standard habitat-core gallery PNGs for one ``StudyResult``.

    Produces (when data allows): overlay, partition triptych (two-step),
    volume fractions, MSI heatmap, ITH summary, cluster-validation curves,
    and optional label-compare.

    Args:
        cohort: Cohort used for the run (anatomy lookup).
        result: ``StudyResult``-like object with ``habitat_maps`` / optional
            ``units`` / ``habitat_model``.
        prefix: Filename stem prefix (e.g. ``two_step`` → ``two_step_overlay.png``).
        modality: Anatomy key; first image on the subject when omitted.
        map_index: Which subject map to illustrate.
        compare_labels: Optional second label map (same shape) for compare.
        compare_titles: Panel titles for the compare figure.

    Returns:
        Mapping of logical name → written path.
    """
    from habit.kernels.habitat_metrics import (
        habitat_region_stats,
        habitat_volume_fractions,
        ith_score,
        spatial_interaction_matrix,
    )
    from habit.viz import (
        plot_cluster_validation_from_report,
        plot_habitat_label_compare,
        plot_habitat_overlay,
        plot_habitat_volume_fractions,
        plot_ith_summary,
        plot_msi_matrix,
        plot_partition_triptych,
    )

    maps = list(getattr(result, "habitat_maps", ()) or ())
    if not maps:
        raise RuntimeError("save_habitat_study_figures: result has no habitat_maps.")
    habitat_map = maps[map_index]
    subject_id = getattr(habitat_map, "subject_id", None)
    subject = cohort[0]
    for item in cohort:
        if getattr(item, "subject_id", None) == subject_id:
            subject = item
            break
    if modality is None:
        modality = next(iter(subject.images))
    image = np.asarray(subject.image(modality).data)
    labels = np.asarray(habitat_map.label_array, dtype=np.int32)
    indexer = _bbox_indexer(labels)
    image_c = image[indexer].copy()
    labels_c = labels[indexer].copy()

    written: Dict[str, Path] = {}
    written["overlay"] = save_example_figure(
        plot_habitat_overlay(
            image_c, labels_c, axis=0, title=f"{prefix}: habitats"
        ),
        f"{prefix}_overlay.png",
    )

    units = list(getattr(result, "units", ()) or ())
    if units and map_index < len(units):
        sv = np.asarray(units[map_index].label_array, dtype=np.int32)
        if sv.shape == labels.shape:
            written["triptych"] = save_example_figure(
                plot_partition_triptych(
                    image_c, sv[indexer].copy(), labels_c, axis=0
                ),
                f"{prefix}_triptych.png",
            )

    ids = tuple(sorted({int(v) for v in labels_c.ravel() if int(v) != 0}))
    if ids:
        frac = habitat_volume_fractions(labels_c, ids)
        written["volume"] = save_example_figure(
            plot_habitat_volume_fractions(frac),
            f"{prefix}_volume_fractions.png",
        )
        # MSI matrix is indexed 0..max_label; tick labels must cover every
        # habitat slot even when some IDs are absent in this crop/subject.
        n_classes = max(ids) + 1
        msi_ids = tuple(range(1, n_classes))
        msi = spatial_interaction_matrix(labels_c, n_classes=n_classes)
        written["msi"] = save_example_figure(
            plot_msi_matrix(msi, habitat_ids=msi_ids),
            f"{prefix}_msi_matrix.png",
        )
        written["ith"] = save_example_figure(
            plot_ith_summary(
                ith_score(labels_c),
                per_habitat=habitat_region_stats(labels_c),
            ),
            f"{prefix}_ith_summary.png",
        )

    report = None
    model = getattr(result, "habitat_model", None)
    if model is not None:
        report = (getattr(model, "preprocessing_state", None) or {}).get(
            "selection_report"
        )
    if report is None:
        for _sid, sm in (getattr(result, "subject_models", None) or {}).items():
            report = (getattr(sm, "preprocessing_state", None) or {}).get(
                "selection_report"
            )
            if report:
                break
    if report:
        written["validation"] = save_example_figure(
            plot_cluster_validation_from_report(report),
            f"{prefix}_cluster_validation.png",
        )

    if compare_labels is not None:
        cmp = np.asarray(compare_labels, dtype=np.int32)
        if cmp.shape != labels.shape:
            raise RuntimeError(
                "save_habitat_study_figures: compare_labels shape "
                f"{cmp.shape} != labels {labels.shape}."
            )
        written["compare"] = save_example_figure(
            plot_habitat_label_compare(
                image_c,
                labels_c,
                cmp[indexer].copy(),
                titles=compare_titles,
                axis=0,
            ),
            f"{prefix}_label_compare.png",
        )

    print(
        f"Gallery figures ({prefix}): "
        + ", ".join(p.name for p in written.values())
    )
    return written


def glcm_field(
    subject: Subject,
    modality: str,
    *,
    features: Sequence[str] = ("Contrast", "Correlation", "JointEntropy"),
    kernel_radius: int = 1,
    bin_width: float = 25.0,
) -> object:
    """
    Run the built-in ``voxel_radiomics`` extractor for a small GLCM set.

    Kept out of Sphinx literalincludes so gallery pages stay sklearn-short.

    Args:
        subject: Cropped single-modality subject.
        modality: Image / mask key.
        features: GLCM feature names passed to PyRadiomics.
        kernel_radius: Voxel radiomics kernel radius.
        bin_width: Intensity bin width.

    Returns:
        Sparse :class:`~habit.contracts.habitat.VoxelFeatureField`.
    """
    import habit.domain  # registers built-in extractors
    from habit.domain import VoxelFeatureExtractorRegistry

    return VoxelFeatureExtractorRegistry.create(
        "voxel_radiomics",
        modality=modality,
        kernel_radius=kernel_radius,
        params={
            "imageType": {"Original": {}},
            "featureClass": {"glcm": list(features)},
            "setting": {"binWidth": bin_width},
        },
    )(subject)

