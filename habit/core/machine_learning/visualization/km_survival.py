"""
Kaplan-Meier survival plotting utilities

Features aligned to top-tier medical imaging journals requirements:
- Publication-quality styling (font, line widths, vector export)
- KM curves with confidence bands
- Log-rank p-value (two-group or multi-group)
- Cox model hazard ratio (HR) with 95% CI (binary; pairwise vs reference if >2 groups)
- Number-at-risk table
- Median survival per group (optional)

Usage example (programmatic):

    plotter = KMSurvivalPlotter(output_dir="./results/km")
    fig, ax = plotter.plot_km(
        df=dataframe,
        time_col="os_time",
        event_col="os_event",
        group_col="risk_group",
        save_name="KM_OS.pdf",
        time_unit="Months",
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# from lifelines import KaplanMeierFitter, CoxPHFitter
# from lifelines.statistics import logrank_test, multivariate_logrank_test
# from lifelines.plotting import add_at_risk_counts
from matplotlib.lines import Line2D


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_figure(fig: plt.Figure, output_dir: str, save_name: str, dpi: int) -> None:
    file_ext = os.path.splitext(save_name)[1].lower()
    output_path = os.path.join(output_dir, save_name)
    if file_ext == ".pdf":
        fig.savefig(output_path, bbox_inches="tight")
    elif file_ext in [".tif", ".tiff"]:
        fig.savefig(
            output_path,
            bbox_inches="tight",
            dpi=dpi,
            format="tif",
            compression="tiff_lzw",
        )
    else:
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)


def _as_color_list(palette: Optional[Sequence[str]], n: int) -> List:
    if palette is None:
        return sns.color_palette("Set1", n)
    if isinstance(palette, str):
        return sns.color_palette(palette, n)
    # assume list-like of colors
    if len(palette) < n:
        # repeat if not enough colors
        times = int(np.ceil(n / len(palette)))
        return list(palette) * times
    return list(palette)[:n]


@dataclass
class KMSurvivalPlotter:
    output_dir: str
    dpi: int = 600
    font_family: str = "Arial"
    font_size: int = 11

    def __post_init__(self) -> None:
        _ensure_output_dir(self.output_dir)
        # Publication-quality defaults
        mpl.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.font_size,
                "axes.linewidth": 1.2,
                "axes.labelsize": self.font_size,
                "axes.titlesize": self.font_size + 1,
                "xtick.labelsize": self.font_size - 1,
                "ytick.labelsize": self.font_size - 1,
                "legend.fontsize": self.font_size - 1,
                "pdf.fonttype": 42,  # embed TrueType
                "ps.fonttype": 42,
                "savefig.dpi": self.dpi,
                "figure.dpi": self.dpi,
            }
        )

    def plot_km(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: str,
        save_name: str = "KM_Curve.pdf",
        time_unit: str = "Months",
        group_order: Optional[Sequence] = None,
        palette: Optional[Sequence[str]] = None,
        show_ci: bool = True,
        show_risk_table: bool = True,
        show_hr: bool = False,
        hr_reference: Optional[str] = None,
        figsize: Tuple[float, float] = (5.5, 5.0),
        y_label: str = "Survival probability",
        x_label: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Tuple[float, float] = (0.0, 1.0),
        annotate_median: bool = False,
        legend_loc: str = "best",
        legend_ncol: int = 1,
        legend_outside: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot KM curves by groups with risk table and annotations.

        Args:
            df: DataFrame containing survival data
            time_col: Duration column (numeric)
            event_col: Event column (1=event, 0=censored)
            group_col: Grouping column (categorical)
            save_name: Output file name; extension controls format
            time_unit: Label for x-axis (e.g., 'Months')
            group_order: Optional manual ordering of groups
            palette: Matplotlib/seaborn palette name or list of colors
            show_ci: Whether to draw confidence bands
            show_risk_table: Whether to render number-at-risk table
            show_hr: Whether to compute and display HR
            hr_reference: Reference group for HR (default: first in order)
            figsize: Figure size in inches
            y_label: Y-axis label
            x_label: X-axis label (defaults to time_unit)
            xlim: Optional x-axis range
            ylim: Y-axis range
            annotate_median: Add median survival to legend label
            legend_loc: Legend location ('best', 'upper right', 'lower left', etc.)
            legend_ncol: Number of columns in legend
            legend_outside: Whether to place legend outside the plot area
        """
        # Validate columns
        for col in [time_col, event_col, group_col]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        work_df = (
            df[[time_col, event_col, group_col]]
            .dropna()
            .copy()
        )

        # Prepare groups
        if group_order is None:
            groups = list(pd.unique(work_df[group_col]))
        else:
            groups = [g for g in group_order if g in set(work_df[group_col])]
        if len(groups) < 1:
            raise ValueError("No groups available after filtering")

        colors = _as_color_list(palette, len(groups))

        # Set up figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Fit KM per group
        km_fitters: List[KaplanMeierFitter] = []
        legend_labels: List[str] = []
        
        # Define censor marker style (can be customized)
        censor_styles = {"ms": 4, "marker": "+"}

        for idx, group_value in enumerate(groups):
            mask = work_df[group_col] == group_value
            if not np.any(mask):
                continue
            t = work_df.loc[mask, time_col].astype(float).values
            e = work_df.loc[mask, event_col].astype(int).values

            kmf = KaplanMeierFitter(label=str(group_value))
            kmf.fit(t, event_observed=e)
            km_fitters.append(kmf)

            median_txt = ""
            if annotate_median:
                try:
                    median_val = kmf.median_survival_time_
                    if np.isfinite(median_val):
                        median_txt = f" (median {median_val:.1f})"
                except Exception:
                    median_txt = ""
            legend_labels.append(f"{group_value}{median_txt}")

            kmf.plot_survival_function(
                ax=ax,
                ci_show=show_ci,
                linewidth=1.0,
                color=colors[idx],
                show_censors=True,
                censor_styles=censor_styles,
            )

        # Axes formatting
        if x_label is None:
            x_label = f"Time ({time_unit})"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(True, linestyle="--", alpha=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Optimized Legend
        self._setup_legend(ax, legend_labels, colors[:len(legend_labels)], censor_styles, legend_loc, legend_ncol, legend_outside)

        # Statistical annotations
        p_text = self._compute_logrank_text(work_df, time_col, event_col, group_col)
        hr_text = self._compute_hr_text(work_df, time_col, event_col, group_col, groups, show_hr, hr_reference)

        annotation_lines = []
        if p_text:
            annotation_lines.append(p_text)
        if hr_text:
            annotation_lines.append(hr_text)
        if annotation_lines:
            ax.text(
                0.98,
                0.04 if show_risk_table else 0.06,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=self.font_size - 1,
                bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.4"),
            )

        # Number at risk table
        if show_risk_table and km_fitters:
            try:
                # Use default labels from fitters to avoid attribute errors across lifelines versions
                add_at_risk_counts(*km_fitters, ax=ax)
            except Exception:
                # Fallback: ignore risk table errors to avoid breaking plotting
                pass

        fig.tight_layout()
        _save_figure(fig, self.output_dir, save_name, self.dpi)
        return fig, ax

    # ---- Internals ----
    def _compute_logrank_text(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: str,
    ) -> str:
        groups = pd.unique(df[group_col])
        if len(groups) < 2:
            return ""
        try:
            if len(groups) == 2:
                g1, g2 = groups.tolist()
                d1 = df[df[group_col] == g1]
                d2 = df[df[group_col] == g2]
                res = logrank_test(
                    d1[time_col], d2[time_col], event_observed_A=d1[event_col], event_observed_B=d2[event_col]
                )
                return f"Log-rank p = {res.p_value:.3g}"
            else:
                res = multivariate_logrank_test(
                    event_durations=df[time_col],
                    groups=df[group_col],
                    event_observed=df[event_col],
                )
                return f"Log-rank p = {res.p_value:.3g}"
        except Exception:
            return ""

    def _compute_hr_text(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        group_col: str,
        groups: Sequence,
        enabled: bool,
        hr_reference: Optional[str],
    ) -> str:
        if not enabled or len(groups) < 2:
            return ""
        try:
            # Prepare Cox input
            cox_df = df[[time_col, event_col, group_col]].copy()
            cox_df[time_col] = cox_df[time_col].astype(float)
            cox_df[event_col] = cox_df[event_col].astype(int)

            # Reference handling
            if hr_reference is None or hr_reference not in set(cox_df[group_col]):
                reference = groups[0]
            else:
                reference = hr_reference

            # Create dummies, drop reference
            dummies = pd.get_dummies(cox_df[group_col], drop_first=False)
            if reference not in dummies.columns:
                # Should not happen, fallback to drop_first=True
                dummies = pd.get_dummies(cox_df[group_col], drop_first=True)
                ref_dropped = True
            else:
                # Drop reference column to set it as baseline in Cox
                dummies = dummies.drop(columns=[reference])
                ref_dropped = False

            design = pd.concat([cox_df[[time_col, event_col]], dummies], axis=1)

            cph = CoxPHFitter()
            cph.fit(
                design,
                duration_col=time_col,
                event_col=event_col,
                show_progress=False,
                robust=False,
            )

            summary = cph.summary
            # Binary case
            if summary.shape[0] == 1:
                hr = float(np.exp(summary.loc[summary.index[0], "coef"]))
                lower = float(np.exp(summary.loc[summary.index[0], "coef lower 95%"]))
                upper = float(np.exp(summary.loc[summary.index[0], "coef upper 95%"]))
                return f"HR (95% CI) = {hr:.2f} ({lower:.2f}-{upper:.2f})"

            # Multi-group: report pairwise vs reference on one line if possible
            parts: List[str] = []
            for idx in summary.index:
                hr = float(np.exp(summary.loc[idx, "coef"]))
                lower = float(np.exp(summary.loc[idx, "coef lower 95%"]))
                upper = float(np.exp(summary.loc[idx, "coef upper 95%"]))
                # idx is the dummy column name, e.g., 'GroupB'
                comp = idx if ref_dropped else idx
                parts.append(f"{comp} vs {reference}: {hr:.2f} ({lower:.2f}-{upper:.2f})")
            return "; ".join(parts)
        except Exception:
            return ""

    def _setup_legend(
        self,
        ax: plt.Axes,
        legend_labels: List[str],
        colors: List,
        censor_styles: Dict,
        legend_loc: str,
        legend_ncol: int,
        legend_outside: bool,
    ) -> None:
        """
        Setup optimized legend with enhanced styling and positioning options.
        
        Args:
            ax: Matplotlib axes object
            legend_labels: List of legend labels
            colors: List of colors corresponding to each legend entry
            censor_styles: Dictionary containing censor marker styles
            legend_loc: Legend location string
            legend_ncol: Number of columns for legend
            legend_outside: Whether to place legend outside plot area
        """
        if not legend_labels:
            return
            
        # Create custom line handles for consistent legend appearance
        custom_handles = []
        # Extract marker info from censor_styles
        marker_type = censor_styles.get("marker", "+")
        marker_size = censor_styles.get("ms", 4) * 2  # Scale up for legend visibility
        
        for i, (label, color) in enumerate(zip(legend_labels, colors)):
            line_handle = Line2D([0], [0], color=color, linewidth=2.5, 
                               marker=marker_type, markersize=marker_size, 
                               markerfacecolor=color, markeredgecolor=color, 
                               markeredgewidth=2, label=label)
            custom_handles.append(line_handle)
            
        # Configure legend properties for better visual appearance
        legend_props = {
            "handles": custom_handles,  # Use custom handles instead of default
            "frameon": True,
            "fancybox": True,  # Rounded corners
            "shadow": True,    # Drop shadow
            "framealpha": 0.95,  # Slightly more opaque
            "facecolor": "white",
            "edgecolor": "lightgray",
            "ncol": legend_ncol,
            "fontsize": self.font_size - 1,
            "columnspacing": 1.2,  # Space between columns
            "handletextpad": 0.5,  # Space between marker and text
            "handlelength": 1.5,   # Length of legend handles
        }
        
        if legend_outside:
            # Place legend outside the plot area (to the right)
            legend_props.update({
                "bbox_to_anchor": (1.02, 1),
                "loc": "upper left",
                "borderaxespad": 0
            })
        else:
            # Intelligent positioning within plot area
            if legend_loc == "best":
                # Use matplotlib's automatic best location
                legend_props["loc"] = "best"
            else:
                # Use specified location
                legend_props["loc"] = legend_loc
                
            # For locations that might overlap with curves, add some padding
            if legend_loc in ["upper right", "lower right", "center right"]:
                legend_props["bbox_to_anchor"] = (0.98, 0.98) if "upper" in legend_loc else (0.98, 0.02)
                legend_props["loc"] = "upper right" if "upper" in legend_loc else "lower right"
            elif legend_loc in ["upper left", "lower left", "center left"]:
                legend_props["bbox_to_anchor"] = (0.02, 0.98) if "upper" in legend_loc else (0.02, 0.02)
                legend_props["loc"] = "upper left" if "upper" in legend_loc else "lower left"
        
        # Create the legend
        legend = ax.legend(**legend_props)
        
        # Additional styling for better appearance
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(0.8)  # Set border line width after legend creation


