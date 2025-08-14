import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from habit.core.machine_learning.visualization.km_survival import KMSurvivalPlotter


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip() == "":
        return None
    return [v.strip() for v in value.split(",") if v.strip() != ""]


def _parse_float_pair(values: Optional[Sequence[str]]) -> Optional[Tuple[float, float]]:
    if values is None:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected two numbers, e.g., --xlim 0 60")
    return float(values[0]), float(values[1])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality Kaplan-Meier survival curves with risk table and statistical annotations."
    )
    parser.add_argument("--csv", required=True, type=str, help="Path to input CSV file")
    parser.add_argument("--time-col", required=True, type=str, help="Time/duration column name")
    parser.add_argument("--event-col", required=True, type=str, help="Event column name (1=event, 0=censored)")
    parser.add_argument("--group-col", required=True, type=str, help="Grouping column name")

    parser.add_argument("--output-dir", type=str, default="./results/km", help="Output directory")
    parser.add_argument("--save-name", type=str, default="KM_Curve.pdf", help="Output filename (pdf/tiff recommended)")
    parser.add_argument("--time-unit", type=str, default="Months", help="Time unit for x-axis label")

    parser.add_argument("--group-order", type=str, default=None, help="Group order, comma-separated, e.g., Low,High")
    parser.add_argument("--palette", type=str, default=None, help="Seaborn palette name or color list (comma-separated hex)")
    parser.add_argument("--hr-reference", type=str, default=None, help="Reference group for Cox HR")

    parser.add_argument("--no-ci", action="store_true", help="Disable confidence bands")
    parser.add_argument("--no-risk-table", action="store_true", help="Disable number-at-risk table")
    parser.add_argument("--show-hr", action="store_true", help="Enable Cox HR annotation")

    parser.add_argument("--figsize", nargs=2, metavar=("W", "H"), help="Figure size in inches, e.g., --figsize 7 6")
    parser.add_argument("--xlim", nargs=2, metavar=("MIN", "MAX"), help="x-axis limits, e.g., --xlim 0 60")
    parser.add_argument("--ylim", nargs=2, metavar=("MIN", "MAX"), help="y-axis limits, e.g., --ylim 0 1")

    parser.add_argument("--font-family", type=str, default="Arial", help="Font family")
    parser.add_argument("--font-size", type=int, default=11, help="Base font size")
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI for non-PDF formats")
    
    # Legend customization options
    parser.add_argument("--legend-loc", type=str, default="best", help="Legend location (best, upper right, lower left, etc.)")
    parser.add_argument("--legend-ncol", type=int, default=1, help="Number of columns in legend")
    parser.add_argument("--legend-outside", action="store_true", help="Place legend outside plot area")
    parser.add_argument("--annotate-median", action="store_true", help="Add median survival to legend labels")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load data
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    # Try to read CSV with different encodings to handle various file formats
    try:
        df = pd.read_csv(args.csv, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(args.csv, encoding='gbk')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(args.csv, encoding='latin-1')
            except UnicodeDecodeError:
                # Fall back to auto-detection or default system encoding
                df = pd.read_csv(args.csv, encoding='utf-8-sig')

    # Parse optional parameters
    group_order = _parse_list(args.group_order)
    palette: Optional[Sequence[str]]
    if args.palette and ("," in args.palette):
        palette = _parse_list(args.palette)
    else:
        palette = args.palette

    figsize = None if args.figsize is None else (float(args.figsize[0]), float(args.figsize[1]))
    xlim = _parse_float_pair(args.xlim)
    ylim = _parse_float_pair(args.ylim)

    # Initialize plotter
    plotter = KMSurvivalPlotter(
        output_dir=args.output_dir,
        dpi=args.dpi,
        font_family=args.font_family,
        font_size=args.font_size,
    )

    # Plot
    plotter.plot_km(
        df=df,
        time_col=args.time_col,
        event_col=args.event_col,
        group_col=args.group_col,
        save_name=args.save_name,
        time_unit=args.time_unit,
        group_order=group_order,
        palette=palette,
        show_ci=not args.no_ci,
        show_risk_table=not args.no_risk_table,
        show_hr=args.show_hr,
        hr_reference=args.hr_reference,
        figsize=(7.0, 6.0) if figsize is None else figsize,
        xlim=xlim,
        ylim=(0.0, 1.0) if ylim is None else ylim,
        annotate_median=args.annotate_median,
        legend_loc=args.legend_loc,
        legend_ncol=args.legend_ncol,
        legend_outside=args.legend_outside,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no command line arguments are provided, use the default config for debugging
        # This block sets sys.argv to use a default CSV and output directory for quick testing
        print("Debug mode: using default test.csv and output_dir=./results/km")
        sys.argv = [
            sys.argv[0],
            "--csv", "H:/results/ml_results/merged_all/merged_output_TestSet_event.csv",
            "--output-dir", "H:/results/ml_results/km",
            "--time-col", "随访时长（month）",
            "--event-col", "是否复发(1，是；0，否)",
            "--group-col", "group",
            "--show-hr",
            "--save-name", "KM_test.pdf"
        ]
    args = build_argparser()
    print(args)
    main(args)

