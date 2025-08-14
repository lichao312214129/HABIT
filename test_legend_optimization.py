#!/usr/bin/env python3
"""
Test script to demonstrate KM curve legend optimization
"""

import pandas as pd
import numpy as np
from habit.core.machine_learning.visualization.km_survival import KMSurvivalPlotter

# Create test data with more groups for better legend demonstration
np.random.seed(42)
n_samples = 100

# Generate survival data for 3 groups
data = []
for group_name in ['High Risk', 'Medium Risk', 'Low Risk']:
    n_group = n_samples // 3
    
    if group_name == 'High Risk':
        # Higher hazard - shorter survival times
        times = np.random.exponential(scale=20, size=n_group)
        event_rates = 0.8
    elif group_name == 'Medium Risk':
        # Medium hazard
        times = np.random.exponential(scale=35, size=n_group)
        event_rates = 0.6
    else:  # Low Risk
        # Lower hazard - longer survival times
        times = np.random.exponential(scale=50, size=n_group)
        event_rates = 0.4
    
    # Generate events (1) and censoring (0)
    events = np.random.binomial(1, event_rates, size=n_group)
    
    for i in range(n_group):
        data.append({
            'time': times[i],
            'event': events[i],
            'group': group_name
        })

df = pd.DataFrame(data)

# Initialize plotter
plotter = KMSurvivalPlotter(
    output_dir="./test_results",
    font_size=12,
    dpi=300
)

print("Testing different legend configurations...")

# Test 1: Default legend (best location)
print("1. Default legend (best location)")
fig1, ax1 = plotter.plot_km(
    df=df,
    time_col="time",
    event_col="event", 
    group_col="group",
    save_name="KM_test_default_legend.pdf",
    time_unit="Months",
    annotate_median=True,
    legend_loc="best",
    legend_ncol=1,
    legend_outside=False
)

# Test 2: Legend in upper right corner
print("2. Legend in upper right corner")
fig2, ax2 = plotter.plot_km(
    df=df,
    time_col="time",
    event_col="event",
    group_col="group", 
    save_name="KM_test_upper_right_legend.pdf",
    time_unit="Months",
    annotate_median=True,
    legend_loc="upper right",
    legend_ncol=1,
    legend_outside=False
)

# Test 3: Legend outside plot area
print("3. Legend outside plot area")
fig3, ax3 = plotter.plot_km(
    df=df,
    time_col="time",
    event_col="event",
    group_col="group",
    save_name="KM_test_outside_legend.pdf", 
    time_unit="Months",
    annotate_median=True,
    legend_loc="best",
    legend_ncol=1,
    legend_outside=True
)

# Test 4: Horizontal legend (2 columns)
print("4. Horizontal legend (2 columns)")
fig4, ax4 = plotter.plot_km(
    df=df,
    time_col="time",
    event_col="event",
    group_col="group",
    save_name="KM_test_horizontal_legend.pdf",
    time_unit="Months", 
    annotate_median=True,
    legend_loc="upper center",
    legend_ncol=2,
    legend_outside=False
)

print("Legend optimization tests completed! Check ./test_results/ for output files.")
print("\nLegend improvements include:")
print("- Enhanced visual styling (rounded corners, shadow, better transparency)")
print("- Intelligent positioning to avoid curve overlap")
print("- Customizable location and layout (ncol, outside placement)")
print("- Better font sizing and spacing")
print("- Thicker legend lines for better visibility")
