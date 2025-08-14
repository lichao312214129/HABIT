#!/usr/bin/env python3
"""
Simple test to verify KM curve legend optimization
"""

import pandas as pd
import os

# Create output directory
os.makedirs("test_results", exist_ok=True)

try:
    from habit.core.machine_learning.visualization.km_survival import KMSurvivalPlotter
    
    # Simple test data
    data = {
        'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'event': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B']
    }
    df = pd.DataFrame(data)
    
    print("Creating KM plotter...")
    plotter = KMSurvivalPlotter(output_dir="./test_results")
    
    print("Generating KM curve with optimized legend...")
    fig, ax = plotter.plot_km(
        df=df,
        time_col="time",
        event_col="event",
        group_col="group",
        save_name="simple_test.pdf",
        legend_loc="upper right",
        legend_ncol=1,
        legend_outside=False
    )
    
    print("Success! KM curve with optimized legend generated.")
    print("Check ./test_results/simple_test.pdf")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
