# Font Configuration for Publication Quality

## Overview

This project has been updated to use Arial font consistently across all visualizations to meet top-tier journal requirements. All plots, charts, and figures now use Arial font family for better publication compliance.

## Changes Made

### 1. Global Font Configuration Module
- **New file**: `habit/utils/font_config.py`
- Provides centralized font configuration for the entire project
- Ensures consistent Arial font usage across all visualizations
- Includes TrueType font embedding for better PDF compatibility

### 2. Updated Visualization Modules

#### Core Visualization Files:
- `habit/core/machine_learning/visualization/plotting.py` - Main plotting module
- `habit/core/machine_learning/visualization/km_survival.py` - Already had Arial (no changes needed)
- `habit/utils/visualization.py` - Cluster visualization utilities
- `habit/utils/visualization_utils.py` - General visualization utilities

#### Feature Selection Visualizations:
- `habit/core/machine_learning/feature_selectors/lasso_selector.py` - Lasso coefficient plots

#### Script Files:
- `scripts/day3_3_get_w_use_BGD.py` - Gradient descent visualization
- `scripts/day3_6_correlation.py` - Correlation analysis plots  
- `scripts/app_dilation_or_erosion.py` - Medical image processing
- `scripts/app_km_survival.py` - Already had Arial (no changes needed)

### 3. Font Configuration Details

The global configuration sets:
- **Font family**: Arial
- **Fallback fonts**: DejaVu Sans, Liberation Sans, Bitstream Vera Sans
- **Font sizes**: Optimized for publication quality
  - Base font: 12pt
  - Axis labels: 12pt  
  - Titles: 14pt
  - Figure titles: 16pt
  - Legends: 11pt
  - Tick labels: 11pt
- **PDF/PostScript**: TrueType font embedding (fonttype 42)

## Usage

### For New Visualizations
```python
from habit.utils.font_config import setup_publication_font

# At the beginning of your visualization code
setup_publication_font()

# Your plotting code here...
```

### For Individual Text Elements
```python
from habit.utils.font_config import get_font_config

config = get_font_config()
plt.xlabel('Your Label', fontfamily=config['fontfamily'])
```

### For Existing Plots
```python
from habit.utils.font_config import apply_font_to_text_elements

# After creating your plot
apply_font_to_text_elements(ax, fontfamily='Arial')
```

## Benefits

1. **Publication Compliance**: Arial font meets requirements of top-tier medical journals
2. **Consistency**: All figures in the project use the same font family
3. **Professional Appearance**: Clean, readable fonts suitable for scientific publications
4. **PDF Compatibility**: TrueType font embedding ensures proper rendering across platforms
5. **Easy Maintenance**: Centralized configuration allows easy future updates

## Files Modified

### New Files:
- `habit/utils/font_config.py` - Global font configuration module
- `doc/FONT_CONFIGURATION.md` - This documentation

### Modified Files:
- `habit/core/machine_learning/visualization/plotting.py`
- `habit/utils/visualization.py`
- `habit/utils/visualization_utils.py`
- `habit/core/machine_learning/feature_selectors/lasso_selector.py`
- `scripts/day3_3_get_w_use_BGD.py`
- `scripts/day3_6_correlation.py`
- `scripts/app_dilation_or_erosion.py`

### Files with Existing Arial Configuration (No Changes):
- `habit/core/machine_learning/visualization/km_survival.py`
- `scripts/app_km_survival.py`
- `test_legend_optimization.py`

## Testing

To verify that Arial font is being used:

1. Generate any plot using the project's visualization modules
2. Check that all text elements (titles, labels, legends) use Arial font
3. Export to PDF and verify font embedding using PDF viewers or tools

## Future Considerations

- The font configuration can be easily updated by modifying `habit/utils/font_config.py`
- Additional font properties (weight, style) can be added to the configuration as needed
- Different font configurations can be created for specific publication requirements
