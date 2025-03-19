# Cortical Analysis

## Advanced Pattern Analysis for Electron Microscopy Images

This repository contains tools for processing and visualizing electron microscopy images, with a focus on pattern analysis using Fourier transform techniques. It provides a comprehensive set of functions for denoising, FFT analysis, and visualization with customizable scale bars and peak labeling.

### Overview

The main functionality includes:

1. **Image Loading and Preprocessing**: Load microscopy images (particularly DM4 format) with automatic pixel size extraction, and apply various denoising methods to enhance signal-to-noise ratio.

2. **FFT Analysis**: Compute and visualize the Fast Fourier Transform (FFT) of images to analyze frequency components and spatial patterns, with automatic peak detection and d-spacing calculation.

3. **Advanced Visualization**: Create publication-quality visualizations with customizable scale bars, intelligent peak labeling, and appropriate color maps for both real-space and frequency-space images.

### Key Components

#### `AdvancedPatternAnalysis` Class

The main class that provides a comprehensive workflow for pattern analysis:

```python
from pattern_analysis import AdvancedPatternAnalysis

# Initialize with a DM4 file
analyzer = AdvancedPatternAnalysis('your_image.dm4')

# Run the complete analysis pipeline
analyzer.run_full_analysis()
```

#### Core Functions

##### Image Loading and Preprocessing

- `load_dm4(file_path)`: Load a DM4 file and extract image data and metadata including pixel size
- `load_image(file_path)`: Load an image file and set up the analyzer
- `multi_scale_denoise(methods, params)`: Apply multiple denoising methods (median, wavelet, bilateral) with customizable parameters

##### FFT Analysis

- `global_fft_analysis(image, apply_window, window_type)`: Perform global FFT analysis on the entire image
- `windowed_fft_analysis(image, window_size, step, apply_window)`: Perform FFT analysis on local windows across the image
- `decompose_patterns(n_components, method)`: Decompose patterns using PCA or NMF

##### Visualization

- `visualize_global_fft(crop_factor, save_path, scalebar_length_nm, scalebar_position, scalebar_color, scalebar_size_vertical, scalebar_font_size)`: Visualize global FFT results with customizable scale bars and peak labels
- `visualize_decomposition(method, save_path)`: Visualize pattern decomposition results
- `add_real_space_scalebar(ax, pixel_size, length_nm, position, color, size_vertical, font_size)`: Add a scale bar to a real-space image
- `add_fft_scalebar(ax, pixel_size, image_shape, length_nm_inv, position, color, size_vertical, font_size)`: Add a scale bar to an FFT image

### Customizable Scale Bars

One of the key features is the ability to customize scale bars for both real-space and FFT images:

```python
# Customize scale bars
fig = analyzer.visualize_global_fft(
    scalebar_length_nm=200,           # Length in nm
    scalebar_position='lower left',   # Position on the image
    scalebar_color='yellow',          # Color for better visibility
    scalebar_size_vertical=10,        # Thickness
    scalebar_font_size=14             # Font size for labels
)
```

Available position options:
- `'lower right'` (default)
- `'lower left'`
- `'upper right'`
- `'upper left'`

### Peak Labels in Cropped FFT

The latest update moves peak labels from the full FFT to the cropped FFT view for better visibility:

- Cyan markers indicate peaks in both the full FFT and cropped FFT
- d-spacing labels are displayed only in the cropped FFT
- Labels are positioned intelligently based on their location in the view
- Each label shows the d-spacing value in nanometers

### Usage Example

```python
from pattern_analysis import AdvancedPatternAnalysis
import matplotlib.pyplot as plt

# Load image from DM4 file
file_path = 'your_image.dm4'
analyzer = AdvancedPatternAnalysis(file_path)

# Apply multi-scale denoising
analyzer.multi_scale_denoise(methods=['median'])

# Run global FFT analysis
analyzer.global_fft_analysis()

# Visualize with customized scale bars and peak labels in cropped FFT
fig = analyzer.visualize_global_fft(
    crop_factor=0.05,                # Crop factor for zooming in on the central region
    scalebar_length_nm=200,          # Larger scale bar (200 nm instead of 100 nm)
    scalebar_position='lower left',  # Position in lower left corner
    scalebar_color='yellow',         # Yellow color for better visibility
    scalebar_size_vertical=10,       # Thicker scale bar
    scalebar_font_size=14            # Larger font size
)
plt.show()
```

### Advanced Features

#### Multi-scale Analysis

The package supports multi-scale analysis for comprehensive pattern assessment:

- **Global FFT Analysis**: Analyze patterns across the entire image
- **Windowed FFT Analysis**: Detect local pattern variations
- **Pattern Decomposition**: Extract principal patterns using PCA or NMF

#### Automatic Peak Detection

- Automatically identifies peaks in the FFT
- Calculates d-spacing values for each peak
- Ranks peaks by intensity
- Labels the top peaks in the visualization

### Technical Notes

- Scale bars are implemented using matplotlib's Rectangle patches for clean appearance
- FFT visualization uses log scaling to enhance visibility of frequency components
- The code handles both real space (nm) and reciprocal space (nm⁻¹) units appropriately
- Pixel coordinates are preserved for measurement purposes
- Peak labels are displayed in the cropped FFT view for better visibility

### Dependencies

- **NumPy**: For numerical operations and FFT calculations
- **Matplotlib**: For plotting images and visualizations
- **SciPy**: For image filtering and signal processing
- **scikit-image**: For advanced image processing and denoising
- **PyWavelets**: For wavelet-based denoising (optional)
- **ncempy**: For loading microscopy file formats like DM4
- **matplotlib-scalebar**: For scale bar functionality

### Notebooks

- **pattern_analysis_notebook_with_peak_labels.ipynb**: Demonstrates the complete workflow with customizable scale bars and peak labels in cropped FFT
- **scale_bar_demo_updated.ipynb**: Shows various options for customizing scale bars