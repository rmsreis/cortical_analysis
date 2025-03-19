# Cortical Analysis

## Image Filtering and Visualization

This repository contains tools for processing and visualizing microscopy images, particularly focused on filtering and Fourier transform analysis of cortical structures.

### Overview

The main functionality includes:

1. **Image Loading and Filtering**: Load microscopy images and apply various filters (such as median filter) to reduce noise while preserving important features.

2. **FFT Analysis**: Compute and visualize the Fast Fourier Transform (FFT) of both original and filtered images to analyze frequency components and spatial patterns.

3. **Visualization**: Create publication-quality visualizations with proper scale bars, clear labeling, and appropriate color maps.

### Key Features

#### Image Processing
- Load microscopy images (supports DM4 format)
- Apply median filtering to reduce noise
- Preserve important structural features

#### FFT Visualization
- Compute 2D FFT of images
- Apply log scaling to enhance visibility of frequency components
- Crop FFT to focus on central features
- Display FFT with appropriate color maps

#### Scale Bars
- Add professional-looking scale bars to both real space and reciprocal space images
- Customize scale bar appearance (length, position, style)
- Properly label with appropriate units (nm for real space, nm⁻¹ for reciprocal space)

### Usage Example

```python
# Load image (assuming 'img' variable contains the image data)

# Apply median filter
img_median = median_filter(img, size=5)

# Calculate FFT
fft_original = np.fft.fftshift(np.fft.fft2(img))
fft_filtered = np.fft.fftshift(np.fft.fft2(img_median))

# Visualize with scale bars
# See notebook_image_filtering_corner.py for complete implementation
```

### Technical Notes

- Scale bars are implemented using matplotlib's Rectangle patches for clean appearance
- FFT visualization uses log scaling to enhance visibility of frequency components
- The code handles both real space (nm) and reciprocal space (nm⁻¹) units appropriately
- Pixel coordinates are preserved for measurement purposes

### Dependencies

- NumPy: For numerical operations and FFT calculations
- Matplotlib: For plotting images and visualizations
- SciPy: For image filtering (scipy.ndimage.median_filter)
- Hyperspy (optional): For loading microscopy file formats like DM4