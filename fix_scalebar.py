#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for scale bar issues in pattern_analysis_notebook.ipynb

This script demonstrates how to properly display scale bars in Jupyter notebooks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar

# Import from pattern_analysis.py
from pattern_analysis import AdvancedPatternAnalysis, load_dm4

# File path
file_path = 'selected_images/Devon_5E w UA -EtOH_20kX_0035.dm4'

# Create analyzer
print("Creating analyzer...")
analyzer = AdvancedPatternAnalysis(file_path)

# Run global FFT analysis
print("Running global FFT analysis...")
analyzer.global_fft_analysis()

# Visualize with scale bars
print("Visualizing with scale bars...")
fig = analyzer.visualize_global_fft()
plt.savefig('global_fft_with_scalebars.png', dpi=300, bbox_inches='tight')
print("Saved figure to 'global_fft_with_scalebars.png'")

# Create a manual visualization with explicit scale bars
print("\nCreating manual visualization with explicit scale bars...")
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Display original image
axs[0].imshow(analyzer.image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

# Display FFT
axs[1].imshow(analyzer.fft_result['magnitude_log'], cmap='inferno')
axs[1].set_title('FFT Magnitude (log scale)')
axs[1].axis('off')

# Add scale bars manually
pixel_size_val = analyzer.pixel_size[0] if isinstance(analyzer.pixel_size, (list, tuple)) else analyzer.pixel_size
print(f"Pixel size: {pixel_size_val} nm/pixel")

# Real space scale bar
scalebar_length_nm = 100  # length in nm
scalebar_length_px = scalebar_length_nm / pixel_size_val  # convert to pixels
print(f"Real space scale bar: {scalebar_length_nm} nm = {scalebar_length_px:.1f} pixels")

fontprops = fm.FontProperties(size=12)
scalebar = AnchoredSizeBar(axs[0].transData,
                          scalebar_length_px,
                          f'{scalebar_length_nm} nm',
                          'lower right',
                          pad=0.2,
                          color='white',
                          frameon=True,
                          size_vertical=5,
                          fontproperties=fontprops)
axs[0].add_artist(scalebar)

# FFT scale bar
h, w = analyzer.image.shape
freq_spacing = 1.0 / (w * pixel_size_val)  # cycles per nm = nm^-1
fft_scalebar_length_nm_inv = 0.1  # length in nm^-1
fft_scalebar_length_px = fft_scalebar_length_nm_inv / freq_spacing  # convert to pixels
print(f"FFT scale bar: {fft_scalebar_length_nm_inv} nm^-1 = {fft_scalebar_length_px:.1f} pixels")

fft_scalebar = AnchoredSizeBar(axs[1].transData,
                              fft_scalebar_length_px,
                              f'{fft_scalebar_length_nm_inv} nm⁻¹',
                              'lower right',
                              pad=0.2,
                              color='white',
                              frameon=True,
                              size_vertical=5,
                              fontproperties=fontprops)
axs[1].add_artist(fft_scalebar)

plt.tight_layout()
plt.savefig('manual_scalebars.png', dpi=300, bbox_inches='tight')
print("Saved figure to 'manual_scalebars.png'")

# Try a different approach with matplotlib_scalebar
print("\nTrying matplotlib_scalebar directly...")
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(analyzer.image, cmap='gray')
ax.set_title('Using matplotlib_scalebar')
ax.axis('off')

scalebar = ScaleBar(pixel_size_val, 'nm', 
                   location='lower right',
                   box_alpha=0.5,
                   color='white',
                   font_properties={'size': 12})
ax.add_artist(scalebar)

plt.tight_layout()
plt.savefig('matplotlib_scalebar.png', dpi=300, bbox_inches='tight')
print("Saved figure to 'matplotlib_scalebar.png'")

print("\nAll done! Check the generated PNG files to see if scale bars are visible.")
