#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Advanced Pattern Analysis for Electron Microscopy Images

This script implements advanced analysis techniques for microscopy images:
1. Multi-scale denoising
2. Global FFT analysis for periodic structures
3. Windowed FFT with PCA/NMF decomposition for localized patterns

Author: Roberto dos Reis, PhD - Northwestern University/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
import os

# Scientific libraries
from scipy import ndimage, signal
from scipy.ndimage import median_filter, gaussian_filter
import skimage
from skimage import io, exposure, restoration, feature, filters, transform
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from sklearn.decomposition import PCA, NMF

# Import for DM4 files
import ncempy.io as nio


def load_dm4(file_path):
    """Load a DM4 file and return its data and metadata
    
    Parameters:
    -----------
    file_path : str
        Path to the DM4 file
        
    Returns:
    --------
    dict
        Dictionary containing data and metadata
    """
    dmData = nio.read(file_path)
    return dmData


def add_real_space_scalebar(ax, pixel_size, length_nm=100, position='lower right', color='white', size_vertical=5, font_size=12):
    """
    Add a scale bar to a real-space image
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to
    pixel_size : float
        Pixel size in nm/pixel
    length_nm : float, optional
        Length of the scale bar in nm
    position : str, optional
        Position of the scale bar ('lower right', 'lower left', 'upper right', 'upper left')
    color : str, optional
        Color of the scale bar
    size_vertical : int, optional
        Vertical size of the scale bar in pixels
    font_size : int, optional
        Font size for the scale bar label
    """
    # Convert length from nm to pixels
    length_pixels = length_nm / pixel_size
    fontprops = fm.FontProperties(size=font_size)
    scalebar = AnchoredSizeBar(ax.transData,
                              length_pixels,
                              f'{length_nm} nm',
                              position,
                              pad=0.2,
                              color=color,
                              frameon=True,
                              size_vertical=size_vertical,
                              fontproperties=fontprops)
    ax.add_artist(scalebar)


def add_fft_scalebar(ax, pixel_size, image_shape, length_nm_inv=0.1, position='lower right', color='white', size_vertical=5, font_size=12):
    """
    Add a scale bar to an FFT image (reciprocal space)
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to
    pixel_size : float
        Pixel size in nm/pixel
    image_shape : tuple
        Shape of the image (height, width)
    length_nm_inv : float, optional
        Length of the scale bar in nm^-1
    position : str, optional
        Position of the scale bar ('lower right', 'lower left', 'upper right', 'upper left')
    color : str, optional
        Color of the scale bar
    size_vertical : int, optional
        Vertical size of the scale bar in pixels
    font_size : int, optional
        Font size for the scale bar label
    """
    # Calculate the FFT scale - reciprocal of real space scale
    h, w = image_shape
    
    # The frequency spacing in the FFT is 1/N where N is the image dimension in real space units
    freq_spacing = 1.0 / (w * pixel_size)  # cycles per nm = nm^-1
    
    # Convert the desired length in nm^-1 to pixels in the FFT
    length_pixels = length_nm_inv / freq_spacing
    
    fontprops = fm.FontProperties(size=font_size)
    scalebar = AnchoredSizeBar(ax.transData,
                              length_pixels,
                              f'{length_nm_inv} nm⁻¹',
                              position,
                              pad=0.2,
                              color=color,
                              frameon=True,
                              size_vertical=size_vertical,
                              fontproperties=fontprops)
    ax.add_artist(scalebar)


class AdvancedPatternAnalysis:
    """
    Class for advanced pattern analysis of electron microscopy images
    """
    
    def __init__(self, file_path=None, image=None, pixel_size=None):
        """
        Initialize with either a file path or an image array
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the image file (supports DM4 format)
        image : ndarray, optional
            Image array if already loaded
        pixel_size : float or tuple, optional
            Pixel size in nm, can be a tuple (x, y) or single value
        """
        self.file_path = file_path
        self.image = image
        self.pixel_size = pixel_size
        self.denoised_image = None
        self.fft_result = None
        self.windowed_fft_results = None
        self.pca_components = None
        self.nmf_components = None
        
        # Load image if file path is provided
        if file_path is not None:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """
        Load image from file
        
        Parameters:
        -----------
        file_path : str
            Path to the image file
        """
        self.file_path = file_path
        
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.dm4':
            # Load DM4 file
            dm_data = load_dm4(file_path)
            self.image = dm_data['data']
            self.pixel_size = dm_data['pixelSize']
            print(f"Loaded DM4 image with shape {self.image.shape}")
            print(f"Pixel size: {self.pixel_size} nm")
        else:
            # Load using skimage for other formats
            self.image = io.imread(file_path, as_gray=True)
            print(f"Loaded image with shape {self.image.shape}")
            
            if self.pixel_size is None:
                print("Warning: Pixel size not provided. Using default value of 1.0")
                self.pixel_size = 1.0
    
    def multi_scale_denoise(self, methods=None, params=None):
        """
        Apply multi-scale denoising to the image
        
        Parameters:
        -----------
        methods : list of str, optional
            List of denoising methods to apply
            Options: 'median', 'gaussian', 'bilateral', 'nl_means', 'wavelet', 'tv'
        params : dict, optional
            Parameters for each denoising method
            
        Returns:
        --------
        denoised_image : ndarray
            The denoised image
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        if methods is None:
            methods = ['median', 'gaussian', 'wavelet']
            
        if params is None:
            params = {
                'median': {'size': 3},
                'gaussian': {'sigma': 1.0},
                'bilateral': {'sigma_color': 0.1, 'sigma_spatial': 1.0},
                'nl_means': {'patch_size': 5, 'patch_distance': 6, 'h': 0.1},
                'wavelet': {'wavelet': 'db1', 'mode': 'soft', 'wavelet_levels': 3},
                'tv': {'weight': 0.1}
            }
        
        # Start with the original image
        denoised = self.image.copy().astype(np.float32)
        
        # Apply each method in sequence
        for method in methods:
            if method == 'median':
                denoised = median_filter(denoised, size=params['median']['size'])
                
            elif method == 'gaussian':
                denoised = gaussian_filter(denoised, sigma=params['gaussian']['sigma'])
                
            elif method == 'bilateral':
                denoised = restoration.denoise_bilateral(
                    denoised, 
                    sigma_color=params['bilateral']['sigma_color'],
                    sigma_spatial=params['bilateral']['sigma_spatial']
                )
                
            elif method == 'nl_means':
                denoised = restoration.denoise_nl_means(
                    denoised,
                    patch_size=params['nl_means']['patch_size'],
                    patch_distance=params['nl_means']['patch_distance'],
                    h=params['nl_means']['h']
                )
                
            elif method == 'wavelet':
                denoised = restoration.denoise_wavelet(
                    denoised,
                    wavelet=params['wavelet']['wavelet'],
                    mode=params['wavelet']['mode'],
                    wavelet_levels=params['wavelet']['wavelet_levels']
                )
                
            elif method == 'tv':
                denoised = restoration.denoise_tv_chambolle(
                    denoised,
                    weight=params['tv']['weight']
                )
                
            else:
                print(f"Warning: Unknown denoising method '{method}'. Skipping.")
        
        # Store the denoised image
        self.denoised_image = denoised
        
        return denoised
    
    def global_fft_analysis(self, image=None, apply_window=True, window_type='hanning'):
        """
        Perform global FFT analysis to detect periodic structures
        
        Parameters:
        -----------
        image : ndarray, optional
            Image to analyze, uses denoised image if available, otherwise original
        apply_window : bool, optional
            Whether to apply a window function to reduce edge effects
        window_type : str, optional
            Type of window function to apply ('hanning', 'hamming', 'tukey')
            
        Returns:
        --------
        fft_result : dict
            Dictionary containing FFT results
        """
        if image is None:
            if self.denoised_image is not None:
                image = self.denoised_image
            elif self.image is not None:
                image = self.image
            else:
                raise ValueError("No image available for FFT analysis")
        
        # Apply window function if requested
        if apply_window:
            if window_type == 'hanning':
                window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
            elif window_type == 'hamming':
                window = np.outer(np.hamming(image.shape[0]), np.hamming(image.shape[1]))
            elif window_type == 'tukey':
                # Tukey window (tapered cosine)
                alpha = 0.5  # Tukey parameter
                window_h = signal.tukey(image.shape[0], alpha)
                window_w = signal.tukey(image.shape[1], alpha)
                window = np.outer(window_h, window_w)
            else:
                raise ValueError(f"Unknown window type: {window_type}")
                
            windowed_image = image * window
        else:
            windowed_image = image
        
        # Compute FFT
        fft = np.fft.fft2(windowed_image)
        fft_shifted = np.fft.fftshift(fft)
        
        # Compute magnitude spectrum
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log1p(magnitude)
        
        # Compute phase spectrum
        phase = np.angle(fft_shifted)
        
        # Calculate frequency axes in reciprocal space (nm^-1)
        h, w = image.shape
        pixel_size_x = self.pixel_size[0] if isinstance(self.pixel_size, (list, tuple)) else self.pixel_size
        pixel_size_y = self.pixel_size[1] if isinstance(self.pixel_size, (list, tuple)) else self.pixel_size
        
        # Calculate frequency spacing in reciprocal space (nm^-1)
        # The frequency spacing is 1/(N*pixel_size) where N is the image dimension
        freq_x = np.fft.fftshift(np.fft.fftfreq(w, d=pixel_size_x))  # nm^-1
        freq_y = np.fft.fftshift(np.fft.fftfreq(h, d=pixel_size_y))  # nm^-1
        
        # Find peaks in the magnitude spectrum
        # Exclude the DC component (center)
        center_y, center_x = h // 2, w // 2
        mask = np.ones_like(magnitude, dtype=bool)
        mask[center_y-5:center_y+5, center_x-5:center_x+5] = False
        
        # Find local maxima
        magnitude_masked = magnitude.copy()
        magnitude_masked[~mask] = 0
        
        # Use skimage's peak_local_max to find peaks
        peak_threshold = np.percentile(magnitude_masked, 99)
        peaks = feature.peak_local_max(
            magnitude_masked,
            min_distance=10,
            threshold_abs=peak_threshold,
            num_peaks=20
        )
        
        # Calculate peak properties
        peak_properties = []
        for peak in peaks:
            y, x = peak
            peak_value = magnitude[y, x]
            freq_value_x = freq_x[x]
            freq_value_y = freq_y[y]
            
            # Calculate distance from center (in frequency units)
            dist_from_center = np.sqrt(freq_value_x**2 + freq_value_y**2)
            
            # Calculate angle
            angle = np.arctan2(freq_value_y, freq_value_x) * 180 / np.pi
            
            # Calculate d-spacing (nm)
            d_spacing = 1 / dist_from_center if dist_from_center > 0 else float('inf')
            
            peak_properties.append({
                'position': (y, x),
                'frequency': (freq_value_y, freq_value_x),
                'magnitude': peak_value,
                'distance': dist_from_center,
                'angle': angle,
                'd_spacing': d_spacing
            })
        
        # Sort peaks by magnitude
        peak_properties.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Store results
        self.fft_result = {
            'fft': fft_shifted,
            'magnitude': magnitude,
            'magnitude_log': magnitude_log,
            'phase': phase,
            'freq_x': freq_x,
            'freq_y': freq_y,
            'peaks': peak_properties
        }
        
        return self.fft_result
    
    def windowed_fft_analysis(self, image=None, window_size=64, step=32, apply_window=True):
        """
        Perform windowed FFT analysis for localized pattern detection
        
        Parameters:
        -----------
        image : ndarray, optional
            Image to analyze, uses denoised image if available, otherwise original
        window_size : int, optional
            Size of the square window
        step : int, optional
            Step size for sliding the window
        apply_window : bool, optional
            Whether to apply a window function to each patch
            
        Returns:
        --------
        windowed_fft_results : dict
            Dictionary containing windowed FFT results
        """
        if image is None:
            if self.denoised_image is not None:
                image = self.denoised_image
            elif self.image is not None:
                image = self.image
            else:
                raise ValueError("No image available for windowed FFT analysis")
        
        # Pad image if needed to ensure it can be divided into windows
        h, w = image.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        
        if pad_h > 0 or pad_w > 0:
            padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        else:
            padded_image = image
        
        # Create sliding windows
        patches = view_as_windows(padded_image, (window_size, window_size), step=step)
        n_h, n_w = patches.shape[:2]
        
        # Prepare window function if needed
        if apply_window:
            window_func = np.outer(np.hanning(window_size), np.hanning(window_size))
        
        # Initialize arrays to store FFT magnitudes
        fft_magnitudes = np.zeros((n_h, n_w, window_size, window_size), dtype=np.float32)
        
        # Process each patch
        for i in range(n_h):
            for j in range(n_w):
                patch = patches[i, j].copy()
                
                if apply_window:
                    patch = patch * window_func
                
                # Compute FFT
                fft = np.fft.fftshift(np.fft.fft2(patch))
                magnitude = np.log1p(np.abs(fft))
                
                # Store magnitude
                fft_magnitudes[i, j] = magnitude
        
        # Reshape for PCA/NMF
        n_patches = n_h * n_w
        reshaped_magnitudes = fft_magnitudes.reshape(n_patches, window_size * window_size)
        
        # Store results
        self.windowed_fft_results = {
            'fft_magnitudes': fft_magnitudes,
            'reshaped_magnitudes': reshaped_magnitudes,
            'n_h': n_h,
            'n_w': n_w,
            'window_size': window_size,
            'step': step,
            'patch_positions': [
                (i * step, j * step) for i in range(n_h) for j in range(n_w)
            ]
        }
        
        return self.windowed_fft_results
    
    def decompose_patterns(self, n_components=5, method='pca'):
        """
        Decompose windowed FFT patterns using PCA or NMF
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to extract
        method : str, optional
            Decomposition method ('pca' or 'nmf')
            
        Returns:
        --------
        components : dict
            Dictionary containing decomposition results
        """
        if self.windowed_fft_results is None:
            raise ValueError("Run windowed_fft_analysis first")
        
        # Get reshaped magnitudes
        X = self.windowed_fft_results['reshaped_magnitudes']
        
        # Apply decomposition
        if method.lower() == 'pca':
            decomposer = PCA(n_components=n_components)
            components = decomposer.fit_transform(X)
            components_images = decomposer.components_.reshape(n_components, 
                                                              self.windowed_fft_results['window_size'], 
                                                              self.windowed_fft_results['window_size'])
            explained_variance = decomposer.explained_variance_ratio_
            
            self.pca_components = {
                'components': components,
                'components_images': components_images,
                'explained_variance': explained_variance,
                'n_components': n_components
            }
            result = self.pca_components
            
        elif method.lower() == 'nmf':
            decomposer = NMF(n_components=n_components, init='random', random_state=0)
            components = decomposer.fit_transform(X)
            components_images = decomposer.components_.reshape(n_components, 
                                                              self.windowed_fft_results['window_size'], 
                                                              self.windowed_fft_results['window_size'])
            
            self.nmf_components = {
                'components': components,
                'components_images': components_images,
                'n_components': n_components
            }
            result = self.nmf_components
            
        else:
            raise ValueError(f"Unknown decomposition method: {method}")
        
        return result
    
    def visualize_global_fft(self, crop_factor=0.05, save_path=None, scalebar_length_nm=100, scalebar_position='lower right', scalebar_color='white', scalebar_size_vertical=5, scalebar_font_size=12):
        """
        Visualize global FFT results
        
        Parameters:
        -----------
        crop_factor : float, optional
            Factor to crop the central portion of the FFT
        save_path : str, optional
            Path to save the figure
        scalebar_length_nm : float, optional
            Length of the scale bar in nm
        scalebar_position : str, optional
            Position of the scale bar ('lower right', 'lower left', 'upper right', 'upper left')
        scalebar_color : str, optional
            Color of the scale bar
        scalebar_size_vertical : int, optional
            Vertical size of the scale bar in pixels
        scalebar_font_size : int, optional
            Font size for the scale bar label
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        """
        if self.fft_result is None:
            raise ValueError("Run global_fft_analysis first")
            
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot original image
        if self.denoised_image is not None:
            axs[0].imshow(self.denoised_image, cmap='gray')
            axs[0].set_title('Denoised Image')
        else:
            axs[0].imshow(self.image, cmap='gray')
            axs[0].set_title('Original Image')
        
        axs[0].axis('off')
        
        # Plot full FFT
        axs[1].imshow(self.fft_result['magnitude_log'], cmap='inferno')
        axs[1].set_title('Full FFT Magnitude (log scale)')
        axs[1].axis('off')
        
        # Plot cropped FFT
        h, w = self.fft_result['magnitude_log'].shape
        center_h, center_w = h // 2, w // 2
        crop_h, crop_w = int(h * crop_factor), int(w * crop_factor)
        
        cropped_fft = self.fft_result['magnitude_log'][
            center_h - crop_h:center_h + crop_h, 
            center_w - crop_w:center_w + crop_w
        ]
        
        axs[2].imshow(cropped_fft, cmap='inferno')
        axs[2].set_title('Cropped FFT Magnitude (central region)')
        axs[2].axis('off')
        
        # Mark peaks on the full FFT (just markers, no labels)
        for peak in self.fft_result['peaks'][:10]:  # Show top 10 peaks
            y, x = peak['position']
            axs[1].plot(x, y, 'o', color='cyan', markersize=5)
        
        # Mark peaks on the cropped FFT with labels
        for peak in self.fft_result['peaks'][:10]:  # Show top 10 peaks
            y, x = peak['position']
            
            # Calculate position in cropped FFT
            cropped_x = x - (center_w - crop_w)
            cropped_y = y - (center_h - crop_h)
            
            # Check if the peak is within the cropped region
            if (0 <= cropped_x < 2*crop_w) and (0 <= cropped_y < 2*crop_h):
                # Plot marker
                axs[2].plot(cropped_x, cropped_y, 'o', color='cyan', markersize=5)
                
                # Add peak info as text
                if cropped_x < crop_w:  # Left side
                    text_x = cropped_x + 5
                    ha = 'left'
                else:  # Right side
                    text_x = cropped_x - 5
                    ha = 'right'
                    
                if cropped_y < crop_h:  # Top half
                    text_y = cropped_y + 5
                    va = 'top'
                else:  # Bottom half
                    text_y = cropped_y - 5
                    va = 'bottom'
                    
                axs[2].text(text_x, text_y, 
                           f"d={peak['d_spacing']:.2f}nm", 
                           color='white', fontsize=8, 
                           ha=ha, va=va,
                           bbox=dict(facecolor='black', alpha=0.5, pad=1))
        
        # Add scale bars
        pixel_size_val = self.pixel_size[0] if isinstance(self.pixel_size, (list, tuple)) else self.pixel_size
            
        # Add real space scale bar to the original/denoised image
        add_real_space_scalebar(axs[0], pixel_size_val, length_nm=scalebar_length_nm, position=scalebar_position, color=scalebar_color, size_vertical=scalebar_size_vertical, font_size=scalebar_font_size)
        
        # Add FFT scale bar to the full FFT
        add_fft_scalebar(axs[1], pixel_size_val, self.image.shape, length_nm_inv=0.1, position=scalebar_position, color=scalebar_color, size_vertical=scalebar_size_vertical, font_size=scalebar_font_size)
        
        # Add FFT scale bar to the cropped FFT (adjusted for cropping)
        # For the cropped FFT, we need to adjust the scale bar length based on the crop factor
        add_fft_scalebar(axs[2], pixel_size_val, self.image.shape, length_nm_inv=0.05, position=scalebar_position, color=scalebar_color, size_vertical=scalebar_size_vertical, font_size=scalebar_font_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_decomposition(self, method='pca', save_path=None):
        """
        Visualize decomposition results
        
        Parameters:
        -----------
        method : str, optional
            Decomposition method ('pca' or 'nmf')
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        """
        # Check if decomposition has been performed
        if method.lower() == 'pca' and self.pca_components is None:
            raise ValueError("Run decompose_patterns with method='pca' first")
        elif method.lower() == 'nmf' and self.nmf_components is None:
            raise ValueError("Run decompose_patterns with method='nmf' first")
            
        # Get components
        if method.lower() == 'pca':
            components = self.pca_components
            title = 'PCA Components'
        else:
            components = self.nmf_components
            title = 'NMF Components'
            
        n_components = components['n_components']
        components_images = components['components_images']
        
        # Create figure
        fig, axs = plt.subplots(2, n_components, figsize=(3*n_components, 6))
        
        # Plot component patterns (FFT space)
        for i in range(n_components):
            axs[0, i].imshow(components_images[i], cmap='inferno')
            
            if method.lower() == 'pca':
                var_explained = components['explained_variance'][i] * 100
                axs[0, i].set_title(f'Component {i+1}\n({var_explained:.1f}% var)')
            else:
                axs[0, i].set_title(f'Component {i+1}')
                
            axs[0, i].axis('off')
        
        # Plot component weights (spatial distribution)
        component_weights = components['components']
        n_h = self.windowed_fft_results['n_h']
        n_w = self.windowed_fft_results['n_w']
        
        for i in range(n_components):
            # Reshape weights to match spatial layout
            weights = component_weights[:, i].reshape(n_h, n_w)
            
            # Plot weights
            im = axs[1, i].imshow(weights, cmap='viridis')
            axs[1, i].set_title(f'Spatial Distribution {i+1}')
            axs[1, i].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axs[1, i], fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def run_full_analysis(self, denoising_methods=None, window_size=64, n_components=5, 
                         decomposition_method='pca', save_dir=None):
        """
        Run the full analysis pipeline
        
        Parameters:
        -----------
        denoising_methods : list of str, optional
            List of denoising methods to apply
        window_size : int, optional
            Size of the window for windowed FFT
        n_components : int, optional
            Number of components for decomposition
        decomposition_method : str, optional
            Decomposition method ('pca' or 'nmf')
        save_dir : str, optional
            Directory to save results
            
        Returns:
        --------
        results : dict
            Dictionary containing all results
        """
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. Denoise the image
        print("Applying multi-scale denoising...")
        self.multi_scale_denoise(methods=denoising_methods)
        
        # 2. Global FFT analysis
        print("Performing global FFT analysis...")
        self.global_fft_analysis(apply_window=True)
        
        # Save global FFT results
        if save_dir:
            global_fft_path = os.path.join(save_dir, 'global_fft_analysis.png')
            self.visualize_global_fft(save_path=global_fft_path)
            print(f"Global FFT results saved to {global_fft_path}")
        
        # 3. Windowed FFT analysis
        print(f"Performing windowed FFT analysis with window size {window_size}...")
        self.windowed_fft_analysis(window_size=window_size)
        
        # 4. Pattern decomposition
        print(f"Decomposing patterns using {decomposition_method.upper()} with {n_components} components...")
        self.decompose_patterns(n_components=n_components, method=decomposition_method)
        
        # Save decomposition results
        if save_dir:
            decomp_path = os.path.join(save_dir, f'{decomposition_method}_decomposition.png')
            self.visualize_decomposition(method=decomposition_method, save_path=decomp_path)
            print(f"Decomposition results saved to {decomp_path}")
        
        # Return all results
        return {
            'denoised_image': self.denoised_image,
            'fft_result': self.fft_result,
            'windowed_fft_results': self.windowed_fft_results,
            'pca_components': self.pca_components,
            'nmf_components': self.nmf_components
        }


# Example usage
if __name__ == "__main__":
    # Example file path
    file_path = 'selected_images/Devon_5E w UA -EtOH_20kX_0035.dm4'
    
    # Load DM4 file directly to inspect metadata
    print("\nLoading DM4 file to inspect metadata:")
    dm_data = load_dm4(file_path)
    print(f"Image shape: {dm_data['data'].shape}")
    print(f"Pixel size: {dm_data['pixelSize']} nm/pixel")
    
    # Create analyzer
    print("\nCreating analyzer and running full analysis:")
    analyzer = AdvancedPatternAnalysis(file_path)
    
    # Run global FFT analysis
    analyzer.global_fft_analysis()
    
    # Visualize with default scale bars
    print("\nVisualizing with default scale bars:")
    fig = analyzer.visualize_global_fft()
    plt.savefig('fft_default_scalebars.png', dpi=300, bbox_inches='tight')
    
    # Visualize with customized scale bars
    print("\nVisualizing with customized scale bars:")
    fig = analyzer.visualize_global_fft(
        scalebar_length_nm=200,           # Larger scale bar (200 nm instead of 100 nm)
        scalebar_position='lower left',   # Position in lower left corner
        scalebar_color='yellow',          # Yellow color for better visibility
        scalebar_size_vertical=10,        # Thicker scale bar
        scalebar_font_size=14             # Larger font size
    )
    plt.savefig('fft_custom_scalebars.png', dpi=300, bbox_inches='tight')
    
    # Run full analysis
    print("\nRunning full analysis:")
    results = analyzer.run_full_analysis(
        denoising_methods=['median', 'wavelet'],
        window_size=128,
        n_components=5,
        decomposition_method='pca',
        save_dir='analysis_results'
    )
    
    print("\nAnalysis complete!")
    print("\nTop 5 peaks found in FFT:")
    for i, peak in enumerate(results['fft_result']['peaks'][:5]):
        print(f"Peak {i+1}: d-spacing = {peak['d_spacing']:.2f} nm, angle = {peak['angle']:.1f}°, magnitude = {peak['magnitude']:.2e}")