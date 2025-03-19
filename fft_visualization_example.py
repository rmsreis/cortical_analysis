# Example of proper sequence for FFT visualization

# Import necessary libraries
import matplotlib.pyplot as plt
from pattern_analysis import AdvancedPatternAnalysis

# Load the image
file_path = 'selected_images/Devon_5E w UA -EtOH_20kX_0035.dm4'
analyzer = AdvancedPatternAnalysis(file_path)

# IMPORTANT: Run global_fft_analysis before visualize_global_fft
analyzer.global_fft_analysis()

# Now you can visualize with customized scale bars
fig = analyzer.visualize_global_fft(
    crop_factor=0.05,
    scalebar_length_nm=200,           # Larger scale bar (200 nm instead of 100 nm)
    scalebar_position='lower left',   # Position in lower left corner
    scalebar_color='yellow',          # Yellow color for better visibility
    scalebar_size_vertical=10,        # Thicker scale bar
    scalebar_font_size=14             # Larger font size
)
plt.show()
