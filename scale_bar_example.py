# Example code to add to your notebook for customizing scale bars

# IMPORTANT: You must run global_fft_analysis before visualize_global_fft
analyzer.global_fft_analysis()

# Option 1: Default scale bars
fig1 = analyzer.visualize_global_fft(crop_factor=0.05)
plt.show()

# Option 2: Larger yellow scale bars
fig2 = analyzer.visualize_global_fft(
    crop_factor=0.05,
    scalebar_length_nm=200,           # Larger scale bar (200 nm instead of 100 nm)
    scalebar_color='yellow',          # Yellow color for better visibility
    scalebar_size_vertical=10,        # Thicker scale bar
    scalebar_font_size=14             # Larger font size
)
plt.show()

# Option 3: Red scale bars in upper left corner
fig3 = analyzer.visualize_global_fft(
    crop_factor=0.05,
    scalebar_length_nm=150,           # Medium-sized scale bar (150 nm)
    scalebar_position='upper left',   # Position in upper left corner
    scalebar_color='red',             # Red color
    scalebar_size_vertical=8,         # Medium thickness
    scalebar_font_size=12             # Standard font size
)
plt.show()
