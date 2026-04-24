# Simple sharpening
sharpen = iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.7, 2.0))

# Directional edge detection
directed_edges = iaa.DirectedEdgeDetect(alpha=0.5, direction=0.25)  # 90 degrees

# Custom convolution
custom_conv = iaa.Conv2d(kernel=[[1, 2, 1], [2, 4, 2], [1, 2, 1]])