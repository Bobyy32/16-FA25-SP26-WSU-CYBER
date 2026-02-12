# Basic edge detection
aug = iaa.EdgeDetect(alpha=0.5)

# Directional edge detection (e.g., horizontal edges)
aug = iaa.DirectedEdgeDetect(alpha=0.7, direction=0)

# Custom sharpening
aug = iaa.Sharpen(alpha=0.3, lightness=1.0)

# Custom convolution with specific kernel
kernel = [[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]
aug = iaa.Convolve(matrix=kernel)