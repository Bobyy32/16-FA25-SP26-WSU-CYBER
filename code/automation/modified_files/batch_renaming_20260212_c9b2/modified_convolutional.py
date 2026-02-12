# Basic sharpening
aug = iaa.Sharpen(alpha=0.5)

# Edge detection with random alpha
aug = iaa.EdgeDetect(alpha=(0.0, 0.75))

# Directional edge detection
aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=0.25)  # 90 degrees

# Custom convolution
kernel = [[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]
aug = iaa.Conv2d(kernel=kernel, alpha=0.5)