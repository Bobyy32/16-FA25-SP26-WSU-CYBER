# Basic sharpening
aug = iaa.Sharpen(alpha=0.5)

# Directional edge detection
aug = iaa.DirectedEdgeDetect(alpha=0.7, direction=0.25)  # 90 degrees

# Custom convolution
aug = iaa.Conv2D(kernel=[[1,2,1], [2,4,2], [1,2,1]], alpha=0.5)