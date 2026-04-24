# Basic edge detection
aug = iaa.EdgeDetect(alpha=0.5)

# Directional edge detection
aug = iaa.DirectedEdgeDetect(alpha=0.7, direction=0.25)  # 90 degrees

# Sharpening
aug = iaa.Sharpen(alpha=0.3, lightness=0.5)