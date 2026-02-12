# Basic edge detection
aug = iaa.EdgeDetect(alpha=0.5)

# Directional edge detection (e.g., vertical edges)
aug = iaa.DirectedEdgeDetect(alpha=0.7, direction=0.25)  # 90 degrees

# Sharpening with custom strength
aug = iaa.Sharpen(alpha=0.5, lightness=2.0)