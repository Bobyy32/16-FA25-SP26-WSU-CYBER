# Basic edge detection
aug = iaa.EdgeDetect(alpha=0.5)

# Directional edge detection (edges from top)
aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=0)

# Sharpen with random strength
aug = iaa.Sharpen(alpha=(0.0, 0.5))

# Emboss effect
aug = iaa.Emboss(alpha=0.3)