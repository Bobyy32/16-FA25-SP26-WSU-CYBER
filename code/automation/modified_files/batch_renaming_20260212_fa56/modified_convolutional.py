# Basic sharpening
sharpen = iaa.Sharpen(alpha=0.5)

# Emboss effect
emboss = iaa.Emboss(alpha=0.3)

# Edge detection
edges = iaa.EdgeDetect(alpha=0.7)

# Directional edge detection
dir_edges = iaa.DirectedEdgeDetect(alpha=0.5, direction=0.25)  # 90 degrees