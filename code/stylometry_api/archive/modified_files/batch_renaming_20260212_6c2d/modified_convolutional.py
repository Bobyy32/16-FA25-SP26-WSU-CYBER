# Basic edge detection
aug = iaa.EdgeDetect(alpha=1.0)

# Directional edge detection
aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=0)  # Top edges only

# Random angle edge detection  
aug = iaa.DirectedEdgeDetect(alpha=(0.0, 0.3), direction=(0.0, 1.0))