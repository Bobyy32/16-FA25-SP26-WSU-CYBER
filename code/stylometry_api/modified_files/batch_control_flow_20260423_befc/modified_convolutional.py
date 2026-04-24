# Basic edge detection with random blending factor
aug = iaa.DirectedEdgeDetect(alpha=(0.5, 0.75), direction=(0.0, 1.0))

# Constant edge detection at 90 degrees (vertical edges)
aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=0.25)

# Edge detection with constant alpha and random direction
aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=(0.0, 1.0))