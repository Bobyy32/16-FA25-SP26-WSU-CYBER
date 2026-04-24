# Sharpen with 50% strength
aug = iaa.Sharpen(alpha=0.5)

# Emboss with random direction
aug = iaa.Emboss(alpha=(0.1, 0.5))

# Edge detection with random blending
aug = iaa.EdgeDetect(alpha=(0.0, 0.75))

# Directed edge detection from random angle
aug = iaa.DirectedEdgeDetect(alpha=0.5, direction=(0.0, 1.0))