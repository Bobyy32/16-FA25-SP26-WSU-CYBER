import imgaug.augmenters as iaa

aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=0.25)  # Top edges only