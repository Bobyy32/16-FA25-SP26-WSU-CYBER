import imgaug.augmenters as iaa
aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=90/360)