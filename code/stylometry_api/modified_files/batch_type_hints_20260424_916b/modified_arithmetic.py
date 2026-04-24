import imgaug.augmenters as iaa
    # Old way
    iaa.ContrastNormalization(alpha=1.5)
    # New way (preferred)
    iaa.contrast.LinearContrast(alpha=1.5)