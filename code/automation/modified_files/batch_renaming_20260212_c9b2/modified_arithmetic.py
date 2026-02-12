# Invert images
aug = iaa.Invert(p=0.5)  # 50% chance to invert

# JPEG compression
aug = iaa.JpegCompression(compression=(70, 99))  # Moderate compression

# Contrast normalization (deprecated)
aug = iaa.ContrastNormalization(alpha=(0.5, 1.5))