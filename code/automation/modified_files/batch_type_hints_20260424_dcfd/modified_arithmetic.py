import imgaug.augmenters as iaa

# Deprecated version (for compatibility)
aug = iaa.ContrastNormalization(alpha=1.5)

# Updated version
aug = iaa.LinearContrast(alpha=1.5)

# JPEG Compression
aug = iaa.JpegCompression(compression=(20, 50))
aug = iaa.JpegCompression(compression=80)