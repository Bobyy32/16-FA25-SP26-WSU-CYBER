# Invert images
invert_aug = iaa.Invert()

# Adjust contrast
contrast_aug = iaa.ContrastNormalization(alpha=(0.5, 1.5))

# JPEG compression
jpeg_aug = iaa.JpegCompression(compression=(70, 99))