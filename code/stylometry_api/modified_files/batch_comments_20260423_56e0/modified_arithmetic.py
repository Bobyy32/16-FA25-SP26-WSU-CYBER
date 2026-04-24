import imgaug.augmenters as iaa

# Contrast augmentation (range 0.5–1.5, uniform)
aug = iaa.ContrastNormalization(alpha=(0.5, 1.5))

# JPEG compression (quality 5–70)
aug = iaa.JpegCompression(compression=(5, 70))

# Per-channel JPEG compression (50% probability)
aug = iaa.JpegCompression(per_channel=0.5)