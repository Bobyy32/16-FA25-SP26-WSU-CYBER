import imgaug.augmenters as iaa

# High contrast per image
aug = iaa.ContrastNormalization((0.5, 1.5))

# Random JPEG compression
aug = iaa.JpegCompression(compression=(70, 99))