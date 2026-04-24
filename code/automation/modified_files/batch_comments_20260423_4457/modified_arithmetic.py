import imgaug.augmenters as iaa

# Change contrast by a random factor between 0.5 and 1.5
aug = iaa.ContrastNormalization((0.5, 1.5))

# Apply JPEG compression with varying strength (corresponds to quality 1–30)
aug = iaa.JpegCompression(compression=(5, 20))