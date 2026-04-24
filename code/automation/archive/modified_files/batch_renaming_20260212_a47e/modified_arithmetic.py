# Simple contrast adjustment
aug = iaa.LinearContrast(1.5)

# Add noise
aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.1*255)

# JPEG compression
aug = iaa.JpegCompression(compression=(70, 99))

# Multiple augmentations
aug = iaa.Sequential([
    iaa.LinearContrast(1.5),
    iaa.AdditiveGaussianNoise(scale=0.1*255),
    iaa.JpegCompression(compression=90)
])