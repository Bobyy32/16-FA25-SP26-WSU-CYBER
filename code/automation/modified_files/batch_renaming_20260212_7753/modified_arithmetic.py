# Basic contrast adjustment
aug = iaa.LinearContrast(1.5)  # Increase contrast

# Add noise
aug = iaa.AdditiveGaussianNoise(20)  # Add Gaussian noise

# JPEG compression
aug = iaa.JpegCompression(80)  # Moderate compression

# Multiple operations
aug = iaa.Sequential([
    iaa.GaussianBlur(1),
    iaa.LinearContrast(1.5),
    iaa.AdditiveGaussianNoise(10)
])