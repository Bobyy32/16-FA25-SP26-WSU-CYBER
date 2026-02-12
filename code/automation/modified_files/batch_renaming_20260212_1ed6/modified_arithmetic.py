# Basic usage
aug = iaa.Add((0, 50))  # Add random values between 0-50
aug = iaa.Multiply((0.5, 1.5))  # Multiply by random factor 0.5-1.5
aug = iaa.GaussianNoise(scale=0.1)  # Add Gaussian noise
aug = iaa.JpegCompression(compression=(70, 99))  # JPEG compression