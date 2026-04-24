# Invert an image
aug = iaa.Invert()
inverted = aug(image=image)

# Apply gamma correction
aug = iaa.GammaContrast(gamma=1.5)
gamma_corrected = aug(image=image)

# Apply contrast normalization
aug = iaa.LinearContrast(alpha=1.5)
normalized = aug(image=image)