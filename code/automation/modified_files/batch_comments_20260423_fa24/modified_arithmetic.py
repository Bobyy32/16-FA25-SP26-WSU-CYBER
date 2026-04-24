import imgaug.augmenters as iaa

# Increase or decrease contrast per image
aug1 = iaa.ContrastNormalization(alpha=(0.5, 1.5))

# JPEG compression (current implementation)
aug2 = iaa.JpegCompression(compression=(70, 95))

# Per-channel sampling for some images (deprecated pattern)
# aug3 = iaa.ContrastNormalization(per_channel=0.5)
aug3 = iaa.JpegCompression(compression=(70, 95), per_channel=0.5)

# Use seed instead of random_state
aug_seed = iaa.JpegCompression(seed=12345)

# Use to_deterministic() for deterministic behavior
aug_det = iaa.JpegCompression(compression=80).to_deterministic()