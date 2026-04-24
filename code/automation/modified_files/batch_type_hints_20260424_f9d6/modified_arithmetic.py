import imgaug.augmenters as iaa

# Global contrast normalization
aug = iaa.ContrastNormalization(alpha=(0.5, 1.5))

# Per-channel contrast normalization for 50% of images
aug = iaa.ContrastNormalization(alpha=(0.5, 1.5), per_channel=0.5)