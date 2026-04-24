import imgaug.augmenters as iaa

# Uniform alpha between 0.5 and 1.5
aug = iaa.ContrastNormalization((0.5, 1.5))

# 50% of images get per-channel contrast normalization
aug = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)