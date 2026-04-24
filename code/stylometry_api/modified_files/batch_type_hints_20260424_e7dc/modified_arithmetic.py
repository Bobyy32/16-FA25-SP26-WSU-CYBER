# Old (deprecated)
iaa.ContrastNormalization(alpha=1.2)

# New
from imgaug.augmenters.contrast import LinearContrast
LinearContrast(alpha=1.2)