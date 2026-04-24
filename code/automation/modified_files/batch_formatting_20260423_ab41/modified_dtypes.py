from imgaug.augmenters.meta import MetaAugmenters
from imgaug import augmentation as ia

# Only allow uint8 inputs
aug = ia.meta.Augmenter()
aug.add_allow_only_uint8('data')
aug.add_allow_only_uint8('data')