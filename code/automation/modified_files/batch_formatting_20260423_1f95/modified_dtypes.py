import imgaug.augmenters as ia

# Example: Use only uint8 dtypes
from imgaug.augmenters.meta import *

@ia.augmenter('my_uint8_auger', dtypes=['uint8'], augmenter=augmenter)
def my_augmentation(img):
    # Custom augmentation logic here
    pass