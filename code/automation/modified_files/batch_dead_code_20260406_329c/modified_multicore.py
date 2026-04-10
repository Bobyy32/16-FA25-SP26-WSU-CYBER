import imgaug as ia
from imgaug.augmenters import GeometricAugmenter
from imgaug.multicore import Pool

# Initialize an augmenter
augmentor = GeometricAugmenter()

# Wrap it in a pool
augment_pool = Pool([augmenter], nb_workers=4)

# Apply to a batch of images
# ... your loading logic ...
augmented_batch = augment_pool.get_batch()