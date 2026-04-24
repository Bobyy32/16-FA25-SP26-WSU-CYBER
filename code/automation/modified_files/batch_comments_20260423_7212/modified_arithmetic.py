import numpy as np
import imgaug.augmenters as iaa

# Create an image to test (e.g., 200x200, RGB)
image = iaa.random_image(seed=42).shape  # Just get shape for example
image = np.random.rand(200, 200, 3).astype('float32')

# 1. Configure the augmenter
# compression=0 means no compression (quality 100%)
# compression=100 means max compression (quality 0%)
# Note: In some versions, the scale might be inverted or normalized differently; 
# check the imgaug documentation for the exact interpretation of 'compression'.
# Typically, higher values = lower quality.
augmentation = iaa.JpegCompression(compression=50)

# 2. Apply the augmenter
# We simulate a batch of images
batch = [image, image] 
augmented_batch = augmentation.batch_augment(batch)

# Verify the augmentation
print("Original shape:", image.shape)
print("Augmented shape:", augmented_batch[0].shape)