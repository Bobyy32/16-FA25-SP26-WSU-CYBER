import imgaug.augmenters as iaa
import numpy as np

# Example 1: Strong compression per image
aug = iaa.JpegCompression(compression=(5, 30))  # Quality 1-30
# Applies a JPEG compression with strength sampled between 5% and 30%

# Example 2: Use a stochastic parameter for dynamic compression per batch
from imgaug.augmenters.parameters import StochasticParameter
from imgaug.random import RNG

rng = RNG()
rng.set_state((0, 2))  # Set deterministic seed
stoch_comp = StochasticParameter(lambda: rng.uniform(5, 30))
aug = iaa.JpegCompression(compression=stoch_comp)

# Example 3: Batch augmentation
images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
batch = iaa.Batch(images, shape=(None, None, None))
augmented_batch = iaa.Sequential([
    iaa.Flip(horizontal=True),
    iaa.JpegCompression(compression=(20, 80)),
]).affect(batch)