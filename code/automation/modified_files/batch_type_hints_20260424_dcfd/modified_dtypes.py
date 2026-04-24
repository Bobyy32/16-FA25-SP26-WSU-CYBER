# Example 1: Validate and clip an array
import numpy as np
from imgaug.augmenters.meta import AllowOnly

array = np.array([1.0, -0.5, 2.5, 3.0])
clip_value = clip(array, min_value=0.0, max_value=3.0, validate=2)
# clip_value = [1.0, 0.0, 2.5, 3.0]  (first two elements validated manually)

# Example 2: Dtype Gate for Augmenter
from imgaug.augmenters import Flip

flip = Flip(p=0.5, gate_dtypes="float32 float64")
img = flip(image, dtype="float32")  # Only float32/64 supported