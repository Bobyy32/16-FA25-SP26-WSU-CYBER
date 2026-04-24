import numpy as np
import imgaug.augmenters as ia
from imgaug.augmenters.meta import Augmenter
from imgaug.core import validate_values

def apply_safe_augmentation(image):
    # Convert image to numpy array
    image_array = np.array(image, dtype=np.uint8)
    
    # Validate values are in range [0, 255]
    min_val, max_val = 0, 255
    validate_values(image_array, min_value=min_val, max_value=max_val)
    
    # Apply allowed dtype gate within an augmenter
    augmenter = ia.Sequential([ia.FlipHorizontal()])
    gate_dtypes_strs(image_array, allowed='uint8', disallowed='uint16', augmenter=augmenter)
    
    return image_array