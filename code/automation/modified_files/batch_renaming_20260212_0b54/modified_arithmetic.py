This code defines several image augmentation classes for the `imgaug` library, which is commonly used for data augmentation in computer vision tasks. Here's a breakdown of what each class does:

1. **`Invert` class**:
   - Inverts pixel values in images (e.g., 0 becomes 255, 255 becomes 0)
   - Supports per-channel inversion
   - Has parameters for thresholding and per-channel behavior

2. **`GammaContrast` class**:
   - Applies gamma correction to images (adjusts brightness)
   - Uses `cv2.pow` for the transformation
   - Supports per-channel gamma values
   - Has parameters for gamma range and per-channel behavior

3. **`SigmoidContrast` class**:
   - Applies sigmoid contrast adjustment
   - Uses `cv2.sigmoid` for transformation
   - Supports per-channel behavior
   - Has parameters for gain, cutoff, and per-channel behavior

4. **`LogContrast` class**:
   - Applies logarithmic contrast adjustment
   - Uses `cv2.log` for transformation
   - Supports per-channel behavior
   - Has parameters for gain and per-channel behavior

5. **`LinearContrast` class**:
   - Applies linear contrast adjustment using `cv2.multiply`
   - Supports per-channel behavior
   - Has parameters for alpha (contrast factor) and per-channel behavior

6. **`AllChannels` class**:
   - Wrapper that applies an augmenter to all channels of an image
   - Ensures all channels are processed uniformly

7. **`AllChannelsIfAllChannels` class**:
   - Special case of `AllChannels` that only applies to images with multiple channels
   - Used internally by other augmenters

8. **`Multiply` class**:
   - Multiplies pixel values by a factor
   - Supports per-channel multiplication
   - Has parameters for multiplier and per-channel behavior

9. **`Add` class**:
   - Adds a constant value to pixel values
   - Supports per-channel addition
   - Has parameters for value and per-channel behavior

10. **`AddElementwise` class**:
    - Adds random values to each pixel (element-wise)
    - Supports per-channel addition
    - Has parameters for value range and per-channel behavior

11. **`MultiplyElementwise` class**:
    - Multiplies each pixel by a random factor (element-wise)
    - Supports per-channel multiplication
    - Has parameters for multiplier range and per-channel behavior

12. **`Clip` class**:
    - Clips pixel values to a specified range
    - Supports per-channel clipping
    - Has parameters for min/max values and per-channel behavior

13. **`Alpha` class**:
    - Combines two images using an alpha blending factor
    - Supports per-channel alpha values
    - Has parameters for alpha value, per-channel behavior, and image blending

14. **`AlphaElementwise` class**:
    - Applies alpha blending with element-wise alpha values
    - Supports per-channel alpha values
    - Has parameters for alpha range, per-channel behavior, and image blending

15. **`Threshold` class**:
    - Applies thresholding to images
    - Supports per-channel thresholding
    - Has parameters for threshold value, per-channel behavior, and thresholding type

16. **`Invert` class (second definition)**:
    - Another definition of the `Invert` class with different parameters
    - Supports thresholding and per-channel behavior

17. **`Solarize` class**:
    - Applies solarization effect (inverts pixel values above a threshold)
    - Supports per-channel behavior
    - Has parameters for threshold value, per-channel behavior, and inversion direction

18. **`ContrastNormalization` function**:
    - Deprecated alias for `LinearContrast` class
    - Used to adjust image contrast using linear transformation

19. **`JpegCompression` class**:
    - Applies JPEG compression to images
    - Simulates compression artifacts by reducing image quality
    - Has parameters for compression strength (0-100)

Each class follows a consistent pattern:
- Inherits from `meta.Augmenter`
- Has `__init__` method for parameter setup
- Implements `_augment_batch_` method for the actual augmentation
- Has `get_parameters` method for introspection
- Supports various parameter types (numbers, tuples, lists, stochastic parameters)
- Handles per-channel behavior through `per_channel` parameter
- Uses OpenCV functions for the actual image transformations

The augmenters are designed to be flexible and support both fixed and random parameter values, making them suitable for various data augmentation scenarios in machine learning pipelines.