The provided code snippets are from the **`imgaug`** library, specifically focusing on image augmenters that modify image properties (contrast and quality).

### 1. Deprecated `ContrastNormalization`
The code shows the `ContrastNormalization` function, which has been wrapped in `@ia.deprecated("imgaug.contrast.LinearContrast")`.
*   **Functionality**: It was a wrapper around the `LinearContrast` augmenter, designed to change the contrast of images by a factor (`alpha`).
*   **Status**: Deprecated. The functionality is now handled by `imgaug.augmenters.contrast.LinearContrast`.
*   **Key Parameters**: `per_channel` (controls sampling of pixels per channel), and deprecated arguments like `random_state` which should be replaced by the standard `seed` parameter.

### 2. New `JpegCompression` Class
This class implements a new augmentation method to degrade image quality by simulating JPEG compression.
*   **Functionality**: It compresses the input images and reloads them. This process discards high-frequency components (such as sharp edges) based on the compression strength.
*   **Parameters**:
    *   `compression` (0 to 100): Higher values result in stronger compression and lower image quality.
    *   `per_channel`: Whether to compress channels separately.
*   **Implementation**: It uses `imgaug.augmenters.arithmetic.compress_jpeg` to perform the compression operation. The `_augment_batch_` method handles the processing of multiple images, while `StochasticParameter` allows for random variation in the compression strength.

In summary, this code demonstrates a transition in the `imgaug` library: the move away from the convenience wrapper `ContrastNormalization` towards the more standard `LinearContrast`, alongside the introduction of `JpegCompression` to handle JPEG quality degradation.