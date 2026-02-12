This code defines several image augmentation classes for the `imgaug` library, which is commonly used for data augmentation in machine learning workflows. Here's a breakdown of what each class does:

1. **`Invert` class**:
   - Inverts the colors of images (pixel value = 255 - original value)
   - Has optional parameters for `min_value`, `max_value`, and `per_channel` to control how inversion is applied
   - Includes `Invert` and `Solarize` as subclasses

2. **`JpegCompression` class**:
   - Compresses images using JPEG compression to simulate quality degradation
   - Takes a `compression` parameter (0-100 scale) where higher values mean more compression/artifacts
   - Useful for simulating real-world image quality variations

3. **`ContrastNormalization` function**:
   - **Deprecated**: This function is deprecated and redirects to `LinearContrast` in the contrast module
   - Used to adjust image contrast by scaling pixel values around the center (128)
   - Has parameters for `alpha` (contrast factor), `per_channel` (whether to apply per channel), and standard augmentation parameters

Key features of these augmentations:
- Support for stochastic parameters (random sampling)
- Per-channel processing options
- Integration with standard `imgaug` augmentation pipeline
- Proper handling of different data types and image formats

These augmentations are commonly used in computer vision tasks to increase dataset diversity and improve model robustness by simulating various real-world image conditions.