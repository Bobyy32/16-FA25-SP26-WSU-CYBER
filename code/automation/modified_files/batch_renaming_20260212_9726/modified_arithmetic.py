This code defines several image augmentation classes for the `imgaug` library, primarily focusing on arithmetic and contrast operations. Here's a breakdown of the main components:

## Key Classes

### 1. **Multiply**
- Multiplies image pixel values by a given factor
- Supports per-channel operations
- Can handle different data types (int, float)
- Uses `numpy.clip` to prevent overflow

### 2. **Add**
- Adds a constant value to pixel intensities
- Supports both constant and stochastic parameters
- Handles different data types appropriately

### 3. **AddElementwise**
- Adds random values to each pixel independently
- Supports per-channel variation
- Can use different distributions (uniform, normal, etc.)

### 4. **WithColorspace**
- Transforms images to a different colorspace for augmentation
- Applies transformations in the specified colorspace
- Returns to original colorspace after processing

### 5. **ClipValues**
- Clips pixel values to a specified range
- Works with different data types
- Supports per-channel clipping

### 6. **GammaCorrection**
- Applies gamma correction to images
- Uses `numpy.power` for the transformation
- Can be applied per channel

### 7. **SigmoidCorrection**
- Applies sigmoid correction to images
- Uses `scipy.special.expit` for the sigmoid function
- Can be applied per channel

### 8. **Invert**
- Inverts pixel values (0→255, 255→0)
- Works with different data types
- Supports per-channel operations

### 9. **MultiplyAndAddToPixelValues**
- Combines multiplication and addition operations
- Applies both transformations in sequence

### 10. **SimplexNoise**
- Adds simplex noise to images
- Uses `noise.simplex_noise` for generation
- Supports per-channel noise

### 11. **FrequencyNoise**
- Adds noise based on frequency components
- Uses FFT for noise generation
- Supports different noise distributions

### 12. **ArithmeticOperationOnChannel**
- Performs arithmetic operations on individual channels
- Allows different operations per channel

### 13. **ContrastNormalization**
- Changes image contrast using linear transformation
- Supports per-channel operations
- Uses `numpy.clip` to maintain valid pixel values

### 14. **LinearContrast**
- Another implementation of contrast normalization
- More flexible than `ContrastNormalization`
- Supports different data types

### 15. **HistogramEqualization**
- Applies histogram equalization to images
- Works with different data types
- Supports per-channel processing

### 16. **AllChannelsHistogramEqualization**
- Histogram equalization applied to all channels
- Ensures consistent processing across channels

### 17. **MultiplyElementwise**
- Multiplies each pixel by a random factor
- Supports different distributions for sampling

### 18. **WithHueAndSaturation**
- Applies transformations in HSV colorspace
- Separates hue and saturation operations
- Supports different data types

### 19. **ChangeHue**
- Changes hue values in HSV colorspace
- Works with different data types

### 20. **ChangeSaturation**
- Changes saturation values in HSV colorspace
- Supports different data types

### 21. **ChangeColorspace**
- Transforms between different colorspaces
- Supports various color conversion operations

### 22. **ChangeColorGamut**
- Changes color gamut using color transformation matrices
- Supports different color transformation methods

### 23. **ChangeColorspaceFromTo**
- Transforms between specific colorspaces
- Uses OpenCV color conversion functions

### 24. **WithColorspace**
- Wrapper that applies augmentation in a different colorspace
- Transforms to target colorspace, applies operation, transforms back

### 25. **WithHueAndSaturation**
- Wrapper for hue and saturation transformations
- Applies transformations in HSV colorspace

### 26. **ChangeHue**
- Hue transformation in HSV colorspace
- Supports different data types

### 27. **ChangeSaturation**
- Saturation transformation in HSV colorspace
- Supports different data types

### 28. **Invert**
- Inverts pixel values
- Works with different data types
- Supports per-channel operations

### 29. **GammaCorrection**
- Gamma correction transformation
- Uses `numpy.power` for computation

### 30. **SigmoidCorrection**
- Sigmoid correction transformation
- Uses `scipy.special.expit` for computation

### 31. **Multiply**
- Multiplies pixel values by a factor
- Supports per-channel operations

### 32. **Add**
- Adds constant value to pixel intensities
- Supports per-channel operations

### 33. **AddElementwise**
- Adds random values to each pixel
- Supports per-channel variation

### 34. **JpegCompression**
- Compresses images using JPEG compression
- Simulates compression artifacts
- Supports different compression levels

## Key Features

1. **Per-channel support**: Most operations can be applied per-channel
2. **Stochastic parameters**: Supports `StochasticParameter` objects for random sampling
3. **Data type handling**: Properly handles different numpy data types
4. **Colorspace transformations**: Supports various color conversions
5. **Noise generation**: Includes noise-based augmentation techniques
6. **Histogram operations**: Provides histogram equalization and related operations
7. **Gamma and sigmoid corrections**: Non-linear intensity transformations

## Usage Examples

The code includes examples showing how to use these augmentations:
- Basic usage with constant values
- Random sampling from ranges
- Per-channel operations
- Colorspace transformations
- Combined operations

The implementation is designed to be compatible with the `imgaug` library's architecture and follows its conventions for parameter handling and augmentation processes.