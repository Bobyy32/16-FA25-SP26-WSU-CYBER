The provided code snippet outlines methods and functionality for a `HeatmapsOnImage` class from the `imgaug` library, primarily focused on handling heatmap data arrays with customizable value ranges. Key points include:

### **Core Functionality**
1. **Value Range Management**  
   - Defines heatmaps using `min_value` and `max_value` attributes, allowing heatmaps to represent ranges beyond `[0, 1.0]`.
   - Converts heatmaps between `[0.0, 1.0]` (float32), `[0, 255]` (uint8), and custom normalized ranges.

2. **Conversion Methods**
   - `to_uint8()`: Converts heatmap to an 8-bit array scaled to `[0, 255]`.
   - `from_uint8(arr_uint8, ...)`: Reconstructs a heatmap from an 8-bit array, scaling values to `[0.0, 1.0]`.
   - `from_0to1(arr_0to1, ...)`: Creates a heatmap object from a float array in `[0.0, 1.0]` range.

3. **Normalization Adjustment**
   - `change_normalization(arr, source, target)`: Projects heatmap values from one range `(min_source, max_source)` to another `(min_target, max_target)`.

4. **Object Handling**
   - `copy()`: Shallow copy of the heatmap object.
   - `deepcopy()`: Full copy, preserving internal array references.

5. **Shape Handling**
   - The `shape` parameter represents the image dimensions (`H, W`), independent of the heatmap array's shape (which may include channel dimensions `C`).

### **Key Notes**
- Heatmaps can be 2D (`H, W`) or 3D (`H, W, C`) arrays.
- Value normalization is applied automatically when converting between ranges.
- Helper functions allow seamless interaction between `uint8` and normalized float representations.

This design supports flexible heatmap handling for tasks like visualization, normalization, and data augmentation in computer vision workflows.