This code defines the `HeatmapsOnImage` class, which is used to represent and manipulate heatmap data in the imgaug library. Here's a breakdown of its key components:

## Main Features

**Core Functionality:**
- Represents heatmaps as float arrays with values in [0.0, 1.0] range
- Stores metadata about the image shape and value range
- Provides methods for converting between different data types

**Key Methods:**

1. **`get_arr()`** - Returns the underlying float array with proper value range conversion
2. **`draw()`** - Visualizes heatmaps on images (returns overlayed image)
3. **`draw_on_image()`** - Draws heatmaps directly onto images
4. **`to_uint8()`** - Converts to uint8 format [0, 255]
5. **`from_uint8()`** - Creates from uint8 array
6. **`from_0to1()`** - Creates from [0.0, 1.0] float array

**Conversion Methods:**
- `change_normalization()` - Converts between different value ranges
- Supports conversion between [0.0, 1.0] and custom ranges like [-1.0, 1.0]

**Image Operations:**
- `resize()` - Resizes heatmaps to match image dimensions
- `pad()` - Pads heatmaps to specific sizes
- `crop()` - Crops heatmaps

**Utility Methods:**
- `copy()` and `deepcopy()` - For creating copies
- `__repr__()` and `__str__()` - String representations

The class is designed to work seamlessly with imgaug's image augmentation pipeline, allowing heatmaps to be transformed alongside images while maintaining proper alignment and coordinate systems.